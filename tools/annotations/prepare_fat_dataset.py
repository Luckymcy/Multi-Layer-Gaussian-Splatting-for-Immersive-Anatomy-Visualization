"""Convert Labelme annotations into auditable masks and unmodified RGB patches.

Only Pillow and NumPy are required. No Labelme internal API is used.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, __version__ as PIL_VERSION


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rasterize(shapes, size):
    """Union shapes within a class; exclude always wins regardless of draw order."""
    masks = {k: Image.new('L', size, 0) for k in ('fat', 'exclude')}
    counts = Counter()
    for shape in shapes:
        label, kind = shape.get('label'), shape.get('shape_type', 'polygon')
        if label not in masks:
            raise ValueError(f'Unknown label: {label!r}')
        points = np.asarray(shape.get('points', []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Invalid shape coordinates')
        if ((points < 0).any() or (points[:, 0] > size[0]).any()
                or (points[:, 1] > size[1]).any()):
            raise ValueError('Shape coordinates outside image')
        draw = ImageDraw.Draw(masks[label])
        if kind == 'polygon':
            if len(points) < 3:
                raise ValueError('Polygon requires >=3 points')
            x, y = points.T
            if abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) < 1:
                raise ValueError('Degenerate polygon')
            draw.polygon([tuple(p) for p in points], fill=255)
        elif kind == 'circle':
            if len(points) != 2:
                raise ValueError('Circle requires center and circumference point')
            x, y = points[0]
            radius = float(np.linalg.norm(points[1] - points[0]))
            if radius <= 0:
                raise ValueError('Degenerate circle')
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
        elif kind == 'rectangle':
            if len(points) != 2 or np.any(points[0] == points[1]):
                raise ValueError('Rectangle requires two nondegenerate opposite corners')
            lo, hi = points.min(0), points.max(0)
            draw.rectangle(tuple(lo) + tuple(hi), fill=255)
        else:
            raise ValueError(f'Unsupported shape type: {kind!r}')
        counts[f'{label}:{kind}'] += 1
    fat, exclude = (np.asarray(masks[k]) > 0 for k in ('fat', 'exclude'))
    return fat, exclude, fat & ~exclude, dict(counts)


def binary_image(mask):
    return Image.fromarray(mask.astype(np.uint8) * 255, mode='L')


def sampling_mask(rgb, valid, margin, highlight_threshold, highlight_min_channel, highlight_dilate):
    """Keep manual masks intact; derived quality exclusion is saved separately."""
    safe = valid.copy()
    if margin:
        padded = np.pad(safe, margin, constant_values=False)
        eroded = np.asarray(binary_image(padded).filter(ImageFilter.MinFilter(2 * margin + 1))) > 0
        safe = eroded[margin:-margin, margin:-margin]
    bright = np.zeros(valid.shape, dtype=bool)
    if highlight_threshold is not None:
        bright = (rgb.max(-1) >= highlight_threshold) & (rgb.min(-1) >= highlight_min_channel) & valid
        if highlight_dilate:
            bright = np.asarray(binary_image(bright).filter(ImageFilter.MaxFilter(2 * highlight_dilate + 1))) > 0
    return safe & ~bright, bright


def patch_positions(mask, size, stride):
    """Integral image accepts only squares with 100% valid pixels."""
    height, width = mask.shape
    if size > min(height, width):
        return []
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    xs = np.unique(np.append(np.arange(0, width-size+1, stride), width-size))
    ys = np.unique(np.append(np.arange(0, height-size+1, stride), height-size))
    xx, yy = np.meshgrid(xs, ys)
    area = integral[yy+size, xx+size] - integral[yy, xx+size] - integral[yy+size, xx] + integral[yy, xx]
    return [(int(x), int(y), int(x+size), int(y+size)) for x, y in zip(xx[area == size*size], yy[area == size*size])]


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.


def thumbnail(rgb):
    return np.asarray(Image.fromarray(rgb).resize((16, 16), Image.Resampling.BOX), dtype=np.float32)


def near_duplicate(candidate, accepted, threshold):
    return any(float(np.abs(candidate - other).mean()) <= threshold for other in accepted)


def load_inputs(seed):
    source_manifest = seed / 'manifest.json'
    rows = json.loads(source_manifest.read_text(encoding='utf-8'))
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate source IDs')
    inputs = []
    annotation_paths = sorted((seed / 'annotations').glob('*.json'))
    if not annotation_paths:
        raise ValueError('No annotation JSON found')
    for annotation in annotation_paths:
        if annotation.stem not in by_id:
            raise ValueError(f'Unknown annotated frame: {annotation.name}')
        row = by_id[annotation.stem]
        image = (seed / row['image']).resolve()
        if seed not in image.parents:
            raise ValueError('Image path escapes seed directory')
        data = json.loads(annotation.read_text(encoding='utf-8'))
        linked_image = (annotation.parent / data['imagePath'].replace('\\', '/')).resolve()
        if linked_image != image:
            raise ValueError(f'Annotation points to a different image: {annotation.name}')
        if sha256(image) != row['image_sha256']:
            raise ValueError(f'Image changed since source manifest: {image.name}')
        with Image.open(image) as opened:
            if opened.mode != 'RGB':
                raise ValueError(f'Expected RGB original; explicit color handling required: {image}')
            rgb = np.asarray(opened).copy()
        if (data['imageWidth'], data['imageHeight']) != (rgb.shape[1], rgb.shape[0]):
            raise ValueError(f'Annotation size mismatch: {annotation.name}')
        fat, exclude, valid, counts = rasterize(data['shapes'], (rgb.shape[1], rgb.shape[0]))
        if not valid.any():
            raise ValueError(f'No valid tissue pixels: {annotation.name}')
        inputs.append({'row': row, 'image': image, 'annotation': annotation,
                       'image_sha256': sha256(image), 'annotation_sha256': sha256(annotation),
                       'rgb': rgb, 'fat': fat, 'exclude': exclude, 'valid': valid, 'counts': counts})
    return inputs, source_manifest, [r['id'] for r in rows if r['id'] not in {p.stem for p in annotation_paths}]


def overlay(rgb, valid, exclude, bright, safe):
    out = rgb.astype(np.float32)
    for mask, color in [(valid, (30, 220, 80)), (exclude, (255, 30, 30)), (bright, (245, 30, 240))]:
        out[mask] = .65*out[mask] + .35*np.asarray(color)
    boundary = safe & ~(np.asarray(binary_image(safe).filter(ImageFilter.MinFilter(3))) > 0)
    out[boundary] = (40, 230, 255)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def contact_sheet(items, path, font, tile=(420, 280), columns=3):
    width, height = tile
    rows = max(1, math.ceil(len(items)/columns))
    sheet = Image.new('RGB', (width*columns, height*rows), '#161616')
    draw = ImageDraw.Draw(sheet)
    for i, (picture, label) in enumerate(items):
        preview = picture.copy()
        preview.thumbnail((width-8, height-32), Image.Resampling.LANCZOS)
        x, y = i % columns * width, i // columns * height
        sheet.paste(preview, (x+(width-preview.width)//2, y))
        draw.text((x+5, y+height-28), label, font=font, fill='white')
    sheet.save(path, quality=93)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 96, 128])
    parser.add_argument('--per-frame-per-size', type=int, default=4)
    parser.add_argument('--margin', type=int, default=4)
    parser.add_argument('--highlight-threshold', type=int, default=250)
    parser.add_argument('--highlight-min-channel', type=int, default=220)
    parser.add_argument('--highlight-dilate', type=int, default=1)
    parser.add_argument('--no-highlight-filter', action='store_true')
    parser.add_argument('--max-iou', type=float, default=.15)
    parser.add_argument('--dedup-mae', type=float, default=3.)
    parser.add_argument('--seed', type=int, default=20261109)
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) < 16 or len(set(args.sizes)) != len(args.sizes):
        parser.error('Use distinct patch sizes >=16')
    if args.per_frame_per_size < 1 or args.margin < 0 or args.highlight_dilate < 0:
        parser.error('Invalid count or mask margin')
    if not 0 <= args.max_iou <= 1 or args.dedup_mae < 0:
        parser.error('Invalid overlap/dedup setting')
    if not 0 <= args.highlight_min_channel <= args.highlight_threshold <= 255:
        parser.error('Invalid highlight settings')
    seed, output = args.seed_dir.resolve(), args.output.resolve()
    if seed == output or seed in output.parents or output in seed.parents:
        parser.error('Output must be separate from annotation seed')
    if output.exists():
        parser.error('Output exists; choose a new version, never overwrite')
    inputs, source_manifest, unannotated = load_inputs(seed)
    source_manifest_hash = sha256(source_manifest)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ['fat', 'exclude', 'valid', 'quality_exclude', 'sampling_valid']:
        (output / 'masks' / folder).mkdir(parents=True)
    (output / 'review/overlays').mkdir(parents=True)
    for size in args.sizes:
        (output / 'patches' / f'{size}px').mkdir(parents=True)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    except OSError:
        font = ImageFont.load_default()
    rng = np.random.default_rng(args.seed)
    accepted_thumbnails = {size: [] for size in args.sizes}
    exact_seen = set()
    frames, patches, counters = [], [], Counter()
    overlay_items, sample_items = [], []
    for item in inputs:
        row, rgb = item['row'], item['rgb']
        frame_id = row['id']
        safe, bright = sampling_mask(rgb, item['valid'], args.margin,
                                    None if args.no_highlight_filter else args.highlight_threshold,
                                    args.highlight_min_channel, args.highlight_dilate)
        mask_values = {'fat': item['fat'], 'exclude': item['exclude'], 'valid': item['valid'],
                       'quality_exclude': bright, 'sampling_valid': safe}
        mask_paths = {}
        for name, mask in mask_values.items():
            rel = f'masks/{name}/{frame_id}.png'
            binary_image(mask).save(output / rel)
            mask_paths[name] = rel
        marked = overlay(rgb, item['valid'], item['exclude'], bright, safe)
        marked.save(output / f'review/overlays/{frame_id}.jpg', quality=94)
        overlay_items.append((marked, frame_id))
        boxes, frame_samples, per_size = [], [], {}
        for size in sorted(args.sizes, reverse=True):
            candidates = patch_positions(safe, size, max(1, size//2))
            rng.shuffle(candidates)
            kept = 0
            for box in candidates:
                if kept >= args.per_frame_per_size:
                    break
                if any(box_iou(box, old) > args.max_iou for old in boxes):
                    counters['overlap_rejected'] += 1
                    continue
                x0, y0, x1, y1 = box
                crop = rgb[y0:y1, x0:x1].copy()
                key = hashlib.sha256(crop.tobytes()).hexdigest()
                small = thumbnail(crop)
                if key in exact_seen:
                    counters['exact_duplicate_rejected'] += 1
                    continue
                if near_duplicate(small, accepted_thumbnails[size], args.dedup_mae):
                    counters['near_duplicate_rejected'] += 1
                    continue
                rel = f'patches/{size}px/{frame_id}_x{x0:04d}_y{y0:04d}_{size}px.png'
                Image.fromarray(crop).save(output / rel)
                # Output is RGB source pixels; no resized/masked/normalized training image.
                record = {'id': Path(rel).stem, 'image': rel, 'image_sha256': sha256(output/rel),
                          'frame_id': frame_id, 'video_id': row['video_id'],
                          'patient_group': row['patient_group'], 'surgery_id': row.get('surgery_id'),
                          'split': 'unassigned', 'tissue_label': 'fat',
                          'annotation_source': 'user_labelme', 'review_status': 'pending_visual_acceptance',
                          'source_image': str(item['image']), 'source_annotation': str(item['annotation']),
                          'source_video': row['source_path'], 'source_frame_index': row['frame_index_requested'],
                          'source_timestamp_seconds_estimate': row['timestamp_seconds_estimate'],
                          'size': size, 'bbox_xyxy_exclusive': list(box), 'valid_fraction': 1.,
                          'manual_excluded_fraction': 0., 'quality_excluded_fraction': 0.,
                          'source_image_sha256': item['image_sha256'],
                          'source_annotation_sha256': item['annotation_sha256']}
                patches.append(record)
                boxes.append(box)
                exact_seen.add(key)
                accepted_thumbnails[size].append(small)
                preview = Image.fromarray(crop)
                frame_samples.append((preview, f'{frame_id} {size}px'))
                kept += 1
            per_size[str(size)] = {'grid_candidates': len(candidates), 'patches_saved': kept}
        # At most one example per size/frame on the compact QA sheet.
        for size in args.sizes:
            representative = next((v for v in frame_samples if v[1].endswith(f' {size}px')), None)
            if representative:
                sample_items.append(representative)
        frames.append({'id': frame_id, 'source_image': str(item['image']),
                       'source_annotation': str(item['annotation']), 'shape_counts': item['counts'],
                       'source_image_sha256': item['image_sha256'],
                       'source_annotation_sha256': item['annotation_sha256'],
                       'mask_paths': mask_paths, 'mask_pixel_counts': {k: int(v.sum()) for k,v in mask_values.items()},
                       'patch_counts': per_size, 'patient_group': row['patient_group'],
                       'review_status': 'pending_visual_acceptance'})
        print(frame_id, 'patches:', sum(v['patches_saved'] for v in per_size.values()), flush=True)
    if not patches:
        raise RuntimeError('No safe patches found; inspect masks before changing thresholds')
    contact_sheet(overlay_items, output/'review/mask_overview.jpg', font)
    contact_sheet(sample_items, output/'review/patch_examples.jpg', font, tile=(300, 190))
    for item in inputs:
        if sha256(item['image']) != item['image_sha256'] or sha256(item['annotation']) != item['annotation_sha256']:
            raise RuntimeError('Input changed while preparing dataset')
    if sha256(source_manifest) != source_manifest_hash:
        raise RuntimeError('Source manifest changed')
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update({'tool_sha256': sha256(Path(__file__)), 'python': platform.python_version(),
                   'numpy': np.__version__, 'pillow': PIL_VERSION})
    summary = {'scope': 'Single-surgery fat patch candidates from user annotations',
               'annotated_frames': len(frames), 'unannotated_frame_ids': unannotated,
               'shape_counts': dict(sum((Counter(f['shape_counts']) for f in frames), Counter())),
               'patches': len(patches), 'patches_by_size': dict(Counter(str(p['size']) for p in patches)),
               'rejections': dict(counters), 'patient_groups': sorted({p['patient_group'] for p in patches}),
               'split': 'unassigned', 'training_ready': False,
               'review_status': 'pending_visual_acceptance', 'inputs_unchanged': True,
               'note': 'Masks encode supplied labels, not clinical tissue verification. Bright-pixel filter is a heuristic. '
                       'Original RGB pixels are retained; PNG decoding/color-space handling must be explicit in training. '
                       'Patch count is not an independent-patient sample count.'}
    for name, obj in [('config.json', config), ('summary.json', summary),
                      ('frame_manifest.json', frames), ('patch_manifest.json', patches)]:
        (output/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'COMPLETED.json').write_text(json.dumps({'inputs_unchanged': True,
        'frames': len(frames), 'patches': len(patches)}, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
