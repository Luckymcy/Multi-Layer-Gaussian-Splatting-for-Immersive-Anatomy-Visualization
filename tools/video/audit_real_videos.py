"""Read-only source video audit; sparse visual samples are not a training split."""
import argparse
import json
from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def metrics(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = ''.join(str(int(x)) for x in (small[:, 1:] > small[:, :-1]).flat)
    return {
        'laplacian_variance_full_frame': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'dark_fraction_luma_lt_20': float((gray < 20).mean()),
        'bright_low_saturation_fraction': float(((hsv[:, :, 2] > 245) & (hsv[:, :, 1] < 40)).mean()),
        'any_channel_clipped_fraction': float((rgb.max(-1) >= 250).mean()),
        'dhash': dhash,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=12)
    args = ap.parse_args()
    videos = sorted(p for p in args.input.iterdir() if p.suffix.lower() in
                    {'.mov', '.mp4', '.avi', '.mkv', '.wmv', '.m4v', '.webm'})
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    except OSError:
        font = ImageFont.load_default()
    report = []
    for index, path in enumerate(videos, 1):
        vid = f'V{index:02d}'
        cap = cv2.VideoCapture(str(path.resolve()))
        if not cap.isOpened():
            report.append({'id': vid, 'path': str(path.resolve()), 'error': 'cannot_open'})
            continue
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        item = {'id': vid, 'path': str(path.resolve()), 'filename': path.name,
                'bytes': path.stat().st_size, 'frame_count': count, 'fps_reported': fps,
                'duration_seconds_estimate': count / fps if fps else None,
                'width_reported': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height_reported': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'codec_fourcc': ''.join(chr((fourcc >> (8 * i)) & 255) for i in range(4)),
                'orientation_meta': cap.get(cv2.CAP_PROP_ORIENTATION_META),
                'patient_id': None, 'patient_group': 'UNVERIFIED_SHARED_GROUP',
                'split': 'unassigned', 'samples': []}
        folder = args.output / vid
        folder.mkdir()
        tile_w, tile_h = 420, 280
        sheet = Image.new('RGB', (tile_w * 3, tile_h * math.ceil(args.samples / 3) + 38), '#151515')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 7), f'{vid}: {path.name}', font=font, fill='white')
        for j, fraction in enumerate(np.linspace(.03, .97, args.samples)):
            target = min(count - 1, int(fraction * count))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok:
                item['samples'].append({'frame_requested': target, 'decode_error': True})
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = target / fps if fps else None
            filename = f'sample_{j:02d}_frame_{target:07d}.png'
            image = Image.fromarray(rgb)
            image.save(folder / filename)
            preview = image.copy()
            preview.thumbnail((tile_w - 8, tile_h - 30))
            x, y = (j % 3) * tile_w, (j // 3) * tile_h + 38
            sheet.paste(preview, (x + (tile_w - preview.width) // 2, y))
            draw.text((x + 6, y + tile_h - 27), f'{j:02d}  {timestamp:.2f}s  frame {target}', font=font, fill='white')
            item['samples'].append({'sample': j, 'frame_requested': target,
                                    'timestamp_seconds_estimate': timestamp,
                                    'decoded_width': image.width, 'decoded_height': image.height,
                                    'image': f'{vid}/{filename}', **metrics(rgb)})
        cap.release()
        sheet.save(args.output / f'{vid}_contact.jpg', quality=92)
        report.append(item)
        print(json.dumps({k: v for k, v in item.items() if k != 'samples'}, ensure_ascii=False), flush=True)
    (args.output / 'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
