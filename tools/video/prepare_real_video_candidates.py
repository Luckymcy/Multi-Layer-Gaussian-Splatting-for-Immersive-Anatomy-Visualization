"""Extract explicit review times without changing source videos or assigning splits."""
import argparse
import hashlib
import json
from pathlib import Path
import math
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_real_videos import metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    inventory = {x['id']: x for x in json.loads(args.inventory.read_text(encoding='utf-8'))}
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'frames').mkdir()
    shutil.copyfile(args.plan, args.output / 'sampling_plan.json')
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    except OSError:
        font = ImageFont.load_default()
    rows = []
    for vid, times in plan['timestamps_seconds'].items():
        item = inventory[vid]
        cap = cv2.VideoCapture(item['path'])
        sheet = Image.new('RGB', (420 * 3, 280 * math.ceil(len(times) / 3) + 38), '#151515')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 7), f'{vid}: {item["filename"]}', font=font, fill='white')
        for index, seconds in enumerate(times):
            frame = round(seconds * item['fps_reported'])
            if frame < 0 or frame >= item['frame_count']:
                raise ValueError(f'Invalid time {vid} {seconds}')
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f'Decode failed {vid} {seconds}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            key = f'{vid}_{index:02d}_{seconds:06.1f}s'
            rel = f'frames/{key}.png'
            image = Image.fromarray(rgb)
            image.save(args.output / rel)
            preview = image.copy()
            preview.thumbnail((412, 250))
            x, y = index % 3 * 420, index // 3 * 280 + 38
            sheet.paste(preview, (x + (420 - preview.width) // 2, y))
            draw.text((x + 6, y + 253), f'{index:02d}  {seconds:.1f}s', font=font, fill='white')
            rows.append({'id': key, 'video_id': vid, 'source_path': item['path'],
                         'frame_index_requested': frame, 'timestamp_seconds_estimate': frame / item['fps_reported'],
                         'image': rel, 'image_sha256': hashlib.sha256((args.output / rel).read_bytes()).hexdigest(),
                         'width': image.width, 'height': image.height,
                         'patient_id': item.get('patient_id'),
                         'patient_group': item.get('patient_group', 'UNVERIFIED_SHARED_GROUP'),
                         'surgery_id': item.get('surgery_id'),
                         'source_confirmation': item.get('source_confirmation'),
                         'split': 'unassigned', 'tissue_label': None,
                         'tissue_hypothesis': 'fat_like_surface_candidate',
                         'annotation_status': 'needs_tissue_and_exclusion_masks',
                         'training_ready': False, **metrics(rgb)})
        cap.release()
        sheet.save(args.output / f'{vid}_dense_contact.jpg', quality=92)
        print(vid, len(times), flush=True)
    (args.output / 'candidate_manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
