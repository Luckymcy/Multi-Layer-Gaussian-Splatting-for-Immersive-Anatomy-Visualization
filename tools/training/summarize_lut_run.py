"""Plot a completed or live LUT run without importing Torch or changing inputs."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Run directory containing artifacts/')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    source, out = run/'artifacts', run/'reports'
    rows = []
    for line in (source/'metrics.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # The final line can be incomplete while a live run is writing.
    if not rows:
        raise ValueError('No complete metric rows')
    out.mkdir(exist_ok=True)
    steps = np.array([r['step'] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for axis, key, label in zip(axes, ['d_loss', 'g_adversarial', 'lut_gradient_norm'],
                                ['Discriminator loss', 'LUT adversarial loss', 'LUT gradient norm']):
        values = np.array([r[key] for r in rows])
        axis.plot(steps, values, alpha=.25, linewidth=.6, label='per step')
        window = min(50, len(values))
        if window > 1:
            axis.plot(steps[window-1:], np.convolve(values, np.ones(window)/window, mode='valid'),
                      linewidth=1.2, label=f'{window}-step mean')
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
        axis.legend(loc='upper left', fontsize=8)
    axes[-1].set_xlabel('Completed optimization step')
    fig.suptitle('L7 LUT training diagnostics (not validation of realism)')
    fig.savefig(out/'training_curves.png', dpi=160)
    plt.close(fig)

    curves_path = source/'lut_curves.json'
    if not curves_path.exists():
        curves_path = source/'lut_curves_latest.json'
    curves = json.loads(curves_path.read_text(encoding='utf-8'))
    hu = np.asarray(curves['hu'])
    initial, final = np.asarray(curves['initial_linear_rgb']), np.asarray(curves['final_linear_rgb'])
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for channel, color, name in zip(range(3), ['red', 'green', 'blue'], ['R', 'G', 'B']):
        ax.plot(hu[0], initial[0, :, channel], color=color, linestyle='--', alpha=.5, label=f'{name} initial')
        ax.plot(hu[0], final[0, :, channel], color=color, marker='o', markersize=3, label=f'{name} learned')
    ax.set(xlabel='HU (fixed control-point positions)', ylabel='Linear RGB', ylim=(-.02, 1.02),
           title=f'L7 LUT controls at checkpoint step {curves.get("step", int(steps[-1]))}')
    ax.grid(alpha=.2)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(out/'lut_controls.png', dpi=160)
    plt.close(fig)

    preview = source/'previews'
    starts = sorted(preview.glob('view_*_start.png'))
    pairs = []
    for start in starts:
        view = re.search(r'view_(\d+)_start', start.stem).group(1)
        final_path = preview/f'view_{view}_final.png'
        if not final_path.exists():
            available = sorted(preview.glob(f'step_*_view_{view}.png'))
            if not available:
                continue
            final_path = available[-1]
        pairs.append((view, start, final_path))
    if pairs:
        size, gap, head = 512, 32, 40
        canvas = Image.new('RGB', (size*2, head+len(pairs)*(size+gap)), '#151515')
        try:
            font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
        except OSError:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 10), 'Start', font=font, fill='white')
        draw.text((size+15, 10), 'Learned / latest checkpoint', font=font, fill='white')
        for row, (view, before, after) in enumerate(pairs):
            y = head+row*(size+gap)
            for column, path in enumerate([before, after]):
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    image.thumbnail((size, size), Image.Resampling.LANCZOS)
                    canvas.paste(image, (column*size+(size-image.width)//2, y))
            draw.text((15, y+size+4), f'Training view {view}', font=font, fill='white')
        canvas.save(out/'fixed_view_comparison.jpg', quality=94)
    summary = {'metric_rows': len(rows), 'latest_step': int(steps[-1]),
               'lut_checkpoint_step': curves.get('step', int(steps[-1])),
               'reports': ['training_curves.png', 'lut_controls.png', 'fixed_view_comparison.jpg'],
               'status': 'complete' if (source/'report.json').exists() else 'running',
               'note': 'Training diagnostics and sampled training views, not an independent evaluation.'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
