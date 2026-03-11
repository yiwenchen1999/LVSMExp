"""
Plot degradation metrics (PSNR, LPIPS, SSIM) for 3 methods:
- pixel-space (NeuralGaffer): degredation_data/iterative_metrics.csv
- pixel-space (Ours): degredation_data/degrade_detail_image_space.csv
- token-space (Ours): degredation_data/degrade_detail_token_space.csv

Step (x-axis) is clipped at 10.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEG_DIR = os.path.join(BASE_DIR, 'degredation_data')
MAX_STEP = 10

# =============================================================
# Parse degrade_detail_* CSV (scene_name, step, psnr, ssim, lpips)
# Exclude scenes with inf PSNR
# =============================================================
def parse_detail_csv(filepath, metric='psnr'):
    bad_scenes = set()
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p = float(row['psnr'])
            except (ValueError, TypeError):
                p = float('inf')
            if np.isinf(p) or (isinstance(p, float) and p > 80):
                bad_scenes.add(row['scene_name'])

    steps = defaultdict(list)
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['scene_name'] in bad_scenes:
                continue
            s = int(row['step'])
            if s <= MAX_STEP:
                steps[s].append(float(row[metric]))

    sorted_steps = sorted(steps.keys())
    avg = np.array([np.mean(steps[s]) for s in sorted_steps])
    return np.array(sorted_steps), avg


# =============================================================
# Parse iterative_metrics CSV (object, view, step, envmap, psnr, ssim, lpips)
# Exclude rows with inf PSNR
# =============================================================
def parse_iterative_csv(filepath, metric='psnr'):
    steps = defaultdict(list)
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p = float(row['psnr'])
            except (ValueError, TypeError):
                continue
            if np.isinf(p) or (isinstance(p, float) and p > 80):
                continue
            s = int(row['step'])
            if s <= MAX_STEP:
                steps[s].append(float(row[metric]))

    sorted_steps = sorted(steps.keys())
    avg = np.array([np.mean(steps[s]) for s in sorted_steps])
    return np.array(sorted_steps), avg


color_neural = '#808080'
color_pixel = '#2b6cb5'
color_token = '#cc2a1a'
lw = 2.8
ms = 7


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, labelsize=12)
    ax.tick_params(axis='both', which='minor', length=0)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())


def load_all():
    neural_csv = os.path.join(DEG_DIR, 'iterative_metrics.csv')
    img_csv = os.path.join(DEG_DIR, 'degrade_detail_image_space.csv')
    tok_csv = os.path.join(DEG_DIR, 'degrade_detail_token_space.csv')

    neural_steps, neural_psnr = parse_iterative_csv(neural_csv, 'psnr')
    neural_, neural_lpips = parse_iterative_csv(neural_csv, 'lpips')
    neural_, neural_ssim = parse_iterative_csv(neural_csv, 'ssim')

    img_steps, img_psnr = parse_detail_csv(img_csv, 'psnr')
    img_, img_lpips = parse_detail_csv(img_csv, 'lpips')
    img_, img_ssim = parse_detail_csv(img_csv, 'ssim')

    tok_steps, tok_psnr = parse_detail_csv(tok_csv, 'psnr')
    tok_, tok_lpips = parse_detail_csv(tok_csv, 'lpips')
    tok_, tok_ssim = parse_detail_csv(tok_csv, 'ssim')

    return {
        'neural': (neural_steps, neural_psnr, neural_lpips, neural_ssim),
        'img': (img_steps, img_psnr, img_lpips, img_ssim),
        'tok': (tok_steps, tok_psnr, tok_lpips, tok_ssim),
    }


data = load_all()

# =============================================================
# PSNR plot
# =============================================================
fig1, ax1 = plt.subplots(figsize=(7, 2.8))
all_vals = []
for key, (steps, psnr, _, _) in data.items():
    if len(steps) > 0:
        all_vals.extend(psnr)

y_min = max(15, int(np.floor(np.min(all_vals)) - 1)) if all_vals else 15
y_max = min(35, int(np.ceil(np.max(all_vals)) + 1)) if all_vals else 32
y_ticks_p = list(range(y_min, y_max + 1, 2))
for yval in y_ticks_p:
    ax1.axhline(y=yval, color='#cccccc', linewidth=0.8, linestyle='--', zorder=0)

ax1.plot(*data['neural'][:2], color=color_neural, linewidth=lw, marker='s',
         markersize=ms, markeredgecolor='white', markeredgewidth=0.8,
         label='Pixel space (NeuralGaffer)', zorder=2)
ax1.plot(*data['img'][:2], color=color_pixel, linewidth=lw, marker='o',
         markersize=ms, markeredgecolor='white', markeredgewidth=0.8,
         label='Pixel space (Ours)', zorder=3)
ax1.plot(*data['tok'][:2], color=color_token, linewidth=lw, marker='o',
         markersize=ms, markeredgecolor='white', markeredgewidth=0.8,
         label='Token space (Ours)', zorder=4)

ax1.set_xlim(-0.3, MAX_STEP + 0.3)
ax1.set_ylim(y_min - 0.5, y_max + 0.5)
ax1.set_xticks(range(0, MAX_STEP + 1))
ax1.set_xticklabels([str(i) for i in range(0, MAX_STEP + 1)], fontsize=12, fontweight='bold')
ax1.set_yticks(y_ticks_p)
ax1.set_yticklabels([str(t) for t in y_ticks_p], fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Progressive Edits', fontsize=13, fontweight='bold', labelpad=6)
ax1.set_ylabel('PSNR (dB) \u2191', fontsize=13, fontweight='bold', labelpad=6)
style_ax(ax1)
ax1.legend(loc='lower left', fontsize=10, frameon=True, framealpha=0.9,
           edgecolor='#cccccc', fancybox=True, borderpad=0.5, handlelength=1.8)
fig1.tight_layout()
out_psnr = os.path.join(DEG_DIR, 'progressive_psnr.png')
fig1.savefig(out_psnr, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved {out_psnr}")
plt.close(fig1)

# =============================================================
# LPIPS plot
# =============================================================
fig2, ax2 = plt.subplots(figsize=(7, 2.8))
all_lpips = []
for key, (steps, _, lpips, _) in data.items():
    if len(steps) > 0:
        all_lpips.extend(lpips)
l_min = max(0.02, float(np.floor(np.min(all_lpips) * 50) / 50)) if all_lpips else 0.02
l_max = min(0.30, float(np.ceil(np.max(all_lpips) * 20) / 20)) if all_lpips else 0.25
y_ticks_l = np.linspace(l_min, l_max, 6)
for yval in y_ticks_l:
    ax2.axhline(y=yval, color='#cccccc', linewidth=0.8, linestyle='--', zorder=0)

ax2.plot(data['neural'][0], data['neural'][2], color=color_neural, linewidth=lw, marker='s',
         markersize=ms, markeredgecolor='white', markeredgewidth=0.8,
         label='Pixel space (NeuralGaffer)', zorder=2)
ax2.plot(data['img'][0], data['img'][2], color=color_pixel, linewidth=lw, marker='o',
         markersize=ms, markeredgecolor='white', markeredgewidth=0.8,
         label='Pixel space (Ours)', zorder=3)
ax2.plot(data['tok'][0], data['tok'][2], color=color_token, linewidth=lw, marker='o',
         markersize=ms, markeredgecolor='white', markeredgewidth=0.8,
         label='Token space (Ours)', zorder=4)

ax2.set_xlim(-0.3, MAX_STEP + 0.3)
ax2.set_ylim(l_min - 0.01, l_max + 0.01)
ax2.set_xticks(range(0, MAX_STEP + 1))
ax2.set_xticklabels([str(i) for i in range(0, MAX_STEP + 1)], fontsize=12, fontweight='bold')
ax2.set_yticks(y_ticks_l)
ax2.set_yticklabels([f'{t:.2f}' for t in y_ticks_l], fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Progressive Edits', fontsize=13, fontweight='bold', labelpad=6)
ax2.set_ylabel('LPIPS \u2193', fontsize=13, fontweight='bold', labelpad=6)
style_ax(ax2)
ax2.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9,
           edgecolor='#cccccc', fancybox=True, borderpad=0.5, handlelength=1.8)
fig2.tight_layout()
out_lpips = os.path.join(DEG_DIR, 'progressive_lpips.png')
fig2.savefig(out_lpips, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved {out_lpips}")
plt.close(fig2)

# =============================================================
# SSIM plot
# =============================================================
fig3, ax3 = plt.subplots(figsize=(7, 2.8))
all_ssim = []
for key, (steps, _, _, ssim) in data.items():
    if len(steps) > 0:
        all_ssim.extend(ssim)
s_min = max(0.7, float(np.floor(np.min(all_ssim) * 20) / 20)) if all_ssim else 0.7
s_max = min(1.0, float(np.ceil(np.max(all_ssim) * 20) / 20)) if all_ssim else 1.0
y_ticks_s = np.linspace(s_min, s_max, 6)
for yval in y_ticks_s:
    ax3.axhline(y=yval, color='#cccccc', linewidth=0.8, linestyle='--', zorder=0)

ax3.plot(data['neural'][0], data['neural'][3], color=color_neural, linewidth=lw, marker='s',
         markersize=ms, markeredgecolor='white', markeredgewidth=0.8,
         label='Pixel space (NeuralGaffer)', zorder=2)
ax3.plot(data['img'][0], data['img'][3], color=color_pixel, linewidth=lw, marker='o',
         markersize=ms, markeredgecolor='white', markeredgewidth=0.8,
         label='Pixel space (Ours)', zorder=3)
ax3.plot(data['tok'][0], data['tok'][3], color=color_token, linewidth=lw, marker='o',
         markersize=ms, markeredgecolor='white', markeredgewidth=0.8,
         label='Token space (Ours)', zorder=4)

ax3.set_xlim(-0.3, MAX_STEP + 0.3)
ax3.set_ylim(s_min - 0.02, s_max + 0.02)
ax3.set_xticks(range(0, MAX_STEP + 1))
ax3.set_xticklabels([str(i) for i in range(0, MAX_STEP + 1)], fontsize=12, fontweight='bold')
ax3.set_yticks(y_ticks_s)
ax3.set_yticklabels([f'{t:.2f}' for t in y_ticks_s], fontsize=12, fontweight='bold')
ax3.set_xlabel('Number of Progressive Edits', fontsize=13, fontweight='bold', labelpad=6)
ax3.set_ylabel('SSIM \u2191', fontsize=13, fontweight='bold', labelpad=6)
style_ax(ax3)
ax3.legend(loc='lower left', fontsize=10, frameon=True, framealpha=0.9,
           edgecolor='#cccccc', fancybox=True, borderpad=0.5, handlelength=1.8)
fig3.tight_layout()
out_ssim = os.path.join(DEG_DIR, 'progressive_ssim.png')
fig3.savefig(out_ssim, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved {out_ssim}")
plt.close(fig3)

print("Done")
