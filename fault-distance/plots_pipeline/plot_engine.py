"""
Base visualization engine for the thesis.
Matplotlib settings: English labels, DPI=300, engineering style.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Font setup
# ---------------------------------------------------------------------------
_font_ok = False
for _family in ['Times New Roman', 'DejaVu Serif', 'serif']:
    try:
        plt.rcParams['font.family'] = _family
        plt.rcParams['axes.unicode_minus'] = False
        _font_ok = True
        break
    except Exception:
        continue

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
PHASE_COLORS = {
    'A': '#E6B800',   # yellow — phase A (GOST standard)
    'B': '#2CA02C',   # green — phase B
    'C': '#D62728',   # red — phase C
}

PHASE_COLORS_LIST = [PHASE_COLORS['A'], PHASE_COLORS['B'], PHASE_COLORS['C']]

COLOR_TRAIN = '#1f77b4'
COLOR_VAL   = '#ff7f0e'
COLOR_BEST  = '#2ca02c'
COLOR_ERROR = '#d62728'
COLOR_RESID = '#9467bd'


def ensure_dir(path: str):
    """Create the parent directory for a file if it does not exist."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


def save_fig(fig, output_path: str):
    """Save the figure at DPI=300 with a white background."""
    ensure_dir(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Saved: {output_path}")
    plt.close(fig)


def setup_figure(figsize=(14, 10), nrows=1, ncols=1):
    """Create a figure with a white background."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor('white')
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes)
    return fig, axes


def annotate_info(ax, text: str, loc='upper right'):
    """Add a text annotation to the plot."""
    if loc == 'upper right':
        ax.text(0.98, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    elif loc == 'upper left':
        ax.text(0.02, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    elif loc == 'lower right':
        ax.text(0.98, 0.05, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))


def add_vertical_line(ax, x, color='gray', linestyle='--', alpha=0.7, label=None):
    """Add a vertical line."""
    ax.axvline(x, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.5, label=label)
