"""
Data augmentation visualization.
"""

import numpy as np
from .plot_engine import setup_figure, save_fig, PHASE_COLORS_LIST, annotate_info


def plot_time_shift(time_ms, original, shifted_left, shifted_right, shift_amount,
                    output_path):
    """
    Visualize time-shift augmentation.

    Parameters
    ----------
    time_ms : np.ndarray
        Time in milliseconds.
    original : np.ndarray, shape (6, T)
        Original signal.
    shifted_left : np.ndarray, shape (6, T)
        Left-shifted signal.
    shifted_right : np.ndarray, shape (6, T)
        Right-shifted signal.
    shift_amount : int
        Shift amount (in samples).
    output_path : str
        Output path.
    """
    fig, axes = setup_figure(figsize=(16, 10), nrows=2, ncols=3)
    axes = axes.reshape(2, 3)
    labels = ['I_A', 'I_B', 'I_C', 'U_A', 'U_B', 'U_C']
    units = ['A', 'A', 'A', 'kV', 'kV', 'kV']

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        color = PHASE_COLORS_LIST[idx % 3]
        ax.plot(time_ms, original[idx], color=color, linewidth=1.5,
                label='Original', alpha=0.9)
        ax.plot(time_ms, shifted_left[idx], color='#ff7f0e', linewidth=1.0,
                label=f'Left shift ({shift_amount})', alpha=0.7, linestyle='--')
        ax.plot(time_ms, shifted_right[idx], color='#2ca02c', linewidth=1.0,
                label=f'Right shift ({shift_amount})', alpha=0.7, linestyle=':')
        ax.set_xlabel('Time, ms')
        ax.set_ylabel(f'{labels[idx]}, {units[idx]}')
        ax.set_title(f'{labels[idx]} — time shift')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Augmentation: time shift (±{shift_amount} samples)',
                 fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, output_path)


def plot_noise_levels(time_ms, original, noisy_dict, output_path):
    """
    Visualize Gaussian noise added at different SNR levels.

    Parameters
    ----------
    time_ms : np.ndarray
        Time in milliseconds.
    original : np.ndarray, shape (6, T)
        Original signal.
    noisy_dict : dict
        {snr_db: signal_array} — noisy signal variants.
    output_path : str
        Output path.
    """
    n_snr = len(noisy_dict)
    # Show only IA and UA for clarity
    fig, axes = setup_figure(figsize=(16, 3 * (n_snr + 1)), nrows=n_snr + 1, ncols=2)

    labels = ['I_A', 'U_A']
    units = ['A', 'kV']
    colors_snr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Original signal (top row)
    for col, (idx, lab, unit) in enumerate(zip([0, 3], labels, units)):
        ax = axes[0, col]
        ax.plot(time_ms, original[idx], color='black', linewidth=1.5, label='Original')
        ax.set_ylabel(f'{lab}, {unit}')
        ax.set_title(f'{lab} — original signal')
        ax.legend(loc='upper right')

    # Noisy signal variants
    for row, (snr_db, noisy_sig) in enumerate(sorted(noisy_dict.items()), start=1):
        color = colors_snr[(row - 1) % len(colors_snr)]
        for col, (idx, lab, unit) in enumerate(zip([0, 3], labels, units)):
            ax = axes[row, col]
            ax.plot(time_ms, original[idx], color='black', linewidth=1.0,
                    label='Original', alpha=0.4)
            ax.plot(time_ms, noisy_sig[idx], color=color, linewidth=0.8,
                    label=f'SNR = {snr_db} dB', alpha=0.9)
            ax.set_ylabel(f'{lab}, {unit}')
            ax.set_title(f'{lab} — SNR = {snr_db} dB')
            ax.legend(loc='upper right')

    for row in range(n_snr + 1):
        axes[row, 0].set_xlabel('Time, ms')
        axes[row, 1].set_xlabel('Time, ms')

    fig.suptitle('Augmentation: Gaussian noise at different SNR levels',
                 fontsize=16, fontweight='bold', y=1.01)
    save_fig(fig, output_path)
