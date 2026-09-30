"""
Visualization of oscillogram preprocessing stages.
"""

import numpy as np
from scipy import signal as scipy_signal
from .plot_engine import setup_figure, save_fig, PHASE_COLORS_LIST, annotate_info, add_vertical_line


def plot_dc_removal(time_ms, sig_before, sig_after, fs, output_path):
    """
    Visualize removal of the aperiodic (DC) component.

    Parameters
    ----------
    time_ms : np.ndarray
        Time in milliseconds.
    sig_before : np.ndarray, shape (6, T)
        Signals before processing.
    sig_after : np.ndarray, shape (6, T)
        Signals after remove_dc_period.
    fs : float
        Sampling frequency [Hz].
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
        ax.plot(time_ms, sig_before[idx], color=color, linewidth=1.0,
                label='Original', alpha=0.6)
        ax.plot(time_ms, sig_after[idx], color=color, linewidth=1.5,
                label='After DC removal', linestyle='-')
        ax.set_xlabel('Time, ms')
        ax.set_ylabel(f'{labels[idx]}, {units[idx]}')
        ax.set_title(f'{labels[idx]} — DC component removal')
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Stage 1: Aperiodic (DC) component removal', fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, output_path)


def plot_t0_detection(time_ms, current, d4, delta_i, t0_idx, fs, output_path):
    """
    Visualize fault inception detection (t0).

    Parameters
    ----------
    time_ms : np.ndarray
        Time in milliseconds.
    current : np.ndarray, shape (T,)
        Phase A current (or total current).
    d4 : np.ndarray, shape (T,)
        Fourth-order difference D4(k).
    delta_i : np.ndarray, shape (T,)
        Cycle-difference index Delta_i(k).
    t0_idx : int
        Fault inception sample index.
    fs : float
        Sampling frequency [Hz].
    output_path : str
        Output path.
    """
    fig, axes = setup_figure(figsize=(14, 10), nrows=3, ncols=1)
    ax_current = axes[0]
    ax_d4 = axes[1]
    ax_delta = axes[2]

    t0_ms = time_ms[t0_idx] if t0_idx < len(time_ms) else time_ms[-1]

    # Current with a t0 marker
    ax_current.plot(time_ms, current, color='#1f77b4', linewidth=1.2)
    ax_current.axvline(t0_ms, color='red', linestyle='--', linewidth=2.0, alpha=0.8,
                       label=f't₀ = {t0_ms:.1f} ms')
    ax_current.set_ylabel('I_A, A')
    ax_current.set_title('Phase A current with fault inception marker')
    ax_current.legend(loc='upper left')

    # D4(k)
    ax_d4.plot(time_ms, np.abs(d4), color='#9467bd', linewidth=1.0)
    ax_d4.axvline(t0_ms, color='red', linestyle='--', linewidth=2.0, alpha=0.8)
    ax_d4.set_ylabel('|D₄(k)|')
    ax_d4.set_title('Fourth-order difference D₄(k) — coarse detection')

    # Delta_i(k)
    ax_delta.plot(time_ms, delta_i, color='#2ca02c', linewidth=1.0)
    ax_delta.axvline(t0_ms, color='red', linestyle='--', linewidth=2.0, alpha=0.8)
    ax_delta.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax_delta.set_xlabel('Time, ms')
    ax_delta.set_ylabel('Δ_i(k), A')
    ax_delta.set_title('Cycle-difference index Δ_i(k) — refined detection')

    annotate_info(ax_current, f'Sample t₀: {t0_idx}', loc='upper right')

    fig.suptitle('Stage 3: Fault inception detection (t₀)', fontsize=16, fontweight='bold', y=1.01)
    save_fig(fig, output_path)


def plot_symseq(time_ms, symseq, output_path):
    """
    Visualize symmetrical components (6 channels).

    Parameters
    ----------
    time_ms : np.ndarray
        Time in milliseconds.
    symseq : np.ndarray, shape (6, T)
        [|I1|, |I2|, |I0|, |U1|, |U2|, |U0|].
    output_path : str
        Output path.
    """
    fig, axes = setup_figure(figsize=(16, 10), nrows=2, ncols=3)
    axes = axes.reshape(2, 3)
    labels = ['|I₁|', '|I₂|', '|I₀|', '|U₁|', '|U₂|', '|U₀|']
    units = ['A', 'A', 'A', 'kV', 'kV', 'kV']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        ax.plot(time_ms, symseq[idx], color=colors[idx], linewidth=1.2)
        ax.set_xlabel('Time, ms')
        ax.set_ylabel(f'{labels[idx]}, {units[idx]}')
        ax.set_title(f'{labels[idx]} — symmetrical component')

    fig.suptitle('Stage 4: Symmetrical components (Fortescue)', fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, output_path)


def plot_normalization(time_ms, sig_before, sig_after, method, output_path):
    """
    Visualize normalization.

    Parameters
    ----------
    time_ms : np.ndarray
        Time in milliseconds.
    sig_before : np.ndarray, shape (6, T)
        Before normalization.
    sig_after : np.ndarray, shape (6, T)
        After normalization.
    method : str
        'standard' or 'minmax'.
    output_path : str
        Output path.
    """
    fig, axes = setup_figure(figsize=(16, 10), nrows=2, ncols=3)
    axes = axes.reshape(2, 3)
    labels = ['I_A', 'I_B', 'I_C', 'U_A', 'U_B', 'U_C']
    units_before = ['A', 'A', 'A', 'kV', 'kV', 'kV']
    units_after = ['a.u.', 'a.u.', 'a.u.', 'a.u.', 'a.u.', 'a.u.']

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        color = PHASE_COLORS_LIST[idx % 3]
        ax.plot(time_ms, sig_before[idx], color=color, linewidth=1.0,
                label=f'Original ({units_before[idx]})', alpha=0.6)
        ax.plot(time_ms, sig_after[idx], color=color, linewidth=1.5,
                label=f'Normalized ({units_after[idx]})', linestyle='-')
        ax.set_xlabel('Time, ms')
        ax.set_ylabel(f'{labels[idx]}')
        ax.set_title(f'{labels[idx]} — normalization ({method})')
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(f'Stage 5: Signal normalization (method: {method})', fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, output_path)
