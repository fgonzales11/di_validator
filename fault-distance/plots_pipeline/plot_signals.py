"""
Oscillogram and spectrum plots.
"""

import numpy as np
from .plot_engine import setup_figure, save_fig, PHASE_COLORS_LIST, annotate_info


def plot_oscillogram_6ch(time_ms, signals, channel_labels, units, title,
                         output_path, t0_ms=None):
    """
    Plot 6 oscillogram channels in a 2×3 grid.

    Parameters
    ----------
    time_ms : np.ndarray, shape (T,)
        Time in milliseconds.
    signals : np.ndarray, shape (6, T)
        Signals: [IA, IB, IC, UA, UB, UC].
    channel_labels : list[str]
        Channel labels, e.g. ['I_A', 'I_B', 'I_C', 'U_A', 'U_B', 'U_C'].
    units : list[str]
        Measurement units, e.g. ['A', 'A', 'A', 'kV', 'kV', 'kV'].
    title : str
        Figure title.
    output_path : str
        Output PNG path.
    t0_ms : float, optional
        Fault inception time in milliseconds, marked with a vertical line.
    """
    fig, axes = setup_figure(figsize=(16, 10), nrows=2, ncols=3)
    axes = axes.reshape(2, 3)

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        color = PHASE_COLORS_LIST[idx % 3]
        ax.plot(time_ms, signals[idx], color=color, linewidth=1.2, label=channel_labels[idx])
        ax.set_xlabel('Time, ms')
        ax.set_ylabel(f'{channel_labels[idx]}, {units[idx]}')
        ax.set_title(f'{channel_labels[idx]}')
        ax.legend(loc='upper right')
        if t0_ms is not None:
            ax.axvline(t0_ms, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f't₀ = {t0_ms:.1f} ms')
            ax.legend(loc='upper right')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, output_path)


def plot_fft_spectrum(time, signal, fs, title, output_path, color=None):
    """
    Plot the one-sided amplitude spectrum of a signal.

    Parameters
    ----------
    time : np.ndarray
        Time axis (for reference).
    signal : np.ndarray, shape (T,)
        One-dimensional signal.
    fs : float
        Sampling frequency [Hz].
    title : str
        Title.
    output_path : str
        Output path.
    color : str, optional
        Line color.
    """
    fig, axes = setup_figure(figsize=(12, 5), nrows=1, ncols=2)
    ax_time = axes[0]
    ax_freq = axes[1]

    # Time-domain plot
    time_ms = np.arange(len(signal)) / fs * 1000.0
    ax_time.plot(time_ms, signal, color=color or '#1f77b4', linewidth=1.0)
    ax_time.set_xlabel('Time, ms')
    ax_time.set_ylabel('Amplitude')
    ax_time.set_title('Time-domain signal')

    # FFT
    N = len(signal)
    yf = np.fft.rfft(signal)
    xf = np.fft.rfftfreq(N, 1.0 / fs)
    amplitude = np.abs(yf) / N
    amplitude[1:-1] *= 2
    # In dB relative to the maximum
    amp_db = 20 * np.log10(amplitude + 1e-12)
    amp_db -= amp_db.max()

    ax_freq.plot(xf, amp_db, color=color or '#1f77b4', linewidth=1.2)
    ax_freq.set_xlabel('Frequency, Hz')
    ax_freq.set_ylabel('Amplitude, dB')
    ax_freq.set_title('Amplitude spectrum')
    ax_freq.set_xlim(0, min(fs / 2, 500))  # show frequencies up to 500 Hz
    ax_freq.axvline(50, color='red', linestyle='--', linewidth=1.0, alpha=0.6, label='50 Hz')
    ax_freq.axvline(100, color='orange', linestyle=':', linewidth=1.0, alpha=0.5, label='100 Hz')
    ax_freq.legend(loc='upper right')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    save_fig(fig, output_path)


def plot_channel_comparison(time_ms, original, processed, title, ylabel,
                            output_path, label_orig='Original', label_proc='Processed',
                            color_orig='#1f77b4', color_proc='#ff7f0e'):
    """
    Compare one channel: original vs. processed.
    The top panel shows both signals; the bottom panel shows their difference.
    """
    fig, axes = setup_figure(figsize=(14, 8), nrows=2, ncols=1)
    ax_top = axes[0]
    ax_bot = axes[1]

    ax_top.plot(time_ms, original, color=color_orig, linewidth=1.2, label=label_orig, alpha=0.9)
    ax_top.plot(time_ms, processed, color=color_proc, linewidth=1.2, label=label_proc, alpha=0.9)
    ax_top.set_ylabel(ylabel)
    ax_top.set_title(title)
    ax_top.legend(loc='upper right')

    diff = processed - original
    ax_bot.plot(time_ms, diff, color='#2ca02c', linewidth=1.0, label='Difference (processed − original)')
    ax_bot.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax_bot.set_xlabel('Time, ms')
    ax_bot.set_ylabel(f'Δ {ylabel}')
    ax_bot.set_title('Signal difference')
    ax_bot.legend(loc='upper right')

    # RMS of the difference
    rms_diff = np.sqrt(np.mean(diff ** 2))
    annotate_info(ax_bot, f'RMS of the difference = {rms_diff:.4f}', loc='upper right')

    save_fig(fig, output_path)


def plot_multi_channel_overlay(time_ms, signals, labels, colors, title,
                               output_path, ylabel='Amplitude'):
    """
    Overlay multiple channels on one plot.
    """
    fig, axes = setup_figure(figsize=(14, 6), nrows=1, ncols=1)
    ax = axes[0]

    for sig, lab, col in zip(signals, labels, colors):
        ax.plot(time_ms, sig, color=col, linewidth=1.2, label=lab, alpha=0.85)

    ax.set_xlabel('Time, ms')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    save_fig(fig, output_path)
