"""
plot_spectrum.py
----------------
Plots a spectrogram (and waveform) for an audio file, with the VAD
decision overlaid on top -- so you can visually check whether the
detected speech segments actually line up with real energy/formant
structure in the recording.
 
Usage:
    python3 plot_spectrum.py audio_samples/clean_generated_announcement.wav
    python3 plot_spectrum.py audio_samples/noise_announcement.wav --out noisy_plot.png
 
Requires: numpy, scipy, matplotlib  (pip install numpy scipy matplotlib)
"""
 
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so it saves a file even with no display; remove this
                        # line if you want plt.show() to pop up a window instead
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly

from spectral_energy_and_periodicity_vad import spectral_energy_vad
 
 
def load_wav(path, target_fs=16000):
    fs, data = wavfile.read(path)
    data = data.astype(np.float32)
    if np.max(np.abs(data)) > 1.0:
        data /= 32768.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if fs != target_fs:
        data = resample_poly(data, target_fs, fs).astype(np.float32)
    return data
 
 
def frames_to_samples(decision, hop_len, n_samples):
    mask = np.zeros(n_samples, dtype=bool)
    for i, d in enumerate(decision):
        start = i * hop_len
        end = min(n_samples, start + hop_len)
        mask[start:end] = bool(d)
    return mask
 
 
def plot_file(path, fs=16000, out=None):
    x = load_wav(path, fs)
    decision, log_energy, thr = spectral_energy_vad(x, fs)
    hop_len = int(round(fs * 10 / 1000))
    vad_mask = frames_to_samples(decision, hop_len, len(x))
 
    t = np.arange(len(x)) / fs
 
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
 
    # 1. waveform + VAD overlay
    axes[0].plot(t, x, linewidth=0.4, color="tab:blue")
    axes[0].plot(t, vad_mask * np.max(np.abs(x)) * 0.9, color="black",
                 linewidth=0.8, label="VAD decision")
    axes[0].set_title(f"Waveform + VAD decision -- {path}")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
 
    # 2. spectrogram
    axes[1].specgram(x + 1e-9 * np.random.randn(len(x)), NFFT=512, Fs=fs,
                      noverlap=256, cmap="magma")
    axes[1].set_title("Spectrogram")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].axhline(300, color="cyan", linewidth=0.6, linestyle="--")
    axes[1].axhline(3400, color="cyan", linewidth=0.6, linestyle="--",
                     label="VAD's 300-3400Hz band")
    axes[1].legend(loc="upper right", fontsize=8)
 
    # 3. per-frame band energy (dB) + threshold trace
    frame_times = np.arange(len(log_energy)) * hop_len / fs
    axes[2].plot(frame_times, log_energy, linewidth=0.7, label="band energy (dB)")
    if np.ndim(thr) > 0:
        axes[2].plot(frame_times, thr, linewidth=0.9, color="red",
                     label="adaptive threshold")
    else:
        axes[2].axhline(thr, color="red", linewidth=0.9, label="threshold")
    axes[2].set_title("Band energy vs. threshold (this is what drives the decision)")
    axes[2].set_ylabel("dB")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right", fontsize=8)
 
    plt.tight_layout()
    out_path = out or (path.rsplit(".", 1)[0] + "_spectrum.png")
    plt.savefig(out_path, dpi=130)
    print(f"saved: {out_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="path to a .wav file")
    parser.add_argument("--out", default=None, help="output image path")
    args = parser.parse_args()
    plot_file(args.audio_path, out=args.out)
 