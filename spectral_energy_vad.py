"""
Spectral-energy-based Voice Activity Detection (VAD)
Starter scaffold for the transit-announcement DSP project.

Pipeline for each frame:
    frame -> Hamming window -> FFT -> band-limited energy -> threshold -> decision

Author notes:
- Keep the frame_len / hop_len IDENTICAL to whatever the segmentation/FFT
  teammates use, so your speech/no-speech mask lines up with their frames.
- Work in dB (log energy): thresholds behave more consistently across SNRs.
"""

import numpy as np


def frame_signal(x, frame_len, hop_len):
    """Slice a 1-D signal into overlapping frames -> (num_frames, frame_len)."""
    if len(x) < frame_len:
        raise ValueError("Signal shorter than one frame.")
    num_frames = 1 + (len(x) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(num_frames)[:, None]
    return x[idx]


def spectral_energy_vad(x, fs,
                        frame_ms=25, hop_ms=10,
                        fmin=300, fmax=3400,
                        threshold_db=6.0,
                        noise_init_frames=10,
                        hangover=5):
    """
    Returns:
        decision   : bool array, True = speech frame
        log_energy : per-frame band energy in dB (useful for plotting/tuning)
        threshold  : the dB threshold that was used
    Parameters worth tuning:
        threshold_db      -> how far above the noise floor counts as speech
        hangover          -> frames of speech kept after energy drops (tail protection)
        noise_init_frames -> leading frames assumed to be noise-only
    """
    frame_len = int(round(fs * frame_ms / 1000))
    hop_len = int(round(fs * hop_ms / 1000))
    window = np.hamming(frame_len)

    frames = frame_signal(x, frame_len, hop_len) * window
    spectrum = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / fs)

    # 1) band-limited energy: only sum the speech band
    band = (freqs >= fmin) & (freqs <= fmax)
    energy = np.sum(np.abs(spectrum[:, band]) ** 2, axis=1)
    log_energy = 10.0 * np.log10(energy + 1e-10)

    # 2) adaptive threshold from the initial (assumed noise-only) frames
    noise_floor = np.median(log_energy[:noise_init_frames])
    threshold = noise_floor + threshold_db
    raw = log_energy > threshold

    # 3) hangover: extend speech regions so quiet tails aren't clipped
    decision = raw.copy()
    count = 0
    for i in range(len(decision)):
        if raw[i]:
            count = hangover
        elif count > 0:
            decision[i] = True
            count -= 1

    return decision, log_energy, threshold


# ---------------------------------------------------------------------------
# Demo / sanity check with a synthetic "announcement": a speech-like burst
# (band-limited tone) sitting inside continuous background noise.
# Replace this block with real audio via scipy.io.wavfile.read once you have it.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fs = 16000
    dur = 3.0
    t = np.arange(int(fs * dur)) / fs

    rng = np.random.default_rng(0)
    noise = 0.05 * rng.standard_normal(len(t))          # constant background
    speech = np.zeros_like(t)
    on = (t > 1.0) & (t < 2.0)                            # "announcement" 1s–2s
    speech[on] = 0.3 * np.sin(2 * np.pi * 1000 * t[on])  # 1 kHz burst in-band
    x = speech + noise

    decision, log_energy, thr = spectral_energy_vad(x, fs)

    hop_len = int(round(fs * 10 / 1000))
    frame_times = np.arange(len(decision)) * hop_len / fs
    speech_frames = frame_times[decision]

    print(f"frames: {len(decision)}   threshold: {thr:.1f} dB")
    print(f"detected speech from "
          f"{speech_frames.min():.2f}s to {speech_frames.max():.2f}s "
          f"(true burst was 1.00s–2.00s)")
