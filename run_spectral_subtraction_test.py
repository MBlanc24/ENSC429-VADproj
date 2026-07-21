"""
run_spectral_subtraction_test.py
---------------------------------
End-to-end test: VAD -> spectral subtraction, on the real transit
announcement audio samples. Mirrors run_test.py's structure/output style.

For each test case, saves an enhanced .wav next to the input (suffixed
_enhanced.wav) and prints:
  - segmental SNR estimate before/after (speech frames vs. noise frames,
    per the VAD's own decision -- so this measures internal consistency,
    not a ground-truth SNR)
  - RMS energy in the frames the VAD flagged as noise, before/after
    (this is the number that should drop the most -- it's the noise
    the algorithm is actually suppressing)

Just run:
    python3 run_spectral_subtraction_test.py

Requires: numpy, scipy  (pip install numpy scipy)
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from spectral_energy_and_periodicity_vad import spectral_energy_vad
from spectral_subtraction import spectral_subtraction
from run_test import load_wav, mix_at_snr  # reuse team's loader/mixer

FS = 16000
FRAME_MS, HOP_MS = 25, 10


def save_wav(path, x, fs=FS):
    peak = np.max(np.abs(x)) + 1e-12
    if peak > 0.98:
        x = x / peak * 0.98
    wavfile.write(path, fs, (x * 32767).astype(np.int16))


def segmental_energy_db(x, decision, hop_len, want_speech):
    """Mean per-frame dB energy over frames matching the VAD label."""
    mask = decision if want_speech else ~decision
    if not np.any(mask):
        return float("nan")
    frame_len = int(round(FS * FRAME_MS / 1000))
    n = len(x)
    energies = []
    for i in np.where(mask)[0]:
        start = i * hop_len
        seg = x[start:start + frame_len]
        if len(seg) == 0:
            continue
        energies.append(np.mean(seg ** 2))
    energies = np.array(energies) + 1e-12
    return 10 * np.log10(np.mean(energies))


def run_case(label, x):
    hop_len = int(round(FS * HOP_MS / 1000))
    decision, _, _ = spectral_energy_vad(x, FS, frame_ms=FRAME_MS, hop_ms=HOP_MS)
    enhanced, diag = spectral_subtraction(
        x, FS, vad_decision=decision, frame_ms=FRAME_MS, hop_ms=HOP_MS,
        alpha=2.0, adaptive_alpha=True, alpha_min=1.0, alpha_max=5.0, beta=0.02,
    )

    noise_before = segmental_energy_db(x, decision, hop_len, want_speech=False)
    noise_after = segmental_energy_db(enhanced, decision, hop_len, want_speech=False)
    speech_before = segmental_energy_db(x, decision, hop_len, want_speech=True)
    speech_after = segmental_energy_db(enhanced, decision, hop_len, want_speech=True)

    print(f"\n--- {label} ---")
    print(f"speech frames: {np.sum(decision)}/{len(decision)}  "
          f"({100 * np.mean(decision):.1f}% flagged speech)")
    print(f"noise-frame energy:  {noise_before:6.1f} dB -> {noise_after:6.1f} dB "
          f"(reduced {noise_before - noise_after:.1f} dB)")
    print(f"speech-frame energy: {speech_before:6.1f} dB -> {speech_after:6.1f} dB "
          f"(changed {speech_before - speech_after:+.1f} dB)")
    print(f"implied SNR (speech-noise): "
          f"{speech_before - noise_before:.1f} dB -> {speech_after - noise_after:.1f} dB")
    print(f"alpha used: {diag['alpha_used'].min():.2f} - {diag['alpha_used'].max():.2f}")

    return enhanced


if __name__ == "__main__":
    # Test 1: clean-ish announcement + real background noise at a moderate SNR
    speech = load_wav("audio_samples/clean_generated_announcement.wav", FS)
    noise = load_wav("audio_samples/backgroundnoise.wav", FS)
    noisy_5db = mix_at_snr(speech, noise, snr_db=5)
    enhanced = run_case("announcement + real background noise @5dB SNR", noisy_5db)
    save_wav("audio_samples/synthetic_5dB_enhanced.wav", enhanced)

    # Test 2: harder case, 0dB SNR
    noisy_0db = mix_at_snr(speech, noise, snr_db=0)
    enhanced = run_case("announcement + real background noise @0dB SNR", noisy_0db)
    save_wav("audio_samples/synthetic_0dB_enhanced.wav", enhanced)

    # Test 3: genuine noisy field recording, no synthetic mixing
    field = load_wav("audio_samples/Noise_announcement.wav", FS)
    enhanced = run_case("Noise_announcement.wav (real noisy field recording)", field)
    save_wav("audio_samples/Noise_announcement_enhanced.wav", enhanced)

    print("\nEnhanced .wav files saved to audio_samples/ -- listen to them "
          "alongside the originals. Expect: noticeably lower hiss/rumble in "
          "gaps between words; if it sounds hollow/watery ('musical noise'), "
          "raise beta or lower alpha_max in spectral_subtraction() and re-run.")
