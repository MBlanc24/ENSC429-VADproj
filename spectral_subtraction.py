"""
Spectral Subtraction speech enhancement.

Pipeline for each frame:
    frame -> Hamming window -> FFT -> magnitude subtraction -> IFFT -> overlap-add

Design choices (read this before tuning anything):

- Frame/hop are 25ms/10ms by default, matching spectral_energy_vad.py and
  spectral_energy_and_periodicity_vad.py. Keeping these identical means a
  VAD decision array lines up 1:1 with these STFT frames -- no resampling
  of the mask needed.

- Noise spectrum estimate: instead of assuming the first N frames are
  noise-only (fragile -- fails if the clip opens mid-speech), this takes
  a VAD decision array and averages the magnitude spectrum over every
  frame the VAD marked as non-speech. That's the whole point of doing
  VAD first: it tells spectral subtraction *where* to learn the noise
  from. If no VAD decision is given, it falls back to the first
  noise_init_frames frames (same fallback style as spectral_energy_vad.py).

- Oversubtraction + spectral floor (Berouti et al., 1979): straight
  magnitude subtraction (mag - noise_mag) leaves "musical noise" --
  isolated spectral peaks left over from imperfect noise estimation that
  sound like random tones/chirps. Two standard fixes, both included:
    alpha (oversubtraction factor) -- subtract more than the raw noise
      estimate to push residual peaks below the floor. Optionally made
      frequency/frame-dependent: lower segmental SNR frames get a larger
      alpha, since the noise estimate is trusted less there.
    beta (spectral floor) -- never let a bin drop to exactly zero; floor
      it at beta * original magnitude. Zero-ed bins are what create the
      "musical" tonal artifacts, so a small floor (0.001-0.05) keeps a bit
      of the original bin as noise-like residual instead.

- Phase: reuses the *noisy* signal's phase, unmodified. This is standard
  for spectral subtraction -- the ear is far more sensitive to magnitude
  errors than phase errors at these SNRs, so estimating phase isn't worth
  the complexity.

- Reconstruction: weighted overlap-add (WOLA), i.e. accumulate windowed
  IFFT frames and divide by the summed window-squared envelope. This
  works for ANY frame/hop combination (not just the 50%-overlap case
  where plain OLA is exact), which matters here since 25ms/10ms is 60%
  overlap, not 50%.
"""

import numpy as np

from spectral_energy_vad import frame_signal


def _oversubtraction_alpha(frame_snr_db, alpha_min=1.0, alpha_max=5.0,
                            snr_low_db=-5.0, snr_high_db=20.0):
    """Frequency-flat, per-frame oversubtraction factor that scales with
    segmental SNR: noisier frames (low SNR) get pushed toward alpha_max,
    cleaner frames relax toward alpha_min. Linear interpolation in dB,
    clipped at the ends. This is the frame-adaptive version of the fixed
    alpha Berouti originally proposed (alpha was fixed at ~3-6 in the
    1979 paper); making it SNR-dependent reduces over-subtraction (and
    the resulting hollow/whispery sound) on frames that are already
    fairly clean."""
    snr = np.clip(frame_snr_db, snr_low_db, snr_high_db)
    frac = (snr_high_db - snr) / (snr_high_db - snr_low_db)  # 1 at low SNR
    return alpha_min + frac * (alpha_max - alpha_min)


def _overlap_add(frames_time, hop_len, window, n_samples):
    """Weighted overlap-add reconstruction from a (num_frames, frame_len)
    array of time-domain frames that have already been windowed once on
    the analysis side. Re-windows on the synthesis side with the SAME
    window and normalizes by the summed window^2 envelope, which gives
    correct reconstruction for arbitrary hop/window combinations (not
    just the 50%-overlap case)."""
    num_frames, frame_len = frames_time.shape
    out = np.zeros(n_samples + frame_len, dtype=np.float64)
    norm = np.zeros(n_samples + frame_len, dtype=np.float64)

    for i in range(num_frames):
        start = i * hop_len
        seg = frames_time[i] * window
        out[start:start + frame_len] += seg
        norm[start:start + frame_len] += window ** 2

    norm = np.where(norm < 1e-8, 1e-8, norm)
    out = out / norm
    return out[:n_samples]


def spectral_subtraction(x, fs,
                          vad_decision=None,
                          frame_ms=25, hop_ms=10,
                          noise_init_frames=10,
                          alpha=2.0,
                          adaptive_alpha=True,
                          alpha_min=1.0, alpha_max=5.0,
                          beta=0.02,
                          noise_update=False,
                          noise_update_rate=0.05):
    """
    Enhance a noisy signal with magnitude spectral subtraction.

    Parameters
    ----------
    x : 1-D float array, the noisy signal (mono).
    fs : sample rate in Hz.
    vad_decision : optional bool array, one entry per frame, True = speech.
        Get this from spectral_energy_vad() / spectral_energy_and_periodicity_vad()
        using the SAME frame_ms/hop_ms passed here. If given, the noise
        spectrum is estimated by averaging magnitude over every frame
        flagged non-speech (decision == False). If None, falls back to
        assuming the first `noise_init_frames` frames are noise-only.
    alpha : base oversubtraction factor (typical range 1-6).
    adaptive_alpha : if True, alpha is scaled per-frame by segmental SNR
        (see _oversubtraction_alpha); alpha above becomes alpha_max at
        low SNR and alpha_min at high SNR instead of a single fixed value.
    beta : spectral floor, as a fraction of the frame's own magnitude
        (typical range 0.001-0.05). Prevents bins from hitting zero,
        which is what causes "musical noise" chirps.
    noise_update : if True, keep refining the noise estimate on frames
        the VAD marks non-speech as the signal progresses (handles noise
        that slowly changes level/character, e.g. varying engine RPM),
        using an exponential running average. If False (default), the
        noise spectrum is estimated once from the whole non-speech mask
        and held fixed -- simpler and fine for roughly stationary noise.
    noise_update_rate : smoothing factor for the running update (only
        used if noise_update=True). Higher = adapts faster.

    Returns
    -------
    enhanced : 1-D float array, same length as x.
    diagnostics : dict with 'noise_mag' (the noise magnitude spectrum
        used, for plotting) and 'alpha_used' (per-frame alpha values).
    """
    frame_len = int(round(fs * frame_ms / 1000))
    hop_len = int(round(fs * hop_ms / 1000))
    window = np.hamming(frame_len)

    frames = frame_signal(x, frame_len, hop_len)
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, axis=1)
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    num_frames, num_bins = mag.shape

    # --- noise magnitude spectrum estimate ---
    if vad_decision is not None:
        vad_decision = np.asarray(vad_decision, dtype=bool)
        if len(vad_decision) != num_frames:
            raise ValueError(
                f"vad_decision has {len(vad_decision)} frames but framing "
                f"x with frame_ms={frame_ms}, hop_ms={hop_ms} gives "
                f"{num_frames} frames -- make sure both used the same "
                f"frame_ms/hop_ms on the same signal."
            )
        noise_mask = ~vad_decision
        if not np.any(noise_mask):
            raise ValueError(
                "vad_decision marks every frame as speech -- no non-speech "
                "frames available to estimate the noise spectrum from."
            )
        noise_mag_initial = mag[noise_mask].mean(axis=0)
    else:
        noise_mag_initial = mag[:noise_init_frames].mean(axis=0)
        noise_mask = np.zeros(num_frames, dtype=bool)
        noise_mask[:noise_init_frames] = True

    # --- segmental SNR per frame (for adaptive alpha) ---
    frame_energy_db = 10 * np.log10(np.sum(mag ** 2, axis=1) + 1e-10)
    noise_energy_db = 10 * np.log10(np.sum(noise_mag_initial ** 2) + 1e-10)
    frame_snr_db = frame_energy_db - noise_energy_db

    if adaptive_alpha:
        alpha_per_frame = _oversubtraction_alpha(frame_snr_db, alpha_min, alpha_max)
    else:
        alpha_per_frame = np.full(num_frames, alpha)

    # --- subtraction, frame by frame (loop needed if noise_update tracks
    #     a running estimate; harmless overhead otherwise) ---
    enhanced_mag = np.zeros_like(mag)
    noise_mag = noise_mag_initial.copy()

    for i in range(num_frames):
        a = alpha_per_frame[i]
        subtracted = mag[i] - a * noise_mag
        floor = beta * mag[i]
        enhanced_mag[i] = np.maximum(subtracted, floor)

        if noise_update and noise_mask[i]:
            noise_mag = (1 - noise_update_rate) * noise_mag + noise_update_rate * mag[i]

    # --- reconstruct with noisy phase, IFFT, overlap-add ---
    enhanced_spectrum = enhanced_mag * np.exp(1j * phase)
    enhanced_frames = np.fft.irfft(enhanced_spectrum, n=frame_len, axis=1)

    enhanced = _overlap_add(enhanced_frames, hop_len, window, len(x))

    diagnostics = {
        "noise_mag": noise_mag_initial,
        "alpha_used": alpha_per_frame,
        "frame_snr_db": frame_snr_db,
    }
    return enhanced, diagnostics


# ---------------------------------------------------------------------------
# Demo / sanity check: same synthetic burst-in-noise signal used in
# spectral_energy_vad.py, so you can see subtraction working without any
# real audio file on hand.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from spectral_energy_vad import spectral_energy_vad

    fs = 16000
    dur = 3.0
    t = np.arange(int(fs * dur)) / fs

    rng = np.random.default_rng(0)
    noise = 0.08 * rng.standard_normal(len(t))
    speech = np.zeros_like(t)
    on = (t > 1.0) & (t < 2.0)
    speech[on] = 0.3 * np.sin(2 * np.pi * 1000 * t[on])
    x = speech + noise

    decision, _, _ = spectral_energy_vad(x, fs)
    enhanced, diag = spectral_subtraction(x, fs, vad_decision=decision)

    def rms_db(sig):
        return 10 * np.log10(np.mean(sig ** 2) + 1e-12)

    noise_only_before = x[~np.repeat(decision, int(fs * 0.010))[:len(x)]] \
        if False else x[(t < 1.0) | (t > 2.0)]
    noise_only_after = enhanced[(t < 1.0) | (t > 2.0)]

    print(f"noise-only region RMS before: {rms_db(noise_only_before):.1f} dB")
    print(f"noise-only region RMS after:  {rms_db(noise_only_after):.1f} dB")
    print(f"noise reduction: {rms_db(noise_only_before) - rms_db(noise_only_after):.1f} dB")
    print(f"alpha range used: {diag['alpha_used'].min():.2f} - {diag['alpha_used'].max():.2f}")
