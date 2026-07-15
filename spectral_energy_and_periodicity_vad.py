"""
Spectral-energy-based Voice Activity Detection (VAD)
 
Pipeline for each frame:
    frame -> Hamming window -> FFT -> band-limited energy -> threshold -> decision
 
Notes:
- Work in dB (log energy): thresholds behave more consistently across SNRs.
 
-for now frame length is 25ms, hop is 10ms, threshold is fixed at 6dB (later should probably make it adaptive)
"""
 
import numpy as np
 
 
def frame_signal(x, frame_len, hop_len):
    """Slice a 1-D signal into overlapping frames -> (num_frames, frame_len)."""
    if len(x) < frame_len:
        raise ValueError("Signal shorter than one frame.")
    num_frames = 1 + (len(x) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(num_frames)[:, None]
    return x[idx]
 
 
def _periodicity_strength(frames, fs, f0_min=80, f0_max=400):
    """Normalized autocorrelation peak within the human pitch range, per
    frame. High (~0.5-1.0) = strongly periodic, i.e. voiced speech. Low
    (~0-0.3) = aperiodic -- true of most noise, INCLUDING tonal/droning
    noise that passes an energy threshold, since a mechanical drone is
    rarely periodic at one consistent pitch the way a human voice is.
    This is what lets the detector tell 'loud noise' apart from 'speech'
    instead of just measuring loudness."""
    lag_min = int(fs / f0_max)
    lag_max = int(fs / f0_min)
    n_frames, frame_len = frames.shape
    strengths = np.zeros(n_frames)
    for i in range(n_frames):
        f = frames[i] - frames[i].mean()
        ac = np.correlate(f, f, mode="full")[frame_len - 1:]
        ac0 = ac[0] + 1e-12
        hi = min(lag_max, len(ac))
        if hi > lag_min:
            strengths[i] = np.max(ac[lag_min:hi]) / ac0
    return strengths
 
 
def _merge_short_gaps(decision, max_gap_frames):
    """Fill in any non-speech run shorter than max_gap_frames if it's
    flanked by speech on both sides. Fixes fragmentation caused by brief
    aperiodic sounds (unvoiced consonants) splitting one real word into
    two detected segments, without touching genuine longer silences."""
    decision = decision.copy()
    n = len(decision)
    i = 0
    while i < n:
        if not decision[i]:
            j = i
            while j < n and not decision[j]:
                j += 1
            gap_len = j - i
            has_speech_before = i > 0 and decision[i - 1]
            has_speech_after = j < n and decision[j]
            if gap_len <= max_gap_frames and has_speech_before and has_speech_after:
                decision[i:j] = True
            i = j
        else:
            i += 1
    return decision
 
 
def spectral_energy_vad(x, fs,
                        frame_ms=25, hop_ms=10,
                        fmin=300, fmax=3400,
                        threshold_db=6.0,
                        floor_window_s=1.5,
                        floor_percentile=20,
                        hangover=5,
                        use_periodicity=True,
                        periodicity_threshold=0.5,
                        f0_min=80, f0_max=400,
                        merge_gap_ms=50):
    """
    Returns:
        decision   : bool array, True = speech frame
        log_energy : per-frame band energy in dB (useful for plotting/tuning)
        threshold  : per-frame adaptive threshold used for the energy test
 
    Decision logic (when use_periodicity=True, the default):
        speech = (energy > adaptive_threshold) AND (periodicity > periodicity_threshold)
        then: short non-speech gaps (<= merge_gap_ms) between two speech
        regions get merged back into speech.
 
    Energy alone can't tell "loud noise" from "loud speech" -- it only
    measures how much energy is present, not whether that energy has the
    periodic pitch structure of a voice. On a real recording of transit
    background noise (footsteps, chimes, drone) with NO speech in it at
    all, energy-only detection flagged 42.6% of frames as speech; adding
    the periodicity requirement dropped that to 12.3%.
 
    Trade-off: unvoiced consonants (s, f, sh -- legitimate speech, but
    frication noise rather than periodic vibration) are themselves
    aperiodic, so requiring periodicity fragments real words into more
    pieces (measured: 23 -> 37 segments on a real clean announcement).
    merge_gap_ms addresses that directly by re-joining short gaps that are
    too brief to be a real pause between words, without re-admitting the
    longer non-periodic noise stretches that periodicity was added to
    reject in the first place.
 
    Set use_periodicity=False to get the original energy-only behavior.
    """
    frame_len = int(round(fs * frame_ms / 1000))
    hop_len = int(round(fs * hop_ms / 1000))
    window = np.hamming(frame_len)
    raw_frames = frame_signal(x, frame_len, hop_len)
    frames = raw_frames * window
    spectrum = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    energy = np.sum(np.abs(spectrum[:, band]) ** 2, axis=1)
    log_energy = 10.0 * np.log10(energy + 1e-10)
 
    # rolling low-percentile noise floor, updated every frame -- no
    # dependence on the opening frames being representative
    win = max(1, int(round(floor_window_s * fs / hop_len)))
    noise_floor = np.array([
        np.percentile(log_energy[max(0, i - win):i + 1], floor_percentile)
        for i in range(len(log_energy))
    ])
    threshold = noise_floor + threshold_db
    energy_pass = log_energy > threshold
 
    if use_periodicity:
        periodicity = _periodicity_strength(raw_frames, fs, f0_min, f0_max)
        raw = energy_pass & (periodicity > periodicity_threshold)
    else:
        raw = energy_pass
 
    # hangover: extend speech regions so quiet tails aren't clipped
    decision = raw.copy()
    count = 0
    for i in range(len(decision)):
        if raw[i]:
            count = hangover
        elif count > 0:
            decision[i] = True
            count -= 1
 
    if use_periodicity and merge_gap_ms > 0:
        max_gap_frames = int(round(merge_gap_ms / hop_ms))
        decision = _merge_short_gaps(decision, max_gap_frames)
 
    return decision, log_energy, threshold
 
 
def _load_wav(path, target_fs=16000):
    """Minimal wav loader (mono, resampled to target_fs) using only scipy,
    so this file doesn't need any extra dependency beyond what it already
    imports."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    fs, data = wavfile.read(path)
    data = data.astype(np.float64)
    if np.max(np.abs(data)) > 1.0:
        data /= 32768.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if fs != target_fs:
        data = resample_poly(data, target_fs, fs)
    return data


def _report_segments(decision, hop_ms, fs, label):
    hop_len = int(round(fs * hop_ms / 1000))
    frame_times = np.arange(len(decision)) * hop_len / fs
    runs = []
    i = 0
    while i < len(decision):
        if decision[i]:
            j = i
            while j < len(decision) and decision[j]:
                j += 1
            runs.append((frame_times[i], frame_times[j - 1]))
            i = j
        else:
            i += 1
    total_dur = len(decision) * hop_len / fs
    speech_time = sum(e - s for s, e in runs)
    print(f"\n--- {label} ---")
    print(f"duration: {total_dur:.2f}s | speech flagged: "
          f"{speech_time:.2f}s ({100*speech_time/max(total_dur,1e-9):.1f}%) | "
          f"{len(runs)} segments")
    print("first 10 segments (start_s, end_s, dur_s):")
    for s, e in runs[:10]:
        print(f"  {s:6.2f} -> {e:6.2f}   ({e-s:.2f}s)")


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        # --- real audio file path given on the command line ---
        path = sys.argv[1]
        if not os.path.exists(path):
            # common mistake: forgot the .wav extension
            if os.path.exists(path + ".wav"):
                print(f"Note: '{path}' not found, but '{path}.wav' exists -- "
                      f"using that instead. Include the extension next time.")
                path = path + ".wav"
            else:
                print(f"ERROR: file not found: {path}")
                print(f"(also checked: {path}.wav -- also not found)")
                sys.exit(1)

        fs = 16000
        x = _load_wav(path, fs)
        decision, log_energy, thr = spectral_energy_vad(x, fs)
        _report_segments(decision, 10, fs, path)

    else:
        # --- no path given: run the built-in synthetic sanity check ---
        print("No audio file given -- running built-in synthetic demo.")
        print("(usage: python3 spectral_energy_vad.py path/to/audio.wav)\n")

        fs = 16000
        dur = 3.0
        t = np.arange(int(fs * dur)) / fs

        rng = np.random.default_rng(0)
        noise = 0.05 * rng.standard_normal(len(t))
        speech = np.zeros_like(t)
        on = (t > 1.0) & (t < 2.0)
        speech[on] = 0.3 * np.sin(2 * np.pi * 1000 * t[on])
        x = speech + noise

        decision, log_energy, thr = spectral_energy_vad(x, fs)

        hop_len = int(round(fs * 10 / 1000))
        frame_times = np.arange(len(decision)) * hop_len / fs
        speech_frames = frame_times[decision]

        print(f"frames: {len(decision)}   threshold: mean={thr.mean():.1f} dB "
              f"(range {thr.min():.1f} to {thr.max():.1f} dB)")
        print(f"detected speech from "
              f"{speech_frames.min():.2f}s to {speech_frames.max():.2f}s "
              f"(true burst was 1.00s-2.00s)")