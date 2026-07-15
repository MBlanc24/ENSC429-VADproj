
"""
run_test.py
-----------
Simple, self-contained test of spectral_energy_vad() (the fixed,
rolling-noise-floor version) against real transit-announcement audio.
 
Just run:
    python3 run_test.py
 
Requires: numpy, scipy  (pip install numpy scipy)
"""
 
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
 
from spectral_energy_vad import spectral_energy_vad
 
 
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
 
 
def mix_at_snr(clean, noise, snr_db):
    n = min(len(clean), len(noise))
    clean, noise = clean[:n], noise[:n]
    p_clean = np.mean(clean ** 2) + 1e-12
    p_noise = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt((p_clean / (10 ** (snr_db / 10))) / p_noise)
    mixed = clean + scale * noise
    peak = np.max(np.abs(mixed))
    if peak > 0.98:
        mixed = mixed / peak * 0.98
    return mixed
 
 
def report(decision, hop_len, fs, label):
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
          f"{speech_time:.2f}s ({100*speech_time/total_dur:.1f}%) | "
          f"{len(runs)} segments")
    print("first 10 segments (start_s, end_s, dur_s):")
    for s, e in runs[:30]:
        print(f"  {s:6.2f} -> {e:6.2f}   ({e-s:.2f}s)")
 
 
if __name__ == "__main__":
    fs = 16000
    hop_len = int(round(fs * 10 / 1000))
 
    # Test 1: fairly clean real announcement, alone
    speech = load_wav("audio_samples/clean_generated_announcement.wav", fs)
    decision, log_energy, thr = spectral_energy_vad(speech, fs)
    report(decision, hop_len, fs, "clean_generated_announcement.wav (clean-ish)")
 
    # Test 2: same announcement + real background noise at a moderate SNR
    noise = load_wav("audio_samples/backgroundnoise.wav", fs)
    noisy = mix_at_snr(speech, noise, snr_db=5)
    decision, log_energy, thr = spectral_energy_vad(noisy, fs)
    report(decision, hop_len, fs, "announcement + real background noise @5dB SNR")
 
    # Test 3: genuinely noisy real field recording, no synthetic mixing
    field = load_wav("audio_samples/noise_announcement.wav", fs)
    decision, log_energy, thr = spectral_energy_vad(field, fs)
    report(decision, hop_len, fs, "noise_announcement.wav (real noisy field recording)")
 
    print("\nWhat to look for: plausible word/phrase-length segments (tens of ms to")
    print("~1s each), not one giant block covering nearly the whole file. A single")
    print("segment covering >90% of the recording usually means the noise floor")
    print("estimate went stale -- see the conversation history for that failure mode.")