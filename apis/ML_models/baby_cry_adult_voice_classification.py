import numpy as np
import librosa
import os
import joblib

class BabyCryAdultVoiceClassification:
    def __init__(self):
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.model = joblib.load(os.path.join(self.current_path, "detect_bbcry", "decision_model_detection.pkl"))
        self.n_fft = 1024  # setting the FFT size to 1024
        self.hop_length = 10*16 # 25ms*16khz samples has been taken
        self.win_length = 25*16 #25ms*16khz samples has been taken for window length
        self.window = 'hann' #hann window used
        self.n_mels=128
        self.n_bands=7 #we are extracting the 7 features out of the spectral contrast
        self.fmin=100 #minimum frequency

    def compute_normalized_energy(self, frames):
        """Tính năng lượng chuẩn hóa của từng khung."""
        energy = np.sum(frames**2, axis=1)
        if np.max(energy) > 0:
            normalized_energy = energy / np.max(energy)  # Chuẩn hóa nếu max > 0
        else:
            normalized_energy = energy  # Giữ nguyên nếu max = 0

        return normalized_energy
    
    def autocorrelation_f0(self, frame, fs, fmin=60, fmax=700):
        """Tính F0 bằng phương pháp tự tương quan."""
        corr = np.correlate(frame, frame, mode='full')[len(frame) - 1:]
        corr[:int(fs / fmax)] = 0  # Loại bỏ tần số quá cao
        peak_idx = np.argmax(corr)
        peak_lag = peak_idx if corr[peak_idx] > 0 else 0
        f0 = fs / peak_lag if peak_lag != 0 else 0
        return f0
    
    def detect_audio_class(self, file_path, frame_length_ms=30, frame_step_ms=15, sr=16000, 
                       energy_threshold=0.001, f0_threshold=400) -> int:

        signal, sr = librosa.load(file_path, sr=sr)

        frame_length = int(frame_length_ms * sr / 1000)
        frame_step = int(frame_step_ms * sr / 1000)
        frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=frame_step).T
        
        normalized_energy = self.compute_normalized_energy(frames)

        silence_count = 0
        voice_count = 0
        cry_count = 0

        for frame, energy in zip(frames, normalized_energy):
            if energy < energy_threshold:
                silence_count += 1
            else:
                f0 = self.autocorrelation_f0(frame, sr) 
                if f0 > f0_threshold:
                    cry_count += 1
                else:
                    voice_count += 1

        # In thông tin phân loại
        print(f"Frames classified as Silence: {silence_count}")
        print(f"Frames classified as Voice: {voice_count}")
        print(f"Frames classified as Cry: {cry_count}")

        # Quyết định lớp dựa trên số lượng khung
        if silence_count > max(voice_count, cry_count):
            return 0
        elif cry_count > voice_count:
            return 1
        else:
            return 0