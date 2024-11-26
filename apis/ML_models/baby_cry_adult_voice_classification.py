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
    
    def difference_function_fast(self, signal, max_lag):
        """Tính hàm hiệu sai nhanh bằng NumPy."""
        size = len(signal)
        padded_signal = np.pad(signal, (0, max_lag), mode='constant', constant_values=0)
        cumsum = np.cumsum(padded_signal**2)
        diff = np.zeros(max_lag)
        for tau in range(1, max_lag):
            # Hiệu quả hơn bằng cách sử dụng cumsum và dot product
            diff[tau] = cumsum[size] - cumsum[tau] - 2 * np.dot(signal[:size-tau], signal[tau:size])
        return diff

    def cumulative_mean_normalized_difference_fast(self, diff):
        """Chuẩn hóa hàm hiệu sai tích lũy (phiên bản nhanh)."""
        cmndf = np.zeros_like(diff)
        cmndf[0] = 1  # Không sử dụng tau = 0
        running_sum = 0
        for tau in range(1, len(diff)):
            running_sum += diff[tau]
            cmndf[tau] = diff[tau] / (running_sum / tau) if running_sum > 0 else 1
        return cmndf

    def find_f0_yin(self, signal, sr, min_freq=50, max_freq=700, threshold=0.1):
        """Tính F0 bằng phương pháp YIN (tối ưu)."""
        # Tính độ trễ tối đa và tối thiểu
        max_lag = int(sr / min_freq)
        min_lag = int(sr / max_freq)

        # Tính hàm hiệu sai nhanh
        diff = self.difference_function_fast(signal, max_lag)

        # Chuẩn hóa hàm hiệu sai
        cmndf = self.cumulative_mean_normalized_difference_fast(diff)

        # Tìm độ trễ (tau) đầu tiên vượt qua ngưỡng
        tau = self.absolute_threshold(cmndf[min_lag:], threshold)
        tau += min_lag  # Bù chỉ số vì cmndf bị cắt

        # Nếu không tìm được độ trễ hợp lệ
        if tau == 0:
            return 0

        # Tính F0 từ độ trễ
        f0 = sr / tau

        # Nếu F0 nằm ngoài khoảng cho phép, trả về 0
        if f0 < min_freq or f0 > max_freq:
            return 0

        return f0

    # Hàm tìm ngưỡng không đổi (giữ nguyên)
    def absolute_threshold(self, cmndf, threshold):
        for tau in range(1, len(cmndf)):
            if cmndf[tau] < threshold:
                return tau
        return 0
    
    def detect_audio_class(self, file_path, frame_length_ms=30, frame_step_ms=15, sr=16000, 
                       energy_threshold=0.005, f0_threshold=400):

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
                f0 = self.find_f0_yin(frame, sr)
                if f0 > f0_threshold:
                    cry_count += 1
                elif f0 == 0:
                    continue
                else:
                    voice_count += 1

        print(f"Frames classified as Silence: {silence_count}")
        print(f"Frames classified as Voice: {voice_count}")
        print(f"Frames classified as Cry: {cry_count}")

        if silence_count > max(voice_count, cry_count):
            if voice_count > cry_count:
                return 0
            else:
                return 1
        elif cry_count > voice_count:
            return 1
        else:
            return 0