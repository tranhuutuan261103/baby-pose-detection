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
    
    def hps_f0(self, signal, sample_rate, min_freq=60, max_freq=800, N_FFT=2048):
        # Áp dụng cửa sổ Hamming để làm mượt tín hiệu
        windowed_signal = signal * np.hamming(len(signal))
        
        # Thực hiện FFT và chỉ lấy phổ biên độ
        fft_N_points = np.fft.fft(windowed_signal, N_FFT)
        spectrum = 2.0/N_FFT * np.abs(fft_N_points[:N_FFT//2])
        frequencies_N_points = sample_rate * np.arange(N_FFT//2) / N_FFT
        
        # Giới hạn tần số trong khoảng quan tâm (80Hz - 400Hz)
        valid_freqs = (frequencies_N_points >= min_freq) & (frequencies_N_points <= max_freq)
        frequencies_N_points = frequencies_N_points[valid_freqs]
        spectrum = spectrum[valid_freqs]
        # Áp dụng HPS (Harmonic Product Spectrum)
        hps_spectrum = np.copy(spectrum)

        # Nhân phổ với các bội số 2, 3, 4,...
        for h in range(2, 4):
            # Downsample bằng cách sử dụng phép nội suy để lấy phổ tương ứng
            downsampled_spectrum = np.interp(
                np.arange(0, len(spectrum), h),  # Các giá trị sau khi nội suy
                np.arange(0, len(spectrum)),     # Các giá trị ban đầu
                spectrum                        # Phổ gốc
            )
            
            # Đảm bảo các phổ có cùng độ dài trước khi nhân
            min_len = min(len(hps_spectrum), len(downsampled_spectrum))
            hps_spectrum[:min_len] *= downsampled_spectrum[:min_len]
            log_spectrum = np.log(np.abs(spectrum) + np.finfo(float).eps)  # Thêm epsilon để tránh log(0)
            
        # Tìm tần số có biên độ lớn nhất sau khi áp dụng HPS
        peak_index = np.argmax(log_spectrum)
        peak_freq = frequencies_N_points[peak_index]
        return peak_freq
    
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
                f0 = self.hps_f0(frame, sr) 
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