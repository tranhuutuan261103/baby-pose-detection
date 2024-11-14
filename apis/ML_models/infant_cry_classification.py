import numpy as np
import librosa
import os
import joblib

class InfantCryClassification:
    def __init__(self):
        self.sample_rate = 16000
        self.frame_length = int(0.032 * self.sample_rate)  # 32ms
        self.hop_length = self.frame_length//2    # 16ms
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.model = joblib.load(os.path.join(self.current_path, "classify_bbcry", "classify_bbcry_RFmodel.pkl"))

    def select_cry_frames_using_energy(self, signal: np.ndarray, frame_length, hop_length, thresh=0.1) -> list:
        energy = np.array([
            sum(abs(signal[i:i+frame_length]**2))
            for i in range(0, len(signal), hop_length)
        ])
        energy_norm = energy / max(energy)
        index_cry = [i for i in range(len(energy)) if energy_norm[i] >= thresh]
        return index_cry

    def split_segments(self, index_voices: list, hop_length, sr, min_duration=0.20) -> list:
        start = index_voices[0]
        segments = []
        for i in range(1, len(index_voices)):
            if index_voices[i] - index_voices[i-1] > 1:
                segments.append((start, index_voices[i-1]))
                start = index_voices[i]
        segments.append((start, index_voices[-1]))
        return [(start, end) for start, end in segments if (end - start) * hop_length / sr >= min_duration]
    
    def extract_features(self, signal, sr, n_mfcc=12, n_fft=2048):
        # Trích xuất MFCC (Mel Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, hop_length=self.hop_length, n_fft=n_fft)
        mfccs_mean = np.mean(mfccs, axis=1)  # Trả về giá trị trung bình của MFCC qua toàn bộ tín hiệu
        
        # Trích xuất RMS (Root Mean Square)
        rms = librosa.feature.rms(y=signal, frame_length=self.frame_length, hop_length=self.hop_length)
        rms_mean = np.mean(rms)  # Trả về giá trị trung bình của RMS
        
        # Trích xuất Zero-Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=self.frame_length, hop_length=self.hop_length)
        zcr_mean = np.mean(zcr)  # Trả về giá trị trung bình của Zero-Crossing Rate
        
        # Trả về tất cả các đặc trưng: MFCCs, RMS, và Zero-Crossing Rate
        return np.concatenate([mfccs_mean, [rms_mean], [zcr_mean]])
    
    def predict(self, audio_path):
        try:
            print(f"Predicting infant cry for {audio_path}...")
            signal, sr = librosa.load(audio_path, sr=16000)
            
            # Chọn các frame chứa tiếng khóc dựa vào năng lượng chuẩn hóa
            index_cry = self.select_cry_frames_using_energy(signal, self.frame_length, self.hop_length)
            
            # Chia các frame thành các segment
            segments = self.split_segments(index_cry, self.hop_length, sr)

            X = []
            
            # Trích xuất đặc trưng cho từng segment
            for (start, end) in segments:
                segment = signal[start * self.hop_length : end * self.hop_length]
                features = self.extract_features(segment, sr)
                X.append(features)

            # Dự đoán
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"Exception: {str(e)}")
            return []