import numpy as np
import librosa
import os
import joblib

class BabyCryAdultVoiceClassification:
    def __init__(self):
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.model = joblib.load(os.path.join(self.current_path, "detect_bbcry", "randomforest_model.pkl"))

    def normalize_audio_length(self, y, sr, target_duration=7):
        target_length = int(sr * target_duration)
        
        if len(y) > target_length:
            # Nếu tín hiệu dài hơn 7s, cắt bớt
            y = y[:target_length]
        elif len(y) < target_length:
            # Nếu tín hiệu ngắn hơn 7s, thêm padding
            padding = target_length - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        
        return y
    
    # Hàm trích xuất đặc trưng
    def extract_acoustic_features(self, y, sr):
        win_length = int (0.03 * sr)
        hop_length = win_length//2
        n_fft = 2048
        # Trích xuất Mel Scale
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=15)
        mel_scale = np.mean(mel_spectrogram, axis=1)  # Tính trung bình theo chiều dọc

        # Trích xuất MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        mfcc_features = np.mean(mfccs, axis=1)  # Tính trung bình theo chiều dọc

        # Trích xuất Constant-Q Chromagram với điều chỉnh tham số
        fmin = librosa.note_to_hz('C2')  # Tần số tối thiểu khoảng 65Hz
        n_bins = 36  # Giảm số lượng tần số bin
        cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins)
        chroma = librosa.feature.chroma_cqt(hop_length=hop_length, C=np.abs(cqt), sr=sr)
        cqc_features = np.mean(chroma, axis=1)  # Tính trung bình theo chiều dọc

        # Kết hợp các đặc trưng
        features = np.hstack([mel_scale, mfcc_features, cqc_features])
        
        return features
    
    def predict(self, audio_path):
        filepath = os.path.join("", audio_path)
        y, sr = librosa.load(filepath, sr=16000)
        
        # Chuẩn hóa độ dài tín hiệu
        y = self.normalize_audio_length(y, sr)

        # Trích xuất đặc trưng
        features = self.extract_acoustic_features(y, sr)

        # Dự đoán
        prediction = self.model.predict(features.reshape(1, -1))[0]

        print(f"Prediction: {prediction}")

        return prediction