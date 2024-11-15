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
    def extract_features(self, y, sr):    
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.win_length,window=self.window).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr,n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window='hann',n_mels=self.n_mels).T,axis=0)
        print(mel.shape)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, y=y, sr=sr).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, y=y, sr=sr,n_fft=self.n_fft,
                                                        hop_length=self.hop_length, win_length=self.win_length,
                                                        n_bands=self.n_bands, fmin=self.fmin).T,axis=0)
        tonnetz =np.mean(librosa.feature.tonnetz(y=y, sr=sr).T,axis=0)
        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        print(features.shape)
        return features
    
    def predict(self, audio_path):
        filepath = os.path.join("", audio_path)
        y, sr = librosa.load(filepath, sr=16000)
        
        # Chuẩn hóa độ dài tín hiệu
        y = self.normalize_audio_length(y, sr)

        # Trích xuất đặc trưng
        features = self.extract_features(y, sr)

        # Dự đoán
        prediction = self.model.predict(features.reshape(1, -1))[0]

        print(f"Prediction: {prediction}")

        return prediction