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
        self.model = joblib.load(os.path.join(self.current_path, "classify_bbcry", "randomforest_model_classification.pkl"))
        self.n_fft = 1024  # setting the FFT size to 1024
        self.hop_length = 10*16 # 25ms*16khz samples has been taken
        self.win_length = 25*16 #25ms*16khz samples has been taken for window length
        self.window = 'hann' #hann window used
        self.n_mels=128
        self.n_bands=7 #we are extracting the 7 features out of the spectral contrast
        self.fmin=100 #minimum frequency
    
    def extract_features(self, y, sr):    
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.win_length,window=self.window).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr,n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window='hann',n_mels=self.n_mels).T,axis=0)

        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, y=y, sr=sr).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, y=y, sr=sr,n_fft=self.n_fft,
                                                        hop_length=self.hop_length, win_length=self.win_length,
                                                        n_bands=self.n_bands, fmin=self.fmin).T,axis=0)
        tonnetz =np.mean(librosa.feature.tonnetz(y=y, sr=sr).T,axis=0)
        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        return features
    
    def predict(self, audio_path):
        try:
            print(f"Predicting infant cry for {audio_path}...")
            signal, sr = librosa.load(audio_path, sr=16000)
            
            features = self.extract_features(signal, sr)

            # Dự đoán
            prediction = self.model.predict(features.reshape(1, -1))[0]

            return prediction
        except Exception as e:
            print(f"Exception: {str(e)}")
            return None