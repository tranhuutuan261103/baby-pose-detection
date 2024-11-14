from ML_models.baby_cry_adult_voice_classification import BabyCryAdultVoiceClassification

class BabyCryAdultVoiceClassificationService:
    def __init__(self):
        self.babyCryAdultVoiceClassification = BabyCryAdultVoiceClassification()
    
    def predict(self, audio):
        return self.babyCryAdultVoiceClassification.predict(audio)