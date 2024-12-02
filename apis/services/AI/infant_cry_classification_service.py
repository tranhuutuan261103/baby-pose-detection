from ML_models.infant_cry_classification import InfantCryClassification

class InfantCryClassificationService:
    def __init__(self):
        self.infantCryClassification = InfantCryClassification()
    
    def predict(self, audio):
        return self.infantCryClassification.predict(audio)