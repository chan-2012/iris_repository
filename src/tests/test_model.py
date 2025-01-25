import unittest
import pandas as pd
from src.model import train_model, predict
import joblib


class TestModel(unittest.TestCase):
    def test_model_training(self):
        model = train_model()
        self.assertIsNotNone(model)

    def test_model_prediction(self):
        model = joblib.load("model.joblib")
        test_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                                 columns=['SepalLengthCm', 'SepalWidthCm',
                                          'PetalLengthCm', 'PetalWidthCm'])
        prediction = predict(model, test_data)
        self.assertIsNotNone(prediction)


if __name__ == '__main__':
    unittest.main()
