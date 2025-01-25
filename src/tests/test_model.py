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

        # Create a more representative test case
        test_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                                 columns=['SepalLengthCm',
                                          'SepalWidthCm',
                                          'PetalLengthCm',
                                          'PetalWidthCm'])
        prediction = predict(model, test_data)

        # Print the prediction (for demonstration)
        print(f"Prediction: {prediction}")

        # Add a more specific assertion
        self.assertEqual(len(prediction), 1)  # Make sure we got prediction
        # Since we know this test data should be setosa
        self.assertEqual(prediction[0], 'Iris-setosa')  # Check prediction

        test_data_versicolor = pd.DataFrame([[6.0, 3.0, 4.5, 1.5]],
                                            columns=['SepalLengthCm',
                                                     'SepalWidthCm',
                                                     'PetalLengthCm',
                                                     'PetalWidthCm'])
        prediction_versicolor = predict(model, test_data_versicolor)
        print(f"Prediction Versicolor: {prediction_versicolor}")
        self.assertEqual(prediction_versicolor[0], 'Iris-versicolor')


if __name__ == '__main__':
    unittest.main()
