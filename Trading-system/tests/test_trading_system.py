import unittest
import numpy as np
from core.trading_model import TradingModel
from core.data_processor import DataProcessor

class TestTradingSystem(unittest.TestCase):

# ------------------------------------------------------------- #
    def setUp(self):
        self.model = TradingModel(input_shape=(60, 4), action_space=3)
        self.processor = DataProcessor()
# ------------------------------------------------------------- #

# ------------------------------------------------------------- #
    def test_model_prediction(self):
        state = np.random.randn(60, 4)
        prediction = self.model.predict(state)
        self.assertEqual(prediction.shape, (3,))
# ------------------------------------------------------------- #

# -----------------------simulação de dados-------------------- #
    def test_data_processing(self):
        test_data = [{
            'timestamp': '2023-01-01T00:00:00',
            'symbol': 'TEST',
            'price': '100.0',
            'volume': '1000'
        }]
        processed = self.processor._validate_data(test_data[0])
        self.assertIsNotNone(processed)
# ------------------------------------------------------------- #
    def test_feature_creation(self):
        window = [{
            'timestamp': f'2023-01-01T00:00:{i:02d}',
            'symbol': 'TEST',
            'price': str(100 + i),
            'volume': str(1000 + i*10)
        } for i in range(60)]
        
        features = self.processor.create_features(window)
        self.assertEqual(features.shape[1], 4)
# ------------------------------------------------------------- #
if __name__ == '__main__':
    unittest.main()
# ------------------------------------------------------------- #
