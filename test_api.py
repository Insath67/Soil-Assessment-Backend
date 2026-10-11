"""Model/API regression checks. Run: python -m unittest -v test_api"""
import os
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from app import app, model, FEATURES, LABELS, SAMPLES

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_all_crops_and_shap_additivity(self):
        for expected, sample in SAMPLES.items():
            with self.subTest(crop=expected):
                response = self.client.post('/predict', json=sample)
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assertEqual(data['recommended_crop'], expected)
                self.assertEqual(len(data['recommendations']), 5)
                self.assertAlmostEqual(sum(c['probability'] for c in data['recommendations']), 1, places=5)
                frame = pd.DataFrame([[sample[f] for f in FEATURES]], columns=FEATURES)
                margins = model.predict(frame, output_margin=True)[0]
                for crop in data['recommendations']:
                    total = crop['base_value'] + sum(f['shap_value'] for f in crop['factors'])
                    self.assertAlmostEqual(total, float(margins[LABELS.index(crop['crop'])]), places=4)
                    self.assertEqual(len(crop['factors']), 7)
                    for factor in crop['factors']:
                        self.assertEqual(factor['direction'], 'supports' if factor['shap_value'] > 0 else 'opposes' if factor['shap_value'] < 0 else 'neutral')

    def test_invalid_inputs(self):
        for body in [None, [], {}, {**SAMPLES['Tea'], 'N': True}, {**SAMPLES['Tea'], 'ph': 15}, {**SAMPLES['Tea'], 'N': '40'}, {**SAMPLES['Tea'], 'rainfall': None}, {**SAMPLES['Tea'], 'weather_mode': 'unknown'}]:
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/predict', json=body).status_code, 400)

    def test_live_weather_without_key(self):
        with patch.dict(os.environ, {'OPENWEATHER_API_KEY': ''}):
            self.assertEqual(self.client.post('/predict', json={**SAMPLES['Tea'], 'weather_mode': 'live'}).status_code, 503)

    def test_live_weather_preserves_annual_rain(self):
        weather = {'temperature': 25., 'humidity': 80., 'source': 'Mock provider', 'observed_at': None}
        with patch('app.get_realtime_weather', return_value=weather):
            response = self.client.post('/predict', json={**SAMPLES['Tea'], 'weather_mode': 'live'})
            self.assertEqual(response.status_code, 200)
            inputs = response.get_json()['inputs']
            self.assertEqual(inputs['temperature'], 25.)
            self.assertEqual(inputs['rainfall'], SAMPLES['Tea']['rainfall'])

    def test_outside_training_range_warns(self):
        data = self.client.post('/predict', json={**SAMPLES['Tea'], 'N': 999}).get_json()
        self.assertTrue(any('Nitrogen' in warning for warning in data['warnings']))

    def test_metadata(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        self.assertEqual(self.client.get('/metadata').get_json()['features'], FEATURES)

if __name__ == '__main__':
    unittest.main()
