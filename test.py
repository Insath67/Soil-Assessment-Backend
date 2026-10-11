"""Smoke test against a running local server: python test.py."""
import requests

base = 'http://127.0.0.1:5000'
metadata = requests.get(f'{base}/metadata', timeout=10)
metadata.raise_for_status()
response = requests.post(f'{base}/predict', json={**metadata.json()['samples']['Tea'], 'weather_mode': 'manual'}, timeout=30)
response.raise_for_status()
result = response.json()
print('Recommended:', result['recommended_crop'])
print(result['explanation'])
for crop in result['recommendations']:
    print(f"{crop['crop']}: {crop['score_percent']}%")
