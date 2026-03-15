import requests
import json

r = requests.get('https://api.openweathermap.org/data/2.5/forecast?lat=27.71&lon=85.32&appid=c0fdf4e74031a94a71626a0fdcb31e59&units=metric')
data = r.json()
print(json.dumps(data['list'][:2], indent=2))
