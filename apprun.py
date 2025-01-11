import requests

url = "http://127.0.0.1:5000/generate"
data = {"query": "What new app should I build?"}
response = requests.post(url, json=data)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")
