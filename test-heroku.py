import requests


app_url = "https://jedhamlproduction.herokuapp.com/predict"
response = requests.post(app_url, json={
    "input": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]]
})
print(response.json())