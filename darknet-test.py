import requests

#Klient
BASE = "http://127.0.0.1:5000/"

response = requests.post(BASE + "/place/1")
print(response.json())