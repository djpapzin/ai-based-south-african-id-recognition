import requests

url = "http://localhost:8080/api/projects/7"
headers = {"Authorization": "Token c4965d5c5515d76644b18343d198685d738ce34a"}

try:
    print("Testing project access...")
    response = requests.get(url, headers=headers)
    print(f"Status code: {response.status_code}")
    print("Response:", response.text[:500])
except Exception as e:
    print(f"Error: {str(e)}") 