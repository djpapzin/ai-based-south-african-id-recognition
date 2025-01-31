import requests

API_TOKEN = "c4965d5c5515d76644b18343d198685d738ce34a"
BASE_URL = "http://localhost:8080"

print("Testing Label Studio connection...")
try:
    # Test basic connection
    response = requests.get(BASE_URL)
    print(f"Basic connection: {response.status_code}")
    
    # Test API connection
    headers = {'Authorization': f'Token {API_TOKEN}'}
    api_response = requests.get(f"{BASE_URL}/api/projects/", headers=headers)
    print(f"API connection: {api_response.status_code}")
    print("Response:", api_response.text)
    
except Exception as e:
    print(f"Error: {str(e)}") 