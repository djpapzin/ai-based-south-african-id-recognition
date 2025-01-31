import json

# Configuration
API_TOKEN = "c4965d5c5515d76644b18343d198685d738ce34a"
BASE_URL = "http://localhost:8080"
headers = {
    'Authorization': f'Token {API_TOKEN}',
    'Accept': 'application/json'
}

def test_connection():
    print(f"Testing connection to Label Studio at {BASE_URL}")
    print(f"Using API token: {API_TOKEN}")
    
    try:
        # Test basic connection
        print("\n1. Testing basic connection...")
        response = requests.get(BASE_URL)
        print(f"Basic connection status: {response.status_code}")
        
        # Test API connection
        print("\n2. Testing API connection...")
        api_response = requests.get(f"{BASE_URL}/api/projects/", headers=headers)
        print(f"API status code: {api_response.status_code}")
        print("API response headers:", json.dumps(dict(api_response.headers), indent=2))
        print("\nAPI response content:", api_response.text[:500])  # Show first 500 chars
        
        if api_response.status_code == 200:
            data = api_response.json()
            print("\nSuccessfully parsed JSON response:")
            print(json.dumps(data, indent=2))
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {BASE_URL}")
        print("Please make sure Label Studio is running")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print("Raw response:", api_response.text)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
 