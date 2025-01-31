import requests

base_url = "http://localhost:8080"
headers = {"Authorization": "Token c4965d5c5515d76644b18343d198685d738ce34a"}

try:
    # Try different task endpoints
    endpoints = [
        "/api/tasks?project=7",
        "/api/projects/7/tasks",
        "/api/tasks?project_id=7"
    ]
    
    for endpoint in endpoints:
        print(f"\nTrying endpoint: {endpoint}")
        url = base_url + endpoint
        response = requests.get(url, headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                print(f"Found {len(data)} tasks")
            elif isinstance(data, dict):
                results = data.get('results', [])
                print(f"Found {len(results)} tasks")
                print(f"Total count: {data.get('count', 'unknown')}")
            print("First few bytes of response:", response.text[:200])
except Exception as e:
    print(f"Error: {str(e)}") 