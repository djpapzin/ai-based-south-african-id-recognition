import requests

headers = {'Authorization': 'Token c4965d5c5515d76644b18343d198685d738ce34a'}
base_url = 'http://localhost:8080'

# Test connection and get projects
try:
    response = requests.get(f"{base_url}/api/projects/", headers=headers)
    projects = response.json()
    print("\nAvailable projects:")
    for project in projects:
        print(f"ID: {project['id']}, Title: {project['title']}")
except Exception as e:
    print(f"Error connecting to Label Studio: {str(e)}") 