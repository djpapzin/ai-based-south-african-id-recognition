from zipfile import ZipFile
import os
import json

def check_zip(zip_path):
    print(f"\nChecking {zip_path}...")
    
    with ZipFile(zip_path) as zf:
        # List all files in zip
        print("\nFiles in zip:")
        for name in zf.namelist():
            print(f"  {name}")
        
        # Extract and check result.json
        if "result.json" in zf.namelist():
            zf.extract("result.json", "temp")
            with open("temp/result.json") as f:
                data = json.load(f)
                print(f"\nContents of result.json:")
                print(f"  Images: {len(data.get('images', []))}")
                print(f"  Annotations: {len(data.get('annotations', []))}")
                print(f"  Categories: {data.get('categories', [])}")
                
                if data.get('images'):
                    print("\nFirst image entry:")
                    print(json.dumps(data['images'][0], indent=2))
                
                if data.get('annotations'):
                    print("\nFirst annotation entry:")
                    print(json.dumps(data['annotations'][0], indent=2))
        else:
            print("No result.json found!")
            
        # Check images directory
        images = [f for f in zf.namelist() if f.startswith('images/')]
        print(f"\nNumber of files in images/: {len(images)}")
        if images:
            print("First few image files:")
            for img in images[:5]:
                print(f"  {img}")

try:
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    
    # Check both datasets
    check_zip("dataset1.zip")
    check_zip("dataset2.zip")
    
finally:
    # Cleanup
    if os.path.exists("temp"):
        import shutil
        shutil.rmtree("temp") 