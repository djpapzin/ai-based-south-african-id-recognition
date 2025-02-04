import json
import os
from pathlib import Path

def analyze_image_paths(dataset_path: str):
    """
    Analyze image paths and locations to find discrepancies.
    """
    # Load the JSON
    json_path = os.path.join(dataset_path, "result.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get all physical images
    image_dir = os.path.join(dataset_path, "images")
    physical_images = set(os.path.basename(str(p)) for p in Path(image_dir).glob("*.[jJ][pP][eE][gG]"))
    
    # Analyze JSON image entries
    print("\nAnalyzing JSON image entries:")
    print("-" * 50)
    
    # Group images by pattern
    pattern_groups = {
        "A_A": [],
        "A_B": [],
        "A": [],
        "B": []
    }
    
    for img in data["images"]:
        filename = img["file_name"].split("\\")[-1]
        if "_A_A" in filename:
            pattern_groups["A_A"].append(filename)
        elif "_A_B" in filename:
            pattern_groups["A_B"].append(filename)
        elif filename.endswith("_A.jpeg"):
            pattern_groups["A"].append(filename)
        elif filename.endswith("_B.jpeg"):
            pattern_groups["B"].append(filename)
    
    print("\nImage Pattern Analysis:")
    for pattern, files in pattern_groups.items():
        print(f"\n{pattern} pattern files: {len(files)}")
        if files:
            print("Examples:")
            for f in sorted(files)[:3]:
                print(f"  - {f}")
    
    # Check for corresponding pairs
    print("\nChecking for A/B pairs:")
    pairs = {}
    unpaired = []
    
    for img in data["images"]:
        filename = img["file_name"].split("\\")[-1]
        base_id = filename.split("-")[1].split("_")[0]  # Extract ID number
        
        if base_id not in pairs:
            pairs[base_id] = {"A": None, "B": None}
        
        if "_A.jpeg" in filename or "_A_A.jpg" in filename or "_A_B.jpg" in filename:
            pairs[base_id]["A"] = filename
        elif "_B.jpeg" in filename:
            pairs[base_id]["B"] = filename
    
    # Analyze pairs
    complete_pairs = 0
    a_only = 0
    b_only = 0
    
    for id_num, pair in pairs.items():
        if pair["A"] and pair["B"]:
            complete_pairs += 1
        elif pair["A"]:
            a_only += 1
            unpaired.append(f"Missing B for: {pair['A']}")
        elif pair["B"]:
            b_only += 1
            unpaired.append(f"Missing A for: {pair['B']}")
    
    print(f"\nPair Statistics:")
    print(f"Complete A/B pairs: {complete_pairs}")
    print(f"A-side only: {a_only}")
    print(f"B-side only: {b_only}")
    
    if unpaired:
        print("\nUnpaired Images (first 10):")
        for msg in sorted(unpaired)[:10]:
            print(f"  {msg}")
    
    # Check file paths in JSON
    print("\nAnalyzing file paths in JSON:")
    path_patterns = {}
    for img in data["images"]:
        path = img["file_name"]
        pattern = "\\".join(path.split("\\")[:-1])  # Get path without filename
        path_patterns[pattern] = path_patterns.get(pattern, 0) + 1
    
    print("\nFile path patterns used in JSON:")
    for pattern, count in path_patterns.items():
        print(f"  {pattern}: {count} files")

if __name__ == "__main__":
    dataset_path = r"C:\Users\lfana\Downloads\dj_dataset_02feb"
    analyze_image_paths(dataset_path) 