import os
from pathlib import Path
from collections import defaultdict

def count_files(directory):
    """
    Count files in directory and subdirectories, grouped by extension.
    """
    file_counts = defaultdict(int)
    total_files = 0
    total_dirs = 0
    
    print(f"\nCounting files in: {directory}")
    
    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(directory):
        total_dirs += len(dirs)
        for file in files:
            extension = Path(file).suffix.lower()
            file_counts[extension] += 1
            total_files += 1
    
    # Print summary
    print("\nSummary:")
    print(f"Total directories: {total_dirs}")
    print(f"Total files: {total_files}")
    print("\nFiles by extension:")
    for ext, count in sorted(file_counts.items()):
        print(f"{ext or 'no extension'}: {count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Count files in directory by extension')
    parser.add_argument('directory', help='Directory to count files in')
    
    args = parser.parse_args()
    count_files(args.directory) 