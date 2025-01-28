import os
from pathlib import Path

def check_and_clean_directories(directories):
    """
    Check all files in given directories and remove anything that's not an image file.
    """
    allowed_extensions = {'.jpg', '.jpeg', '.png'}  # Added .png to allowed extensions
    file_stats = {
        'images': 0,
        'removed': 0,
        'by_extension': {}
    }
    
    for directory in directories:
        print(f"\nProcessing directory: {directory}")
        
        # Walk through directory and subdirectories
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                extension = Path(file).suffix.lower()
                
                # Track extension statistics
                if extension not in file_stats['by_extension']:
                    file_stats['by_extension'][extension] = 0
                file_stats['by_extension'][extension] += 1
                
                # Check if file should be removed
                if extension not in allowed_extensions:
                    try:
                        os.remove(file_path)
                        print(f"Removed: {file_path}")
                        file_stats['removed'] += 1
                    except Exception as e:
                        print(f"Error removing {file_path}: {str(e)}")
                else:
                    file_stats['images'] += 1
    
    # Print summary
    print("\nSummary:")
    print(f"Total image files: {file_stats['images']}")
    print(f"Total files removed: {file_stats['removed']}")
    print("\nFiles found by extension:")
    for ext, count in sorted(file_stats['by_extension'].items()):
        print(f"{ext}: {count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check and clean directories to ensure only image files remain')
    parser.add_argument('directories', nargs='+', help='Directories to process')
    
    args = parser.parse_args()
    check_and_clean_directories(args.directories) 