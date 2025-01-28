import os
import glob
from pathlib import Path

def can_delete_pdf(pdf_path):
    """
    Check if a PDF can be safely deleted by verifying its JPEG files exist.
    """
    # Get PDF filename without extension
    pdf_name = Path(pdf_path).stem
    pdf_dir = os.path.dirname(pdf_path)
    
    # Look for corresponding JPEG files
    jpeg_pattern = os.path.join(pdf_dir, f"{pdf_name}_page_*.jpg")
    existing_jpegs = glob.glob(jpeg_pattern)
    
    # Only return True if at least one JPEG exists
    return len(existing_jpegs) > 0

def delete_converted_pdfs(directory):
    """
    Delete PDF files that have been successfully converted to JPEG.
    """
    deleted_count = 0
    skipped_count = 0
    
    # Walk through directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                
                if can_delete_pdf(pdf_path):
                    try:
                        os.remove(pdf_path)
                        print(f"Deleted: {pdf_path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {pdf_path}: {str(e)}")
                        skipped_count += 1
                else:
                    print(f"Skipping {pdf_path} - No JPEG files found")
                    skipped_count += 1
    
    return deleted_count, skipped_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Delete PDF files that have been converted to JPEG')
    parser.add_argument('directories', nargs='+', help='Directories containing PDF files to process')
    
    args = parser.parse_args()
    
    total_deleted = 0
    total_skipped = 0
    
    for directory in args.directories:
        print(f"\nProcessing directory: {directory}")
        deleted, skipped = delete_converted_pdfs(directory)
        total_deleted += deleted
        total_skipped += skipped
    
    print(f"\nSummary:")
    print(f"Total PDFs deleted: {total_deleted}")
    print(f"Total PDFs skipped: {total_skipped}") 