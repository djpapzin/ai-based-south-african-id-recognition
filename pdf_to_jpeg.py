import os
from pdf2image import convert_from_path
from pathlib import Path
import glob

def convert_pdf_to_jpeg(pdf_path, output_dir=None, dpi=300):
    """
    Convert a PDF file to JPEG images.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save the JPEG files. If None, uses the same directory as PDF
        dpi (int, optional): DPI for the output images. Default is 300
    
    Returns:
        list: List of paths to the generated JPEG files
    """
    try:
        # Validate PDF path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get PDF filename without extension
        pdf_name = Path(pdf_path).stem
        
        # Check if JPEGs already exist
        existing_jpegs = glob.glob(os.path.join(output_dir, f"{pdf_name}_page_*.jpg"))
        if existing_jpegs:
            print(f"Skipping {pdf_path} - JPEG files already exist")
            return existing_jpegs
        
        # Convert PDF to images
        print(f"Converting {pdf_path} to JPEG...")
        images = convert_from_path(pdf_path, dpi=dpi)
        
        # Save each page as JPEG
        jpeg_paths = []
        for i, image in enumerate(images):
            # Create output filename
            output_file = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.jpg")
            
            # Save image
            image.save(output_file, 'JPEG')
            jpeg_paths.append(output_file)
            print(f"Saved page {i+1} as {output_file}")
        
        return jpeg_paths
    
    except Exception as e:
        print(f"Error converting PDF to JPEG: {str(e)}")
        return []

def process_directory(input_dir, output_base_dir=None):
    """
    Process all PDF files in a directory and its subdirectories.
    
    Args:
        input_dir (str): Input directory containing PDF files
        output_base_dir (str, optional): Base directory for output files
    """
    # Get all PDF files in directory and subdirectories
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    total_files = len(pdf_files)
    print(f"\nFound {total_files} PDF files in {input_dir}")
    
    # Process each PDF file
    for i, pdf_path in enumerate(pdf_files, 1):
        # Create corresponding output directory structure
        rel_path = os.path.relpath(os.path.dirname(pdf_path), input_dir)
        if output_base_dir:
            output_dir = os.path.join(output_base_dir, rel_path)
        else:
            output_dir = os.path.dirname(pdf_path)
        
        print(f"\nProcessing file {i}/{total_files}: {pdf_path}")
        convert_pdf_to_jpeg(pdf_path, output_dir)

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert PDF to JPEG images')
    parser.add_argument('input_dirs', nargs='+', help='Input directories containing PDF files')
    parser.add_argument('--output_dir', help='Base directory for output files (optional)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output images (default: 300)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process each input directory
    for input_dir in args.input_dirs:
        print(f"\nProcessing directory: {input_dir}")
        process_directory(input_dir, args.output_dir) 