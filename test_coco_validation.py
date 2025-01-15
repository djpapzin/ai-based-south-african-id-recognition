import argparse
from coco_utils import validate_coco_format, convert_labelstudio_to_coco
import logging

def main():
    parser = argparse.ArgumentParser(description='Validate COCO format and convert Label Studio exports')
    parser.add_argument('--mode', choices=['validate', 'convert'], required=True,
                      help='Operation mode: validate existing COCO file or convert from Label Studio')
    parser.add_argument('--input', required=True,
                      help='Input file path (COCO JSON for validate mode, Label Studio JSON for convert mode)')
    parser.add_argument('--image-dir', required=True,
                      help='Directory containing the images')
    parser.add_argument('--output', 
                      help='Output file path for convert mode')

    args = parser.parse_args()

    if args.mode == 'validate':
        success = validate_coco_format(args.input, args.image_dir)
        if success:
            print("✅ COCO file validation successful!")
        else:
            print("❌ COCO file validation failed. Check the log file for details.")
    
    elif args.mode == 'convert':
        if not args.output:
            parser.error("--output is required for convert mode")
        
        success = convert_labelstudio_to_coco(args.input, args.image_dir, args.output)
        if success:
            print(f"✅ Successfully converted to COCO format: {args.output}")
            print("✅ Validation of generated file passed!")
        else:
            print("❌ Conversion failed. Check the log file for details.")

if __name__ == "__main__":
    main() 