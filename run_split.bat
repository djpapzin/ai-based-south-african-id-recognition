@echo off
python split_coco_dataset.py --input annotated_images/result_fixed_dimensions.json --image-dir annotated_images/images --output-dir pipeline_output --train-split 0.8 --val-split 0.2 --seed 42 