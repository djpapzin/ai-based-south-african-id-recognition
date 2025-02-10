import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import Levenshtein
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ground_truth/evaluation_results/evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class OCREvaluator:
    def __init__(self):
        self.fields = [
            'id_number', 'surname', 'names', 'nationality',
            'country_of_birth', 'date_of_birth', 'sex', 'citizenship_status'
        ]
        self.failed_evaluations = []
        
    def load_results(self, llm_path: str, ocr_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load results from both LLM and OCR JSON files with error handling."""
        try:
            with open(llm_path, 'r') as f:
                llm_results = json.load(f)
            logging.info(f"Loaded LLM results from {llm_path}")
            
            with open(ocr_path, 'r') as f:
                ocr_results = json.load(f)
            logging.info(f"Loaded OCR results from {ocr_path}")
            
            return llm_results, ocr_results
        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {str(e)}")
            raise
            
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison with error handling."""
        try:
            if text is None:
                return ""
            return str(text).upper().strip().replace(" ", "")
        except Exception as e:
            logging.warning(f"Error normalizing text '{text}': {str(e)}")
            return ""

    def calculate_field_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Levenshtein distance."""
        str1 = self.normalize_text(str1)
        str2 = self.normalize_text(str2)
        
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        max_len = max(len(str1), len(str2))
        distance = Levenshtein.distance(str1, str2)
        similarity = 1 - (distance / max_len)
        return similarity

    def evaluate_dataset(self, id_type: str) -> Tuple[Dict, pd.DataFrame]:
        """Evaluate OCR results against ground truth with error handling."""
        llm_path = f"ground_truth/raw_llm_responses/{id_type}_ids/results.json"
        ocr_path = f"ground_truth/ocr_results/{id_type}_ids/results.json"
        
        try:
            llm_results, ocr_results = self.load_results(llm_path, ocr_path)
            
            # Create a mapping of filenames to results
            llm_map = {r['image_id']: r['ground_truth'] for r in llm_results}
            ocr_map = {r['filename']: r for r in ocr_results}
            
            evaluation_data = []
            metrics = {
                'total_images': len(llm_results),
                'processed_images': 0,
                'field_metrics': {field: {
                    'exact_matches': 0,
                    'avg_similarity': 0.0,
                    'std_similarity': 0.0
                } for field in self.fields}
            }
            
            field_similarities = defaultdict(list)
            
            for filename in llm_map.keys():
                try:
                    if filename not in ocr_map:
                        logging.warning(f"File {filename} not found in OCR results")
                        continue
                        
                    llm_entry = llm_map[filename]
                    ocr_entry = ocr_map[filename]
                    metrics['processed_images'] += 1
                    
                    row_data = {
                        'filename': filename,
                        'image_quality': ocr_entry.get('confidence', {}).get('image_quality', 'unknown')
                    }
                    
                    for field in self.fields:
                        llm_value = llm_entry.get(field)
                        ocr_value = ocr_entry.get(field)
                        similarity = self.calculate_field_similarity(llm_value, ocr_value)
                        
                        field_similarities[field].append(similarity)
                        
                        if similarity == 1.0:
                            metrics['field_metrics'][field]['exact_matches'] += 1
                        
                        row_data.update({
                            f'{field}_gt': llm_value,
                            f'{field}_ocr': ocr_value,
                            f'{field}_similarity': similarity
                        })
                    
                    evaluation_data.append(row_data)
                    
                except Exception as e:
                    logging.error(f"Error processing file {filename}: {str(e)}")
                    self.failed_evaluations.append((filename, str(e)))
                    continue
            
            # Calculate final metrics
            processed = metrics['processed_images']
            for field in self.fields:
                similarities = field_similarities[field]
                metrics['field_metrics'][field].update({
                    'avg_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'exact_match_rate': metrics['field_metrics'][field]['exact_matches'] / processed * 100
                })
            
            return metrics, pd.DataFrame(evaluation_data)
            
        except Exception as e:
            logging.error(f"Error evaluating results for {id_type} IDs: {str(e)}")
            raise

    def generate_report(self):
        """Generate comprehensive evaluation report with error handling."""
        reports = {}
        metrics_collection = {}
        
        for id_type in ['new', 'old']:
            logging.info(f"\nEvaluating {id_type} IDs...")
            try:
                metrics, df = self.evaluate_dataset(id_type)
                reports[id_type] = df
                metrics_collection[id_type] = metrics
                
                # Save detailed results
                output_dir = f"ground_truth/evaluation_results/{id_type}_ids"
                os.makedirs(output_dir, exist_ok=True)
                
                # Save detailed CSV
                df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
                
                # Save metrics JSON
                with open(f"{output_dir}/metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                # Generate and save summary report
                self._save_summary_report(output_dir, id_type, metrics, df)
                
            except Exception as e:
                logging.error(f"Error generating report for {id_type} IDs: {str(e)}")
                continue
        
        # Save failed evaluations report
        if self.failed_evaluations:
            self._save_failed_evaluations_report()
        
        return reports, metrics_collection

    def _save_summary_report(self, output_dir: str, id_type: str, metrics: Dict, df: pd.DataFrame):
        """Save summary report to file."""
        with open(f"{output_dir}/summary_report.txt", 'w') as f:
            f.write(f"OCR Evaluation Report - {id_type.upper()} IDs\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Images: {metrics['total_images']}\n")
            f.write(f"Processed Images: {metrics['processed_images']}\n\n")
            
            f.write("Field-wise Analysis:\n")
            f.write("-" * 30 + "\n")
            for field, stats in metrics['field_metrics'].items():
                f.write(f"\n{field.upper()}:\n")
                f.write(f"Average Similarity: {stats['avg_similarity']:.2%}\n")
                f.write(f"Std Deviation: {stats['std_similarity']:.2%}\n")
                f.write(f"Exact Matches: {stats['exact_matches']}/{metrics['processed_images']} ")
                f.write(f"({stats['exact_match_rate']:.1f}%)\n")
            
            f.write("\nImage Quality Distribution:\n")
            f.write("-" * 30 + "\n")
            quality_dist = df['image_quality'].value_counts()
            for quality, count in quality_dist.items():
                f.write(f"{quality}: {count} ({count/len(df):.1%})\n")

    def _save_failed_evaluations_report(self):
        """Save failed evaluations to a separate file."""
        output_dir = "ground_truth/evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/failed_evaluations.txt", 'w') as f:
            f.write("Failed Evaluations Report\n")
            f.write("=" * 50 + "\n\n")
            for filename, error in self.failed_evaluations:
                f.write(f"File: {filename}\n")
                f.write(f"Error: {error}\n")
                f.write("-" * 30 + "\n")

def main():
    try:
        evaluator = OCREvaluator()
        reports, metrics = evaluator.generate_report()
        
        logging.info("\nEvaluation complete! Results saved in ground_truth/evaluation_results/")
        logging.info("\nOverall Performance Summary:")
        logging.info("=" * 50)
        
        for id_type, metrics_data in metrics.items():
            logging.info(f"\n{id_type.upper()} IDs:")
            logging.info("-" * 30)
            for field, stats in metrics_data['field_metrics'].items():
                logging.info(f"{field:20} Exact Match Rate: {stats['exact_match_rate']:5.1f}% "
                           f"(Avg Similarity: {stats['avg_similarity']:.1%})")
                
        if evaluator.failed_evaluations:
            logging.warning(f"\nWarning: {len(evaluator.failed_evaluations)} evaluations failed. "
                          "See failed_evaluations.txt for details.")
    
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
