import os
import re
from typing import Dict, List, Tuple
from datetime import datetime
import math

def parse_log_file(file_path: str) -> Dict[str, List[float]]:
    results = {
        'rgb': {'accuracy': [], 'balanced_accuracy': []},
        'ir': {'accuracy': [], 'balanced_accuracy': []}
    }
    
    with open(file_path, 'r') as file:
        content = file.read()
        
    patterns = [
        r"Test Set: Modality: (\w+), Accuracy: ([\d.]+), Balanced Accuracy: ([\d.]+)"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            modality, accuracy, balanced_accuracy = match
            if modality in ['rgb', 'ir']:
                results[modality]['accuracy'].append(float(accuracy))
                results[modality]['balanced_accuracy'].append(float(balanced_accuracy))
    
    return results

def calculate_statistics(data: List[float]) -> Tuple[float, float]:
    n = len(data)
    if n == 0:
        return 0, 0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)
    return mean, std_dev

def main(log_directory: str, output_log_path: str):
    all_results = {
        'rgb': {'accuracy': [], 'balanced_accuracy': []},
        'ir': {'accuracy': [], 'balanced_accuracy': []}
    }
    
    for filename in os.listdir(log_directory):
        if filename.endswith('.log'):
            file_path = os.path.join(log_directory, filename)
            results = parse_log_file(file_path)
            
            for modality in ['rgb', 'ir']:
                all_results[modality]['accuracy'].extend(results[modality]['accuracy'])
                all_results[modality]['balanced_accuracy'].extend(results[modality]['balanced_accuracy'])
    
    with open(output_log_path, 'w') as output_file:
        output_file.write(f"Log Analysis Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write(f"Analyzed directory: {log_directory}\n\n")
        
        for modality in ['rgb', 'ir']:
            avg_accuracy, std_accuracy = calculate_statistics(all_results[modality]['accuracy'])
            avg_balanced_accuracy, std_balanced_accuracy = calculate_statistics(all_results[modality]['balanced_accuracy'])
            
            result_str = (f"{modality.upper()}:\n"
                          f"  Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n"
                          f"  Balanced Accuracy: {avg_balanced_accuracy:.4f} ± {std_balanced_accuracy:.4f}")
            print(result_str)
            output_file.write(result_str + "\n\n")
        
        total_files = sum(len(data['accuracy']) for data in all_results.values())
        output_file.write(f"Total files analyzed: {total_files}")
    
    print(f"\nResults have been saved to {output_log_path}")

if __name__ == "__main__":
    log_directory = "/home/bas06400/Thesis/VIP/src/zlogs/zs/rgb_ir"  
    output_log_path = "/home/bas06400/Thesis/VIP/src/zlogs/zs/rgb_ir/ir_analysis_results.log"  
    main(log_directory, output_log_path)