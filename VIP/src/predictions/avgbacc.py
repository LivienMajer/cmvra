import json
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def concatenate_metrics(metric_files):
    all_metrics = {}
    for file in metric_files:
        try:
            metrics = load_metrics(file)
            for modality, data in metrics.items():
                if modality not in all_metrics:
                    all_metrics[modality] = {'predictions': [], 'labels': []}
                all_metrics[modality]['predictions'].extend(data['predictions'])
                all_metrics[modality]['labels'].extend(data['labels'])
        except FileNotFoundError:
            logging.warning(f"File not found: {file}")
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON in file: {file}")
    return all_metrics

def compute_balanced_accuracy(metrics):
    results = {}
    for modality, data in metrics.items():
        balanced_acc = balanced_accuracy_score(data['labels'], data['predictions'])
        results[modality] = balanced_acc
    return results

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    metric_files = config.get('metric_files', [])
    output_file = config.get('output_file', 'combined_balanced_accuracies.json')

    if not metric_files:
        logging.error("No metric files specified in the configuration.")
        return

    # Load and concatenate metrics
    concatenated_metrics = concatenate_metrics(metric_files)

    # Compute balanced accuracy
    balanced_accuracies = compute_balanced_accuracy(concatenated_metrics)

    # Print results
    logging.info("Balanced Accuracies:")
    for modality, acc in balanced_accuracies.items():
        logging.info(f"{modality}: {acc:.4f}")

    # Save results to a file
    with open(output_file, 'w') as f:
        json.dump(balanced_accuracies, f, indent=4)
    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    config_path = "/home/bas06400/Thesis/VIP/src/predictions/metrics_config.json"  # You can change this to use a command-line argument if needed
    main(config_path)