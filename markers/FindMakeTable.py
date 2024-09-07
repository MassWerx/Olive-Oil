import os
import csv
from collections import defaultdict


def combine_files_transposed_to_csv(top_level_directory, output_file):
    # Dictionary to store descriptions and their corresponding values from each file
    data = defaultdict(dict)
    descriptions = []
    descriptions_set = set()
    filenames = []

    # Traverse directories starting from the top level directory
    for root, dirs, files in os.walk(top_level_directory):
        for file in files:
            if file.startswith("metrics_") and file.endswith(".txt"):
                file_path = os.path.join(root, file)
                filenames.append(file_path)
                with open(file_path, 'r') as f:
                    for line in f:
                        # Strip any leading/trailing whitespace characters
                        line = line.strip()
                        if ':' in line:
                            description, value = line.split(':', 1)
                            description = description.strip()
                            value = value.strip()
                            data[file_path][description] = value
                            if description not in descriptions_set:
                                descriptions.append(description)
                                descriptions_set.add(description)

    # Write the transposed data to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header row with descriptions
        header = ['Algorithms'] + descriptions
        csv_writer.writerow(header)

        # Write the data rows
        for file_path in filenames:
            model_name = os.path.basename(file_path).split("_", 1)[-1].rsplit(".", 1)[0]
            row = [model_name] + [data[file_path].get(desc, '') for desc in descriptions]
            csv_writer.writerow(row)


"""
# Example usage
    top_level_directory = '../Classification/oil/Freshness'  # Replace with your top level directory path
    output_csv = 'fresh_3Sep2024_metrics_scikit.csv'  # Desired output CSV file name
    
    top_level_directory = '../Classification/oil/Grade'  # Replace with your top level directory path
    output_csv = 'grade_31Aug2024_metrics_scikit.csv'  # Desired output CSV file name
    
    top_level_directory = '../Classification/oil/Adulteration_can'  # Replace with your top level directory path
    output_csv = 'adult_can_29Aug2024_metrics_scikit.csv'  # Desired output CSV file name
    
    top_level_directory = '../Classification/oil/Adulteration_soy'  # Replace with your top level directory path
    output_csv = 'adult_soy_30Aug2024_metrics_scikit.csv'  # Desired output CSV file name
"""

def main():
    # Example usage
    top_level_directory = '../Classification/oil/Grade'  # Replace with your top level directory path
    output_csv = 'grade_31Aug2024_metrics_scikit.csv'  # Desired output CSV file name
    combine_files_transposed_to_csv(top_level_directory, output_csv)


if __name__ == "__main__":
    main()
