import os
import csv
from collections import defaultdict


def combine_files_transposed_to_csv(top_level_directory, output_file):
    # Dictionary to store descriptions and their corresponding values from each file type and model
    data = defaultdict(lambda: defaultdict(dict))
    descriptions = defaultdict(list)
    descriptions_set = defaultdict(set)
    filenames = defaultdict(list)

    # Traverse directories starting from the top level directory
    for root, dirs, files in os.walk(top_level_directory):
        for file in files:
            if file.startswith("metrics_") and file.endswith(".txt"):
                file_path = os.path.join(root, file)
                filenames[file].append(file_path)
                with open(file_path, 'r') as f:
                    for line in f:
                        # Strip any leading/trailing whitespace characters
                        line = line.strip()
                        if ':' in line:
                            description, value = line.split(':', 1)
                            description = description.strip()
                            value = value.strip()
                            data[file][file_path][description] = value
                            if description not in descriptions_set[file]:
                                descriptions[file].append(description)
                                descriptions_set[file].add(description)

    # Write the transposed data to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write headers for each type of metrics file
        for metrics_file in filenames:
            # Write the header row with descriptions
            header = ['Model'] + descriptions[metrics_file]
            csv_writer.writerow(header)

            # Write the data rows
            for file_path in filenames[metrics_file]:
                model_name = os.path.basename(file_path).split("_")[-1].rsplit(".", 1)[0]
                row = [model_name] + [data[metrics_file][file_path].get(desc, '') for desc in
                                      descriptions[metrics_file]]
                csv_writer.writerow(row)
            # Add a blank row for separation between different metrics types
            csv_writer.writerow([])


def main():
    # Example usage
    top_level_directory = '../Classification/norm'  # Replace with your top level directory path
    output_csv = 'norm_combined_output.csv'  # Desired output CSV file name
    combine_files_transposed_to_csv(top_level_directory, output_csv)


if __name__ == "__main__":
    main()
