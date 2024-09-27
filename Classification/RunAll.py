import argparse
import multiprocessing
import subprocess
import os
import csv
from collections import defaultdict
# Function to run a command and harvest the files

def combine_files_transposed_to_csv(top_level_directory, output_file):
    # Dictionary to store descriptions and their corresponding values from each file
    data = defaultdict(dict)
    descriptions = []
    descriptions_set = set()
    filenames = []

    # Traverse directories starting from the top level directory
    for root, dirs, files in os.walk(top_level_directory):
        for file in files:
            if file.startswith("metrics_cv_") and file.endswith(".txt"):
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




def run_command_and_harvest(command, top_level_directory, output_csv):
    print(f"Changing directory to: {top_level_directory}")
    os.chdir(top_level_directory)  # Change to the top-level directory
    # Run the command
    print(f"Running command: {command}")
    subprocess.run(command)


    base_dir = ".."
    # After command finishes, combine metrics files
    print(f"Harvesting metrics from {base_dir} into {output_csv}")
    combine_files_transposed_to_csv(base_dir, output_csv)

if __name__ == "__main__":
    # List of jobs and their corresponding directories and outputs
    # List of commands and their corresponding directories and outputs
    jobs = [
        {
            'command': ['python', '../../GridClassFinal.py', 'Adult_CAN-MALDI_TAG_unnorm_8Sep2024.csv', 'Boruta',
                        'false', 'true', 'false'],
            'top_level_directory': '../Classification/oil/Adulteration_can',
            'output_csv': 'adult_can_8Sep2024_metrics_scikit.csv'
        },
        {
            'command': ['python', '../../GridClassFinal.py', 'Adult_SOY-MALDI_TAG_unnorm_8Sep2024.csv', 'Boruta',
                        'false', 'true', 'false'],
            'top_level_directory': '../Classification/oil/Adulteration_soy',
            'output_csv': 'adult_soy_8Sep2024_metrics_scikit.csv'
        },
        {
            'command': ['python', '../../GridClassFinal.py', 'Freshness_PP_filt_unnorm_9Sep2024.csv', 'Boruta', 'false',
                        'true', 'false'],
            'top_level_directory': '../Classification/oil/Freshness',
            'output_csv': 'fresh_9Sep2024_metrics_scikit.csv'
        },
        {
            'command': ['python', '../../GridClassFinal.py', 'Grade_PP_filt_unnorm_9Sep2024.csv', 'Boruta', 'false',
                        'true', 'false'],
            'top_level_directory': '../Classification/oil/Grade',
            'output_csv': 'grade_9Sep2024_metrics_scikit.csv'
        }
    ]
    # Create a process pool to run each command in parallel
    processes = []

    for job in jobs:
        # Create a separate process for each job
        p = multiprocessing.Process(
            target=run_command_and_harvest,
            args=(job['command'], job['top_level_directory'], job['output_csv'])
        )
        p.start()  # Start the process
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("All commands completed and files harvested.")