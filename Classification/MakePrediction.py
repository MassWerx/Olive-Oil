import os
import shutil
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pretty_html_table import build_table
from sklearn.preprocessing import OneHotEncoder

PPM_ERROR = 1000  # Parts per million error for mass tolerance


def mass_tolerance(feature_mass):
    return PPM_ERROR * feature_mass * 1e-6  # PPM tolerance


def mass_tolerance_error(theo_mass, exp_mass):
    return abs(((theo_mass - exp_mass) / theo_mass) * 1e6)


def build_missing_df(sample_mass, features_mass):
    missing = list(set(sample_mass) - set(features_mass)) + list(set(features_mass) - set(sample_mass))
    missing.sort()
    return pd.DataFrame({'mass': missing}).fillna(0).reset_index(drop=True)

def align_peaks_with_features(df_sample, features, sample_file, min_peak_height, output_dir):
    """
    Align sample peaks (df_sample) with reference features, ensuring that:
    - A new DataFrame is created with all features from the reference list.
    - Any feature not found in df_sample within PPM tolerance is set to min_peak_height.
    - If a peak in df_sample is within PPM tolerance of a feature, its intensity is used.
    - Peaks in df_sample that don't match a feature are logged.

    Parameters:
    - df_sample: The input sample data with mass peaks as columns.
    - features: The list of reference features (masses) to align against.
    - sample_file: The name of the sample file being processed.
    - min_peak_height: The default intensity value for missing peaks.

    Returns:
    - df_sample_aligned: A DataFrame with features aligned to the reference features.
    - output_miss: Debugging output indicating unused peaks in the sample.
    """

    # Initialize a DataFrame with the reference features and set all intensities to min_peak_height
    df_aligned = pd.DataFrame(min_peak_height, index=[0], columns=features)

    # Convert features and sample columns to numpy arrays for easy matching
    mass_feature_lst = np.asarray(features).astype(float)
    mass_sample_list = df_sample.columns.astype(float)

    # Track unused peaks from the sample
    unused_peaks = set(mass_sample_list)
    all_peaks = []

    # Iterate over the reference features and match to peaks in df_sample within PPM tolerance
    for feature in mass_feature_lst:
        # Find the closest sample peak to the current feature
        idx = (np.abs(mass_sample_list - feature)).argmin()
        mass_sample = mass_sample_list[idx]
        diff = abs(mass_sample - feature)

        # If the sample peak is within the mass tolerance, use its intensity
        if diff < mass_tolerance(feature):
            peak_height = df_sample[mass_sample].values[0]  # Get the peak height
            df_aligned[feature] = peak_height  # Use sample intensity
            unused_peaks.discard(mass_sample)  # This peak has been used
            ppm_error = mass_tolerance_error(feature, mass_sample)
            all_peaks.append({'feature': feature, 'sample': mass_sample, 'ppm_error': ppm_error, 'status': 'found', 'peak_height': peak_height})
        else:
            all_peaks.append({'feature': feature, 'sample': 'N/A', 'ppm_error': 'N/A', 'status': 'missing', 'peak_height': min_peak_height})

    # Add unused peaks to the all_peaks list, indicating they are in the original sample but missing from the features
    for peak in unused_peaks:
        peak_height = df_sample[peak].values[0]  # Get the peak height
        all_peaks.append({'feature': 'N/A', 'sample': peak, 'ppm_error': 'N/A', 'status': 'in_sample_not_in_features', 'peak_height': peak_height})

    # Save all peaks to a CSV file
    all_peaks_df = pd.DataFrame(all_peaks)
    all_peaks_df.to_csv(f"{output_dir}_debug_peaks.csv", index=False)

    # Return the DataFrame with aligned features, and ensure it's in the correct shape
    return df_aligned

def default_loader(path):
    return joblib.load(path)


def tf_loader(path):
    return tf.keras.models.load_model(path)


model_loaders = {
    'ANN': tf_loader,  # TensorFlow-specific loader
}


def load_models(root_dir):
    models = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_best_model.pkl'):
                algo_name = file.split('_best_model.pkl')[0]
                model_loader = model_loaders.get(algo_name, default_loader)
                model_path = os.path.join(root, file)
                try:
                    models[algo_name] = model_loader(model_path)
                    print(f"Loaded {algo_name} model from {model_path}")
                except Exception as e:
                    print(f"Error loading {algo_name} from {model_path}: {e}")
    return models


def get_files_from_directory(directory_path, extension=".csv"):
    files = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(extension):
            full_path = os.path.join(directory_path, file_name)
            files.append(full_path)
    return files


def load_data_from_file(input_file, flip):
    """
    Load data from a CSV file into a pandas DataFrame, with an option to transpose the data.
    """
    if flip:
        df = pd.read_csv(input_file, header=None).T
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    else:
        df = pd.read_csv(input_file, header=0)
    return df


def load_data_frame(input_dataframe):
    """
    Process a pandas DataFrame to extract and organize sample names, labels, features, and numeric targets.
    """
    samples = input_dataframe.iloc[:, 0]
    labels = input_dataframe.iloc[:, 1]
    data_table = input_dataframe.iloc[:, 2:]
    features = data_table.columns.values.tolist()
    label_table = pd.get_dummies(labels)
    class_names = label_table.columns.to_numpy()
    encoder = OneHotEncoder(sparse_output=False)
    oil_y_num = encoder.fit_transform(labels.to_numpy().reshape(-1, 1))
    y = np.argmax(oil_y_num, axis=1)
    X = pd.DataFrame(data_table, columns=features)
    min_peak_height = min(X.min())
    ms_info = {
        'class_names': class_names,
        'labels': labels,
        'X': X,
        'y': y,
        'samples': samples,
        'features': [float(i) for i in features],
        'feature_names': features,
        'min_peak_height' :  min_peak_height
    }
    return ms_info

def make_predictions_from_files(directory_path, root_dir, features, loaded_models, min_peak_height, classes_mass_lists):
    list_sample_files = get_files_from_directory(directory_path, extension=".csv")
    debug_dataframes = {}
    single_prediction_info = {}
    single_prediction_table = {}
    sample_updated_dataframe = {}
    missing_percent_msg = ""
    data_type = os.path.basename(root_dir)
    # Extract the directory and the file name
    dir_name = os.path.dirname(list_sample_files[0])
    # Add the 'debug' directory before the file name and append the data_type
    new_debug_dir = os.path.join(dir_name, 'debug', data_type)

    # Remove the directory if it exists
    if os.path.exists(new_debug_dir):
        shutil.rmtree(new_debug_dir)

    # Create the directory again
    os.makedirs(new_debug_dir)

    if len(list_sample_files) != 0:
        for sample_file in list_sample_files:
            file_name = os.path.basename(sample_file)
            # Combine the new directory with the file name
            debug_file_path = os.path.join(new_debug_dir, file_name)

            print(f"Updated file path: {debug_file_path}")

            # Step 1: Read and transpose the sample data
            df_sample = pd.read_csv(sample_file).T
            df_sample.columns = df_sample.iloc[0]
            df_sample = df_sample.drop(df_sample.index[0])

            # Step 2: Align the sample peaks with features
            df_sample = align_peaks_with_features(df_sample, features, sample_file, min_peak_height, f'{debug_file_path}')

            sample_updated_dataframe[sample_file] = df_sample

            df_sample.columns = [
                f"{col:.2f}" if isinstance(col, (int, float)) else str(col)
                for col in df_sample.columns
            ]
            # df_sample.columns = df_sample.columns.astype(str)

            # Step 4: Perform machine learning predictions
            prediction_results = {}
            probability_results = {}

            for model_name, model in loaded_models.items():
                try:
                    predictions = model.predict(df_sample)
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(df_sample)
                    else:
                        probabilities = None

                    prediction_results[model_name] = predictions
                    probability_results[model_name] = probabilities if probabilities is not None else "Not available"

                    print(f"Predictions for {sample_file} using {model_name}: {predictions}")
                    single_prediction_info[sample_file] = f"Predictions using {model_name}: {predictions}"
                    single_prediction_table[sample_file] = build_table(pd.DataFrame(predictions), 'blue_light')
                except Exception as e:
                    print(f"Error making predictions with {model_name} on {sample_file}: {e}")
                    prediction_results[model_name] = f"Error with {model_name}: {e}"
                    probability_results[model_name] = "Error"

            single_prediction_info[sample_file] = {"predictions": prediction_results,
                                                   "probabilities": probability_results}
    else:
        missing_percent_msg = "No samples"

    return (
        single_prediction_info,
        single_prediction_table,
        features,
        sample_updated_dataframe,
        debug_dataframes,
        missing_percent_msg
    )


def main(root_directory, sample_directory,
         training_data_file, flip, output_file):
    print(f"Loading models from directory: {root_directory}")
    loaded_models = load_models(root_directory)

    if 'AdaBoost' in loaded_models:
        print(f"Successfully loaded AdaBoost model: {loaded_models['AdaBoost']}")
    else:
        print("AdaBoost model not found in the directory.")

    if training_data_file:
        training_data = load_data_from_file(training_data_file, flip)
        ms_info = load_data_frame(training_data)
        features = ms_info['features']
        min_peak_height = ms_info['min_peak_height']  # Example value, define as needed
        classes_mass_lists = {}  # Example value, define as needed

        # Collect predictions for each model and each sample
        results = []

        # Make predictions for each sample in the directory
        single_prediction_info, single_prediction_table, features, sample_updated_dataframe, debug_dataframes, missing_percent_msg = make_predictions_from_files(
            sample_directory, root_directory, features, loaded_models, min_peak_height, classes_mass_lists
        )

        # Loop through each sample file and model, and collect the results
        for sample_file, prediction_info in single_prediction_info.items():
            if isinstance(prediction_info, dict):
                for model_name, result_data in prediction_info["predictions"].items():
                    prediction = result_data
                    probabilities = prediction_info["probabilities"][model_name]
                    results.append({
                        'Sample File': sample_file,
                        'Model Name': model_name,
                        'Prediction': prediction,
                        'Probabilities': probabilities
                    })
            else:
                print(
                    f"Unexpected format in prediction_info for sample {sample_file}. Expected dict, got {type(prediction_info)}.")

        # Create a DataFrame from the results
        df_results = pd.DataFrame(results)

        # Save the DataFrame to disk
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No training data file provided.")


def run_multiple_main_calls():
    # Arrays of values for each argument
    # root_directories = ["./oil/Adulteration_can", "./oil/Adulteration_soy" ,"./oil/Freshness", "./oil/Grade"]
    # sample_directories = ["./20230502/Muldi", "./20230502/Muldi",  "./20230502/Dart",  "./20230502/Dart"]
    """training_data_files = [
        "./oil/Adulteration_can/Adult_CAN-MALDI_TAG_unnorm_8Sep2024.csv",
        "./oil/Adulteration_soy/Adult_SOY-MALDI_TAG_unnorm_8Sep2024.csv",
        "./oil/Freshness/Freshness_PP_filt_unnorm_9Sep2024.csv",
        "./oil/Grade/Grade_PP_filt_unnorm_9Sep2024.csv"
    ]"""
    # output_files = ["prediction_results_can.csv", "prediction_results_soy.csv", "prediction_results_fresh.csv","prediction_results_grade.csv"]

    root_directories = [ "./oil/Grade"]
    sample_directories = ["./20230502/Dart"]
    training_data_files = ["./oil/Grade/Grade_PP_filt_unnorm_9Sep2024.csv"]
    output_files = ["prediction_results_grade.csv"]

    # Loop over each set of directories and files, and call main()
    for root_dir, sample_dir, training_data, output_file in zip(
            root_directories, sample_directories, training_data_files, output_files):

        print(f"Running main with root_directory={root_dir}, sample_directory={sample_dir}")
        main(root_directory=root_dir,
             sample_directory=sample_dir,
             training_data_file=training_data,
             flip=False,
             output_file=output_file)

if __name__ == "__main__":
    run_multiple_main_calls()
