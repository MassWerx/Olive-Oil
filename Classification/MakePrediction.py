import os
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


def process_sample_data(sample_features, sample_features_missed, features, df_sample, sample_file):
    df_mass_ppm = pd.DataFrame(sample_features.items(), columns=['sample_mass', 'features_mass']).apply(pd.to_numeric,
                                                                                                        errors='coerce')
    df_mass_ppm['diff'] = df_mass_ppm.apply(lambda x: abs(x.sample_mass - x.features_mass), axis=1)
    df_mass_ppm['ppm'] = df_mass_ppm.apply(lambda x: mass_tolerance_error(x.features_mass, x.sample_mass), axis=1)
    df_sample = df_sample.rename(columns=sample_features)
    df_sample = df_sample[df_sample.columns.intersection(features)]
    df_missing = build_missing_df(df_sample.columns.tolist(), features)
    df_sample_features_missed = pd.DataFrame(list(sample_features_missed.items()),
                                             columns=['sample_missed', 'feature_target']).apply(pd.to_numeric,
                                                                                                errors='coerce')
    df_sample_features_missed['diff'] = df_sample_features_missed.apply(
        lambda x: abs(x.sample_missed - x.feature_target), axis=1)
    df_sample_features_missed['ppm'] = df_sample_features_missed.apply(
        lambda x: mass_tolerance_error(x.feature_target, x.sample_missed), axis=1)
    output1 = build_table(df_mass_ppm, 'blue_light', index=True)
    output2 = build_table(df_sample_features_missed, 'blue_light', index=True)
    output_miss = f'<p><b>{sample_file} : found peaks</b></p>{output1}</p><p><b>{sample_file} : missing peaks</b></p>{output2}</p>'
    return df_sample, output_miss


def calculate_missing_percentage(df_sample, features, min_peak_height, classes_mass_lists):
    missing_percent_msg = '<p>'
    classes_mass_lists['all'] = features
    for compare_list in classes_mass_lists:
        list2 = list(map(str, classes_mass_lists[compare_list]))
        res = len(set(df_sample.columns.tolist()) & set(list2)) / float(
            len(set(df_sample.columns.tolist()) | set(list2))) * 100
        ans = f'Percentage missing of {compare_list} is: ' + str(100 - res)
        missing_percent_msg += f'{ans}<br>'
    missing_percent = 0
    for col in features:
        if col not in df_sample:
            df_sample[col] = float(min_peak_height)
            missing_percent += 1
    missing_percent = (missing_percent / len(features)) * 100
    missing_percent = "{:.2f}".format(missing_percent)
    missing_percent_msg += f' total missing is {missing_percent}</p>'
    return df_sample, missing_percent_msg

def align_peaks_with_features(df_sample, features, mass_tolerance, sample_file):
    """
    Align sample peaks (df_sample) with reference features, preserving intensities,
    and removing any unmatched features. After this step, df_sample should only
    contain valid features that are present in both df_sample and the reference features.

    Parameters:
    - df_sample: The input sample data with mass peaks as columns.
    - features: The list of reference features (masses) to align against.
    - mass_tolerance: A function that calculates the acceptable mass tolerance based on PPM.
    - sample_file: The name of the sample file being processed.

    Returns:
    - df_sample_aligned: The sample data aligned to the reference features.
    - output_miss: Debugging output indicating which peaks were found and which were missed.
    """

    # Convert features and sample columns to numpy arrays for easy matching
    mass_feature_lst = np.asarray(features).astype(float)
    mass_sample_list = df_sample.columns.astype(float)

    sample_features = {}  # To store matched sample masses with corresponding feature masses
    sample_features_missed = {}  # To store sample masses that couldn't be matched

    # Iterate through the sample masses and find the closest feature mass within tolerance
    for mass_sample in mass_sample_list:
        # Find the index of the closest feature mass to the current sample mass
        idx = (np.abs(mass_feature_lst - mass_sample)).argmin()
        mass_feature = mass_feature_lst[idx]
        diff = abs(float(mass_sample) - mass_feature)

        # Only match if the difference is within the acceptable mass tolerance
        if diff < mass_tolerance(mass_feature):
            # Store the sample mass matched to the feature mass
            sample_features[mass_sample] = str(mass_feature)
        else:
            # If no valid match is found, mark the sample mass as missed
            sample_features_missed[float(mass_sample)] = float(mass_feature)

    # Rename columns in df_sample to reflect the matched features
    df_sample = df_sample.rename(columns=sample_features)

    # Keep only the columns (features) that intersect with the known features
    df_sample = df_sample[df_sample.columns.intersection(features)]

    # Process the matched data and build output tables for debugging and reporting
    df_sample_aligned, output_miss = process_sample_data(
        sample_features, sample_features_missed, features, df_sample, sample_file
    )

    return df_sample_aligned, output_miss

def align_peaks_with_features_bad(df_sample, features, mass_tolerance, sample_file):
    mass_feature_lst = np.asarray(features).astype(float)
    mass_sample_list = df_sample.columns.astype(float)
    sample_features = {}
    sample_features_missed = {}
    for mass_sample in mass_sample_list:
        idx = (np.abs(mass_feature_lst - mass_sample)).argmin()
        mass_feature = mass_feature_lst[idx]
        diff = abs(float(mass_sample) - mass_feature)
        if diff < mass_tolerance(mass_feature):
            if str(mass_feature) in sample_features.values():
                mass_key_set = {i for i in sample_features if sample_features[i] == str(mass_feature)}
                if len(mass_key_set) != 1:
                    raise Exception(f'ERROR: Duplicate keys added {len(mass_key_set)}')
                mass_key = list(mass_key_set)[0]
                diff_key = abs(mass_key - mass_feature)
                if diff < diff_key:
                    del sample_features[mass_key]
                    sample_features_missed[float(mass_key)] = float(mass_feature)
                    sample_features[mass_sample] = str(mass_feature)
            else:
                sample_features[mass_sample] = str(mass_feature)
        else:
            sample_features_missed[float(mass_sample)] = float(mass_feature)
    return process_sample_data(sample_features, sample_features_missed, features, df_sample, sample_file)


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
    ms_info = {
        'class_names': class_names,
        'labels': labels,
        'X': pd.DataFrame(data_table, columns=features),
        'y': y,
        'samples': samples,
        'features': features,
        'feature_names': features
    }
    return ms_info


def make_predictions_from_files(directory_path, features, loaded_models, min_peak_height, classes_mass_lists):
    list_sample_files = get_files_from_directory(directory_path, extension=".csv")
    debug_dataframes = {}
    single_prediction_info = {}
    single_prediction_table = {}
    sample_updated_dataframe = {}
    missing_percent_msg = ""

    if len(list_sample_files) != 0:
        for sample_file in list_sample_files:
            # Step 1: Read and transpose the sample data
            df_sample = pd.read_csv(sample_file).T
            df_sample.columns = df_sample.iloc[0]
            df_sample = df_sample.drop(df_sample.index[0])

            # Step 2: Align the sample peaks with features
            df_sample, output_miss = align_peaks_with_features(df_sample, features, mass_tolerance, sample_file)
            debug_dataframes[sample_file] = output_miss

            # Step 3: Calculate missing percentage
            df_sample, missing_percent_msg = calculate_missing_percentage(df_sample, features, min_peak_height,
                                                                          classes_mass_lists)
            sample_updated_dataframe[sample_file] = df_sample

            # Step 4: Perform machine learning predictions
            prediction_results = {}
            probability_results = {}

            for model_name, model in loaded_models.items():
                try:
                    predictions = model.predict(df_sample.values)
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(df_sample.values)
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


def main(root_directory="./oil/Adulteration_can", sample_directory="./20230502/Muldi",
         training_data_file="./oil/Adulteration_can/Adult_CAN-MALDI_TAG_unnorm_8Sep2024.csv", flip=False,
         output_file="prediction_results.csv"):
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
        min_peak_height = 0.01  # Example value, define as needed
        classes_mass_lists = {}  # Example value, define as needed

        # Collect predictions for each model and each sample
        results = []

        # Make predictions for each sample in the directory
        single_prediction_info, single_prediction_table, features, sample_updated_dataframe, debug_dataframes, missing_percent_msg = make_predictions_from_files(
            sample_directory, features, loaded_models, min_peak_height, classes_mass_lists
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


if __name__ == "__main__":
    main()
