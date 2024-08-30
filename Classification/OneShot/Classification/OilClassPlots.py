# https://github.com/scikit-learn-contrib/boruta_py
import json

import pandas as pd
import numpy as np
import argparse
import joblib
import matplotlib.pyplot as plt

import pickle
import sklearn
import sys
import os
import tensorflow as tf
import shap

from scikeras.wrappers import KerasClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, Normalizer
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from pathlib import Path
from tempfile import mkdtemp
import argparse
import random
import shutil
import plotly.express as px
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, recall_score, f1_score, precision_score
from sklearn.svm import SVC
from joblib import Memory

# Graphics
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from PIL import Image
from plotly.subplots import make_subplots

from zipfile import ZipFile
from os.path import basename

# We must use verson 1.5.0
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The scikit-learn version is {}.'.format(joblib.__version__))

# tests
import unittest

print("Python Version:-", sys.version)
print("Pandas Version:-", pd.__version__)
print("SKLearn Version:-", sklearn.__version__)

# Get the current working directory
current_working_dir = os.getcwd()


# create a ZipFile object
def create_zip_file_output(output_file_name, base_dir):
    with ZipFile(f'{output_file_name}.zip', 'w') as zipObj:
        for folderName, subfolders, filenames in os.walk(base_dir):
            for filename in filenames:
                file_path = os.path.join(folderName, filename)
                zipObj.write(file_path, basename(file_path))


# replace file with transposed file
def load_data_from_file(input_file, flip):
    if (flip):
        df = pd.read_csv(input_file, header=None).T
        # add the header
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    else:
        df = pd.read_csv(input_file, header=0)

    return df


# Load Dataframe from file
def load_data_frame(input_dataframe):
    """
    Load Dataframe from file.

    Args:
        input_dataframe (pd.DataFrame): The input dataframe prepared for this module.

    Returns:
        dict: A dictionary containing the following information:
            - 'class_names' (ndarray): An array of class names (labels).
            - 'labels' (Series): The labels from the input dataframe.
            - 'X' (DataFrame): The feature matrix extracted from the input dataframe.
            - 'y' (ndarray): The encoded labels as an array.
            - 'samples' (Series): The sample names from the input dataframe.
            - 'features' (Index): The features (column names) of the input dataframe.
            - 'feature_names' (ndarray): An array of feature names.
            - 'classes_mass_list' (dict): A dictionary containing class mass lists.

    """
    train_oil_file = input_dataframe.copy()

    # Added to understand SHAP bug
    # Ensure all numeric columns are properly typed as float64
    data_table = train_oil_file.iloc[:, 2:]
    data_table = data_table.apply(pd.to_numeric, errors='coerce')

    samples = train_oil_file.iloc[:, 0]
    labels = train_oil_file.iloc[:, 1]
    # data_table = train_oil_file.iloc[:, 2:]
    features = train_oil_file.iloc[:, 2:].columns
    features_names = features.to_numpy().astype(str)
    label_table = pd.get_dummies(labels)
    class_names = label_table.columns.to_numpy()

    # features_names = features.to_numpy().astype(str)
    # Format feature names to have only two significant digits
    # features_names = np.array([f'{float(f):.2f}' for f in features])

    label_table = pd.get_dummies(labels)
    class_names = label_table.columns.to_numpy()

    # Full set
    X = data_table
    encoder = OneHotEncoder(sparse_output=False)
    oil_y_num = encoder.fit_transform(labels.to_numpy().reshape(-1, 1))
    y = np.argmax(oil_y_num, axis=1)

    ms_info = dict()
    ms_info['class_names'] = class_names
    ms_info['labels'] = labels
    ms_info['X'] = pd.DataFrame(X, columns=features_names)
    ms_info['y'] = y
    ms_info['samples'] = samples
    ms_info['features'] = features
    ms_info['feature_names'] = features_names

    print(f' shape of X values {X.shape}')
    print(f' shape of y values {y.shape}')
    return ms_info


def build_tf_model(n_features, n_classes, seed):
    """ Create a TF Model
  Args:
    Xn (Pandas dataframe):  features after elimiination
    n_classes () : lables
    seed (number) : remove random resaults
  Attributes:

  Raises:

  Returns:
    a TensorFlow Model
  """

    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = seed

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
    # for later versions:
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    """from keras import backend as K
  session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
  sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
  K.set_session(sess)"""
    # for later versions:
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)

    drop_out_seed = 0

    # TensorFlow Paramaters
    TF_DROPOUT_RATE = .2  # current from report .3
    TF_EPOCHS = 220  # current from report 140
    TF_LEARN_RATE = 0.001
    TF_NUM_OF_NEURONS = 40  # current from report 32

    tf_model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(TF_NUM_OF_NEURONS, activation=tf.nn.relu),  # input shape required train_x
        tf.keras.layers.Dropout(TF_DROPOUT_RATE, noise_shape=None, seed=drop_out_seed),
        tf.keras.layers.Dense(TF_NUM_OF_NEURONS, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                              activation=tf.nn.relu),
        tf.keras.layers.Dense(n_classes, activation='softmax')])

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(TF_LEARN_RATE,
                                                                 decay_steps=2,  #1, #2, # train_x.size * 1000,
                                                                 decay_rate=1,  #.5, #1,
                                                                 staircase=False)

    tf_optimizer = tf.keras.optimizers.Adam()  #lr_schedule)

    tf_model.compile(optimizer=tf_optimizer,
                     # loss='categorical_crossentropy',
                     loss='sparse_categorical_crossentropy',
                     metrics=['acc'])

    return tf_model


class MyKerasClf():
    """
  Custom Keras Classifier.

  Args:
      n_classes (int): The number of classes in the classification problem.
      seed (int): The seed value for random number generation.

  Attributes:
      n_classes (int): The number of classes in the classification problem.
      seed (int): The seed value for random number generation.

  Methods:
      predict(X):
          Predicts the class labels for the given input data.

      create_model(learn_rate=0.01, weight_constraint=0):
          Creates a Keras model with the specified learning rate and weight constraint.

      fit(X, y, **kwargs):
          Fits the Keras model to the given input data and labels.

      predict_proba(X):
          Predicts class probabilities for the given input data.

  """

    def __init__(self, n_classes, seed) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.seed = seed

    def predict(self, X):
        y_pred_nn = self.clf.predict(X)
        return np.array(y_pred_nn).flatten()

    def create_model(self, learn_rate=0.01, weight_constraint=0):
        model = build_tf_model(self.input_shape, self.n_classes, self.seed)
        return model

    def fit(self, X, y, **kwargs):
        self.input_shape = X.shape[1]
        self.classes_ = np.unique(y)
        self.clf = KerasClassifier(model=self.create_model(), verbose=0, epochs=220, batch_size=100)
        self.clf.fit(X, y, **kwargs)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def get_params(self, deep=True):
        return {"n_classes": self.n_classes, "seed": self.seed}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def get_model(model_name, y):
    output_dir = os.path.join(current_working_dir, 'output')
    ml_algo_model = {
        'RandomForest': RandomForestClassifier(n_jobs=-1, random_state=seed),
        'TensorFlow': MyKerasClf(n_classes=len(np.unique(y)), seed=seed),
        'SVM': SVC(probability=True, random_state=seed),
        'GradientBoosting': GradientBoostingClassifier(random_state=seed),
    }

    best_params = {
        'RandomForest': {'oob_score': True, 'max_features': 'sqrt', 'n_estimators': 100},
        'TensorFlow': {'learn_rate': .001, 'weight_constraint': 1},
        'SVM': {'C': 1, 'kernel': 'rbf'},
        'GradientBoosting': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators' : 100},
    }

    if model_name not in ml_algo_model:
        raise KeyError(f"Model '{model_name}' is not in the list of available models: {list(ml_algo_model.keys())}")

    if model_name not in best_params:
        raise KeyError(
            f"Parameters for model '{model_name}' are not defined. Available keys are: {list(best_params.keys())}")

    # Set input, name, model, best hyper parameters
    model = ml_algo_model[model_name]
    best_params_algo = best_params[model_name]
    model.set_params(**best_params_algo)

    # Write out params
    hyperparameters = model.get_params()
    with open(f'{output_dir}/params_{model_name}.txt', 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

    return model


def get_feature_reduction(feature_reduce_choice):
    reduction_fun = None
    match feature_reduce_choice:
        case None:
            print("No Feature Reduction")
        case 'Boruta':
            np.int = np.int32
            np.float = np.float64
            np.bool = np.bool_
            rf_reducer = RandomForestClassifier(n_jobs=-1, max_depth=5)
            print('Using Boruta')
            boruta = BorutaPy(rf_reducer, n_estimators='auto', verbose=0, random_state=seed)
            reduction_fun = boruta
        case _:
            print("Please select a valid Feature Reduction Method")
            quit()
    return reduction_fun


def gen_output(results, model_name, ms_file_name):
    output_dir = os.path.join(current_working_dir, 'output')
    result_metrics, result_roc, result_shap = results
    # Save fpr, tpr, and thresholds to CSV
    mean_balanced_accuracy, std_balanced_accuracy, mean_recall, std_recall, mean_f1, std_f1, mean_precision, std_precision = result_metrics
    roc_auc, fpr, tpr, thresholds = result_roc
    roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
    roc_data.to_csv(f'output/roc_data_{model_name}.csv', index=False)

    # Save metrics and best parameters to file
    with open(os.path.join(output_dir, f'metrics_{ms_file_name}.txt'), 'w') as f:
        f.write(f'Mean balanced accuracy: {mean_balanced_accuracy}\n')
        f.write(f'Standard deviation of balanced accuracy: {std_balanced_accuracy}\n')
        f.write(f'Mean recall: {mean_recall}\n')
        f.write(f'Standard deviation of recall: {std_recall}\n')
        f.write(f'Mean F1 score: {mean_f1}\n')
        f.write(f'Standard deviation of F1 score: {std_f1}\n')
        f.write(f'Mean precision: {mean_precision}\n')
        f.write(f'Standard deviation of precision: {std_precision}\n')
        f.write(f'AUC: {roc_auc}\n')
        #with open(os.path.join(current_working_dir, f'output_{name}/overall_{name}.txt'), 'w') as f:
        #   f.write(f'Overall So Far: {overall_results}\n')

    if not os.path.exists(os.path.join(current_working_dir, 'zipFiles')):
        os.makedirs(os.path.join(current_working_dir, 'zipFiles'))

    create_zip_file_output(os.path.join(current_working_dir, f'zipFiles/{model_name}_{ms_file_name}'), output_dir)
    #files.download(os.path.join(current_working_dir, f'zipFiles/{model_name}_{ms_file_name}.zip'))


def plot_roc(results, model_name):
    output_dir = os.path.join(current_working_dir, 'output')
    result_metrics, result_roc, result_shap = results
    roc_auc, fpr, tpr, thresholds = result_roc
    # ROC
    # Generate the plot with markers
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=2100, height=1500  # Set size to match 7x5 inches at 300 DPI
    )

    # Add the diagonal line
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    # Customize axes
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    # Customize axes and text sizes
    fig.update_layout(
        title=dict(font=dict(size=48)),  # Adjust title font size
        xaxis=dict(title=dict(font=dict(size=48)), tickfont=dict(size=40)),  # Adjust x-axis title and tick font size
        yaxis=dict(title=dict(font=dict(size=48)), tickfont=dict(size=40)),  # Adjust y-axis title and tick font size
        margin=dict(l=80, r=40, t=80, b=40)
    )

    fig.write_image(f'{output_dir}/{model_name}_ROC.png')


def plot_shap_values(results, model_name, ms_file_name):
    output_dir = os.path.join(os.getcwd(), 'output')
    result_metrics, result_roc, result_shap = results
    shap_values_combined, X_test_combined, selected_features_final = result_shap

    # Since you have binary classification, we'll use SHAP values for class 0 (or you could use class 1)
    shap_values_combined = shap_values_combined[:, :, 0]

    # Ensure consistency in the shapes before plotting
    print(f"Shape of shap_values_combined: {shap_values_combined.shape}")
    print(f"Shape of X_test_combined: {X_test_combined.shape}")
    print(f"Length of selected_features_final: {len(selected_features_final)}")

    ### 1. SHAP Summary Plot for All Features ###
    plt.figure()
    shap.summary_plot(shap_values_combined, X_test_combined, feature_names=selected_features_final, show=False)
    plt.savefig(f'{output_dir}/shap_summary_plot_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_summary_plot_all_folds.png'")

    ### 2. SHAP Beeswarm Plot for All Features ###
    plt.figure()
    shap.summary_plot(shap_values_combined, X_test_combined, feature_names=selected_features_final, plot_type="dot",
                      show=False)
    plt.savefig(f'{output_dir}/shap_beeswarm_plot_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_beeswarm_plot_all_folds.png'")

    ### 3. Bar Plot for Top 20 Features ###
    # Sorting features based on their importance for one sample (e.g., the first sample)
    sample_index = 0
    shap_values_for_sample = shap_values_combined[sample_index]
    sorted_idx = np.argsort(np.abs(shap_values_for_sample))[::-1]

    # Select top 20 features for bar plot and dependence plots
    top_n_features = min(20, len(sorted_idx))  # Limit to 20 features
    top_features = sorted_idx[:top_n_features]
    sorted_features = np.array(selected_features_final)[top_features]

    # Filter X_test_combined to include only the top features
    X_test_filtered = X_test_combined.iloc[:, top_features]

    plt.figure()
    shap.summary_plot(shap_values_combined[:, top_features], X_test_filtered, feature_names=sorted_features,
                      plot_type="bar", show=False)
    plt.savefig(f'{output_dir}/shap_bar_plot_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_bar_plot_all_folds.png'")

    ### 4. Custom Bar Plot for Top 20 Features (Based on Sample 0) ###
    plt.figure(figsize=(10, 8))

    # Here we directly use the SHAP values (including both positive and negative values)
    shap_values_to_plot = shap_values_for_sample[top_features]
    colors = ['red' if val > 0 else 'blue' for val in shap_values_to_plot]

    # Plot the bar plot with both positive and negative values
    plt.barh(sorted_features, shap_values_to_plot, color=colors)
    plt.xlabel('SHAP value')
    plt.title('Top SHAP Values for a Single Sample')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar_plot_custom_sample_0.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_bar_plot_custom_sample_0.png'")

    ### 5. Dependence Plots for Top 3 Features ###
    for i in range(min(3, top_n_features)):  # Safely get the top 3 features, or fewer if less
        feature_idx = top_features[i]  # This is the index within the original set
        plt.figure()
        shap.dependence_plot(feature_idx, shap_values_combined, X_test_combined, feature_names=selected_features_final,
                             show=False)
        plt.savefig(f'{output_dir}/shap_dependence_plot_{sorted_features[i]}.png', dpi=300, bbox_inches='tight')
        plt.close()
    print("Summary plot for all features saved as 'shap_dependence_plots'")

    ### 6. Combined Custom Bar Plot and Summary Plot ###
    bar_plot = Image.open(f'{output_dir}/shap_bar_plot_custom_sample_0.png')
    summary_plot = Image.open(f'{output_dir}/shap_summary_plot_all_folds.png')

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(bar_plot)
    axes[0].axis('off')
    axes[1].imshow(summary_plot)
    axes[1].axis('off')

    fig.suptitle('SHAP Bar Plot and Summary Plot', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_combined_plots.png', dpi=300)
    plt.close()
    print("Summary plot for all features saved as 'shap_combined_plots.png'")

    ### 7. SHAP Waterfall Plot for a Single Sample ###
    base_value = shap_values_combined.mean(axis=0).mean()  # Approximate base value
    shap_explanation = shap.Explanation(values=shap_values_to_plot, base_values=base_value,
                                        feature_names=sorted_features)

    plt.figure()
    shap.waterfall_plot(shap_explanation, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_waterfall_plot_sample_{sample_index}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_waterfall_plot_sample.png'")

    ### 8. SHAP Force Plot for a Single Sample ###
    plt.figure()
    shap.force_plot(base_value, shap_values_to_plot, sorted_features, matplotlib=True, show=False)
    plt.savefig(f'{output_dir}/shap_force_plot_sample_{sample_index}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_force_plot_sample.png'")

    ### 9. SHAP Decision Plot for Multiple Samples ###
    plt.figure()
    shap.decision_plot(base_value, shap_values_combined[:, top_features], sorted_features, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_decision_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_decision_plot.png'")


# Example usage (assuming you have these variables defined)
# plot_shap_values(results, model_name, ms_file_name)


def plot_shap_values_old(results, model_name, ms_file_name):
    output_dir = os.path.join(os.getcwd(), 'output')
    result_metrics, result_roc, result_shap = results
    shap_values_combined, X_test_combined, selected_features_final = result_shap

    # Since you have binary classification, we'll use SHAP values for class 0 (or you could use class 1)
    shap_values_combined = shap_values_combined[:, :, 0]

    # Ensure consistency in the shapes before plotting
    print(f"Shape of shap_values_combined: {shap_values_combined.shape}")
    print(f"Shape of X_test_combined: {X_test_combined.shape}")
    print(f"Length of selected_features_final: {len(selected_features_final)}")

    # Create the summary plot for all features
    plt.figure()
    shap.summary_plot(shap_values_combined, X_test_combined, feature_names=selected_features_final, show=False)
    plt.savefig(f'{output_dir}/shap_summary_plot_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_summary_plot_all_folds.png'")

    # Sorting features based on their importance for one sample (for illustration)
    sample_index = 0
    shap_values_for_sample = shap_values_combined[sample_index]
    sorted_idx = np.argsort(np.abs(shap_values_for_sample))[::-1]

    # Select top features for bar plot and dependence plots
    top_n_features = min(20, len(sorted_idx))  # Limit to 20 features or fewer if less
    top_features = sorted_idx[:top_n_features]
    sorted_features = np.array(selected_features_final)[top_features]

    # Use .iloc to correctly index by integer location
    X_test_filtered = X_test_combined.iloc[:, top_features]

    # Create the bar plot for the top features
    plt.figure()
    shap.summary_plot(shap_values_combined[:, top_features], X_test_filtered,
                      feature_names=sorted_features, plot_type="bar", show=False)
    plt.savefig(f'{output_dir}/shap_bar_plot_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Bar plot for top features saved as 'shap_bar_plot_all_folds.png'")

    # Custom bar plot for top features
    plt.figure(figsize=(10, 8))
    colors = ['red' if shap_values_for_sample[idx] > 0 else 'blue' for idx in top_features]
    plt.barh(sorted_features, np.abs(shap_values_for_sample[top_features]), color=colors)
    plt.xlabel('SHAP value for Sample 0')
    plt.title('Top SHAP Values for a Single Sample')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar_plot_custom_sample_0.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Custom bar plot for sample 0 saved as 'shap_bar_plot_custom_sample_0.png'")

    # Create dependence plots for the top 3 features
    """for feature_idx in top_features[:3]:  # Get indices of top 3 features
        plt.figure()
        shap.dependence_plot(feature_idx, shap_values_combined, X_test_combined,
                             feature_names=selected_features_final, show=False)
        plt.savefig(f'{output_dir}/shap_dependence_plot_{sorted_features[feature_idx]}.png', dpi=300,
                    bbox_inches='tight')
        plt.close()
        print(f"Dependence plot for {sorted_features[feature_idx]} saved")"""

    # Combine the custom bar plot and summary plot into one image
    bar_plot = Image.open(f'{output_dir}/shap_bar_plot_custom_sample_0.png')
    summary_plot = Image.open(f'{output_dir}/shap_summary_plot_all_folds.png')

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(bar_plot)
    axes[0].axis('off')
    axes[1].imshow(summary_plot)
    axes[1].axis('off')

    fig.suptitle('SHAP Bar Plot and Summary Plot', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_combined_plots.png', dpi=300)
    plt.show()
    print("Combined SHAP bar plot and summary plot saved as 'shap_combined_plots.png'")


def safe_log10(num):
    return np.log10(np.clip(num, a_min=1e-9, a_max=None))  # Clip values to avoid log10(0)


def run_model_score(ms_info, model_name, ms_file_name, feature_reduce_choice, normalize_select, log10_select, seed=42):
    np.int = np.int32
    np.float = np.float64
    np.bool = np.bool_

    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']

    print(f'X shape {X.shape}')
    current_working_dir = os.getcwd()
    output_dir = os.path.join(current_working_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    print(f'Starting {model_name}')
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)

    log10_pipe = FunctionTransformer(np.log10) if log10_select else 'passthrough'
    norm_pipe = Normalizer(norm='l1') if normalize_select else 'passthrough'

    pipeline = Pipeline([
        ('normalize', norm_pipe),
        ('transform', log10_pipe),
        ('scaler', StandardScaler()),
        ('Reduction', get_feature_reduction(feature_reduce_choice)),
        ('classifier', RandomForestClassifier())
    ], memory=memory)

    selected_features_union = set()
    selected_features_intersection = None

    # Perform cross-validation and feature selection
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)

        # Get the features used after reduction
        reduction_step = pipeline.named_steps['Reduction']
        if hasattr(reduction_step, 'support_'):
            support_mask = reduction_step.support_
            selected_features = [features[i] for i, flag in enumerate(support_mask) if flag]
        else:
            selected_features = features  # No reduction step, use all features

        selected_features_union.update(selected_features)

        if selected_features_intersection is None:
            selected_features_intersection = set(selected_features)
        else:
            selected_features_intersection.intersection_update(selected_features)

    # Convert the sets to sorted lists before saving
    selected_features_union_list = sorted(list(selected_features_union))
    selected_features_intersection_list = sorted(list(selected_features_intersection))

    # Save the union of selected features
    selected_union_features_file = os.path.join(output_dir, 'selected_features_union.csv')
    pd.Series(selected_features_union_list).to_csv(selected_union_features_file, index=False)
    print(f"Union of selected features saved to '{selected_union_features_file}'.")

    # Save the intersection of selected features
    selected_intersection_features_file = os.path.join(output_dir, 'selected_features_intersection.csv')
    pd.Series(selected_features_intersection_list).to_csv(selected_intersection_features_file, index=False)
    print(f"Intersection of selected features saved to '{selected_intersection_features_file}'.")

    # Retrain the model on the entire dataset using the intersection of features
    selected_features_final = selected_features_intersection_list
    X_filtered = X[selected_features_final]
    pipeline.fit(X_filtered, y)

    # Use TreeExplainer for tree-based models like RandomForest
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(X_filtered)

    # Handle multiclass SHAP values if necessary
    shap_values_to_plot = shap_values[:, :, 0]

    # Save SHAP summary plot
    shap.summary_plot(shap_values_to_plot, X_filtered, feature_names=selected_features_final, show=False)
    plt.savefig(os.path.join(output_dir, f"{ms_file_name}_shap_summary.png"))
    plt.close()  # Close the plot to free memory

    # Evaluate model using cross-validation on final feature set
    scoring = ['balanced_accuracy', 'recall', 'f1', 'precision', 'roc_auc']
    cv_results = cross_validate(pipeline, X_filtered, y, cv=outer_cv, scoring=scoring, return_train_score=False,
                                n_jobs=-1)

    # Compute mean and standard deviation for each metric
    mean_balanced_accuracy = np.mean(cv_results['test_balanced_accuracy'])
    std_balanced_accuracy = np.std(cv_results['test_balanced_accuracy'])

    mean_recall = np.mean(cv_results['test_recall'])
    std_recall = np.std(cv_results['test_recall'])

    mean_f1 = np.mean(cv_results['test_f1'])
    std_f1 = np.std(cv_results['test_f1'])

    mean_precision = np.mean(cv_results['test_precision'])
    std_precision = np.std(cv_results['test_precision'])

    mean_roc_auc = np.mean(cv_results['test_roc_auc'])

    memory.clear(warn=False)

    return (mean_balanced_accuracy, std_balanced_accuracy, mean_recall, std_recall, mean_f1, std_f1, mean_precision,
            std_precision), (mean_roc_auc, None, None, None), (shap_values, X_filtered, selected_features_final)


def run_model_intersection(ms_info, model, ms_file_name, feature_reduce_choice, normalize_select, log10_select):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']

    print(f'X shape {X.shape}')
    current_working_dir = os.getcwd()
    output_dir = os.path.join(current_working_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    print(f'Starting {model}')
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)

    log10_pipe = FunctionTransformer(np.log10) if log10_select else 'passthrough'
    norm_pipe = Normalizer(norm='l1') if normalize_select else 'passthrough'

    print(f'Starting SHAP generation for {model}')

    pipeline = Pipeline([
        ('normalize', norm_pipe),
        ('transform', log10_pipe),
        ('scaler', StandardScaler()),
        ('Reduction', get_feature_reduction(feature_reduce_choice)),
        ('classifier', model)
    ], memory=memory)

    all_y_true = []
    all_y_scores = []
    all_balanced_accuracies = []
    all_recalls = []
    all_f1_scores = []
    all_precisions = []

    selected_features_union = set()
    selected_features_intersection = None

    # List to store rows of features
    feature_selection_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_scores.extend(y_score)
        all_balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        all_recalls.append(recall_score(y_test, y_pred))
        all_f1_scores.append(f1_score(y_test, y_pred))
        all_precisions.append(precision_score(y_test, y_pred))

        # Get the features used after reduction
        reduction_step = pipeline.named_steps['Reduction']
        if hasattr(reduction_step, 'support_'):
            support_mask = reduction_step.support_
            selected_features = [features[i] for i, flag in enumerate(support_mask) if flag]
        else:
            selected_features = features  # No reduction step, use all features

        selected_features_union.update(selected_features)  # Track union of all selected features

        if selected_features_intersection is None:
            selected_features_intersection = set(selected_features)
        else:
            selected_features_intersection.intersection_update(selected_features)

        # Add the selected features as a new row
        feature_selection_rows.append(selected_features)#[str(fold_idx)] + selected_features)

    # Convert the sets to sorted lists before saving
    selected_features_union_list = sorted(list(selected_features_union))
    selected_features_intersection_list = sorted(list(selected_features_intersection))

    # Save the union of selected features
    selected_union_features_file = os.path.join(output_dir, 'selected_features_union.csv')
    pd.Series(selected_features_union_list).to_csv(selected_union_features_file, index=False)
    print(f"Union of selected features saved to '{selected_union_features_file}'.")

    # Save the intersection of selected features
    selected_intersection_features_file = os.path.join(output_dir, 'selected_features_intersection.csv')
    pd.Series(selected_features_intersection_list).to_csv(selected_intersection_features_file, index=False)
    print(f"Intersection of selected features saved to '{selected_intersection_features_file}'.")

    # Determine the maximum number of columns needed
    max_features = max(len(row) for row in feature_selection_rows)

    # Create a DataFrame with dynamic columns
    column_names = ['Fold'] + [f'Feature_{i + 1}' for i in range(max_features - 1)]
    feature_selection_df = pd.DataFrame(feature_selection_rows, columns=column_names)

    # Save the feature selection DataFrame to a CSV file
    feature_selection_file = os.path.join(output_dir, 'feature_selection_per_fold.csv')
    feature_selection_df.to_csv(feature_selection_file, index=False)
    print(f"Feature selection per fold saved to '{feature_selection_file}'.")

    # Use the intersection of selected features for SHAP
    """selected_features_final = selected_features_union_list
    X_filtered = X[selected_features_final]"""

    # Retrain the model on the entire dataset using the intersection of features
    selected_features_final = selected_features_intersection_list
    X_filtered = X[selected_features_final]
    pipeline.fit(X_filtered, y)

    # Use TreeExplainer for tree-based models like RandomForest
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values_class_1 = explainer.shap_values(X_filtered)

    # Generate SHAP values
    """explainer = shap.Explainer(pipeline.named_steps['classifier'], selected_features_final)
    shap_values_class_1 = explainer(selected_features_final)"""

    # Calculate mean and standard deviation for metrics
    mean_balanced_accuracy = np.mean(all_balanced_accuracies)
    std_balanced_accuracy = np.std(all_balanced_accuracies)

    mean_recall = np.mean(all_recalls)
    std_recall = np.std(all_recalls)

    mean_f1 = np.mean(all_f1_scores)
    std_f1 = np.std(all_f1_scores)

    mean_precision = np.mean(all_precisions)
    std_precision = np.std(all_precisions)

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_scores)
    roc_auc = auc(fpr, tpr)

    memory.clear(warn=False)

    return (mean_balanced_accuracy, std_balanced_accuracy, mean_recall, std_recall, mean_f1, std_f1, mean_precision,
            std_precision), (roc_auc, fpr, tpr, thresholds), (shap_values_class_1, X_filtered, selected_features_final)


def run_model_union(ms_info, model_name, ms_file_name, feature_reduce_choice, normalize_select, log10_select):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']

    print(f'X shape {X.shape}')
    current_working_dir = os.getcwd()
    output_dir = os.path.join(current_working_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    print(f'Starting {model_name}')
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)

    log10_pipe = FunctionTransformer(np.log10) if log10_select else 'passthrough'
    norm_pipe = Normalizer(norm='l1') if normalize_select else 'passthrough'

    pipeline = Pipeline([
        ('normalize', norm_pipe),
        ('transform', log10_pipe),
        ('scaler', StandardScaler()),
        ('Reduction', get_feature_reduction(feature_reduce_choice)),
        ('classifier', RandomForestClassifier())
    ], memory=memory)

    all_y_true = []
    all_y_scores = []
    all_balanced_accuracies = []
    all_recalls = []
    all_f1_scores = []
    all_precisions = []

    selected_features_union = set()
    selected_features_intersection = None

    # List to store rows of features
    feature_selection_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_scores.extend(y_score)
        all_balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        all_recalls.append(recall_score(y_test, y_pred))
        all_f1_scores.append(f1_score(y_test, y_pred))
        all_precisions.append(precision_score(y_test, y_pred))

        # Get the features used after reduction
        reduction_step = pipeline.named_steps['Reduction']
        if hasattr(reduction_step, 'support_'):
            support_mask = reduction_step.support_
            selected_features = [features[i] for i, flag in enumerate(support_mask) if flag]
        else:
            selected_features = features  # No reduction step, use all features

        selected_features_union.update(selected_features)  # Track union of all selected features

        if selected_features_intersection is None:
            selected_features_intersection = set(selected_features)
        else:
            selected_features_intersection.intersection_update(selected_features)

        # Add the selected features as a new row
        feature_selection_rows.append([str(fold_idx)] + selected_features)

    # Convert the sets to sorted lists before saving
    selected_features_union_list = sorted(list(selected_features_union))
    selected_features_intersection_list = sorted(list(selected_features_intersection))

    # Save the union of selected features
    selected_union_features_file = os.path.join(output_dir, 'selected_features_union.csv')
    pd.Series(selected_features_union_list).to_csv(selected_union_features_file, index=False)
    print(f"Union of selected features saved to '{selected_union_features_file}'.")

    # Save the intersection of selected features
    selected_intersection_features_file = os.path.join(output_dir, 'selected_features_intersection.csv')
    pd.Series(selected_features_intersection_list).to_csv(selected_intersection_features_file, index=False)
    print(f"Intersection of selected features saved to '{selected_intersection_features_file}'.")

    # Determine the maximum number of columns needed
    max_features = max(len(row) for row in feature_selection_rows)

    # Create a DataFrame with dynamic columns
    column_names = ['Fold'] + [f'Feature_{i + 1}' for i in range(max_features - 1)]
    feature_selection_df = pd.DataFrame(feature_selection_rows, columns=column_names)

    # Save the feature selection DataFrame to a CSV file
    feature_selection_file = os.path.join(output_dir, 'feature_selection_per_fold.csv')
    feature_selection_df.to_csv(feature_selection_file, index=False)
    print(f"Feature selection per fold saved to '{feature_selection_file}'.")

    # Use the intersection of selected features for SHAP
    selected_features_final = selected_features_union_list
    X_filtered = X[selected_features_final]

    # Use TreeExplainer for tree-based models like RandomForest
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values_class_1 = explainer.shap_values(X_filtered)

    # Calculate mean and standard deviation for metrics
    mean_balanced_accuracy = np.mean(all_balanced_accuracies)
    std_balanced_accuracy = np.std(all_balanced_accuracies)

    mean_recall = np.mean(all_recalls)
    std_recall = np.std(all_recalls)

    mean_f1 = np.mean(all_f1_scores)
    std_f1 = np.std(all_f1_scores)

    mean_precision = np.mean(all_precisions)
    std_precision = np.std(all_precisions)

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_scores)
    roc_auc = auc(fpr, tpr)

    memory.clear(warn=False)

    return (mean_balanced_accuracy, std_balanced_accuracy, mean_recall, std_recall, mean_f1, std_f1, mean_precision,
            std_precision), (roc_auc, fpr, tpr, thresholds), (shap_values_class_1, X_filtered, selected_features_final)


def get_results(model_name, ms_input_file, feature_reduce_choice, transpose_select, norm, log10):
    print(f"Starting ... {model_name} / {ms_input_file}")
    ms_file_name = Path(ms_input_file).stem
    df_file = load_data_from_file(ms_input_file, transpose_select)
    ms_info = load_data_frame(df_file)
    the_model = get_model(model_name, ms_info['y'])
    # <------------------------------------------
    results = run_model_intersection(ms_info, the_model, ms_file_name, feature_reduce_choice, norm, log10)
    # results = run_model_union(ms_info, the_model, ms_file_name, feature_reduce_choice, norm, log10)
    # <------------------------------------------
    return results, ms_info


seed = 1234546


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        elif v.lower() == 'none':  # Handle 'none' explicitly
            return None
        else:
            raise argparse.ArgumentTypeError('Boolean value expected or None.')
    raise argparse.ArgumentTypeError('Boolean value expected.')


def str_or_none(v):
    if v.lower() == 'none':
        return None
    return v


def save_input_params(params, output_dir):
    """Save input parameters to a JSON file."""
    params_file = os.path.join(output_dir, 'input_parameters.json')
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Input parameters saved to '{params_file}'.")


def main(ms_input_file, feature_reduce_choice, transpose, norm, log10):
    #try:
    print("Starting ... ")
    if feature_reduce_choice is None:
        print("No feature reduction method selected. Proceeding without feature reduction.")
    else:
        print(f"Feature reduction method selected: {feature_reduce_choice}")

    # Prepare the output directory
    output_dir = os.path.join(current_working_dir, 'output')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save input parameters to a file
    input_params = {
        "ms_input_file": ms_input_file,
        "feature_reduce_choice": feature_reduce_choice,
        "transpose": transpose,
        "norm": norm,
        "log10": log10,
    }
    save_input_params(input_params, output_dir)

    transpose_select = transpose
    feature_reduce_choice = feature_reduce_choice  # None #'Boruta' #'Boruta' #'Boruta'
    model_name = 'RandomForest'  # 'TensorFlow' #'RandomForest'#'RandomForest' #'GradientBoosting' #'SVM' #'ElasticNet'
    results, ms_info = get_results(model_name, ms_input_file, feature_reduce_choice, transpose_select, norm, log10)
    ms_file_name = Path(ms_input_file).stem
    gen_output(results, model_name, ms_file_name)
    plot_roc(results, model_name)
    features = ms_info['feature_names']
    plot_shap_values(results, model_name, ms_file_name)

    if not os.path.exists(os.path.join(current_working_dir, 'zipFiles')):
        os.makedirs(os.path.join(current_working_dir, 'zipFiles'))

    create_zip_file_output(os.path.join(current_working_dir, f'zipFiles/{model_name}_{ms_file_name}'), output_dir)


#except Exception as e:
#    print(f"An error occurred: {e}")
# Optionally log the error to a file or take other actions as needed

# Command to run
# Python ../../../OilClassPlots.py Adult_SOY-MALDI_TAG_unnorm_30Aug2024.csv none false true false

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run regression models with feature reduction.')
    parser.add_argument('ms_input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('feature_reduce_choice', type=str_or_none, nargs='?', default=None,
                        help='Choice of feature reduction method. Defaults to None.')
    parser.add_argument('transpose', type=str2bool, help='Transpose file (true/false)')
    parser.add_argument('norm', type=str2bool, help='Normalize (true/false)')
    parser.add_argument('log10', type=str2bool, help='Take the log 10 of input in the pipeline (true/false)')
    args = parser.parse_args()

    main(args.ms_input_file, args.feature_reduce_choice, args.transpose, args.norm, args.log10)  # , args.set_seed)
