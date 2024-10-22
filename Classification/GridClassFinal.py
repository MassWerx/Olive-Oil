import json

import pandas as pd
import numpy as np
import joblib
import sklearn
import sys
import tensorflow as tf
import shap
import pickle

from scikeras.wrappers import KerasClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, Normalizer
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_val_score, StratifiedKFold, \
    cross_validate

from sklearn.pipeline import Pipeline
import os
from os.path import basename
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZipFile
import argparse
import random
import shutil
import plotly.express as px

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, recall_score, f1_score, precision_score, \
    accuracy_score
from sklearn.svm import SVC

import csv
from collections import defaultdict
import importlib.metadata

from joblib import Memory

import multiprocessing
import subprocess

import warnings

warnings.simplefilter(action='ignore')  #, category=FutureWarning)
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


def create_zip_file_output(output_file_name, base_dir):
    """
    Function: create_zip_file_output
    ---------------------------------
    Creates a zip file containing all files from a specified directory.

    This function compresses all files within the given base directory (including files in subdirectories)
    into a zip file. The resulting zip file will be saved with the provided output file name.

    Parameters:
    -----------
    output_file_name : str
        The name of the output zip file (without the '.zip' extension).

    base_dir : str
        The base directory from which all files will be added to the zip file.

    Returns:
    --------
    None
        The function does not return any value, but a zip file is created in the current directory.
    """

    with ZipFile(f'{output_file_name}.zip', 'w') as zipObj:
        for folderName, subfolders, filenames in os.walk(base_dir):
            for filename in filenames:
                file_path = os.path.join(folderName, filename)
                zipObj.write(file_path, basename(file_path))


def get_feature_reduction(feature_reduce_choice):
    """
    Function: get_feature_reduction
    --------------------------------
    Returns a feature reduction method based on the user's choice.

    This function returns a specific feature reduction function or technique based on the
    provided `feature_reduce_choice`. Currently, it supports 'Boruta' for feature selection
    or skips feature reduction if no valid choice is provided.

    Parameters:
    -----------
    feature_reduce_choice : str or None
        The chosen feature reduction method. Accepted values:
        - None: No feature reduction.
        - 'Boruta': Applies the Boruta feature selection method using a RandomForestClassifier.

    Returns:
    --------
    reduction_fun : callable or None
        The feature reduction function corresponding to the selected method, or None if no reduction
        method is chosen.
    """

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

def load_data_from_file(input_file, flip):
    """
    Load data from a CSV file into a pandas DataFrame, with an option to transpose the data.

    This function reads data from a CSV file and stores it in a pandas DataFrame. If the `flip` parameter is set to
    True, the data is transposed (i.e., rows become columns and columns become rows), the first transposed row is
    used as the new header, and this row is then removed from the DataFrame. If `flip` is False, the file is read
    with the first row as the header.

    Parameters:
    -----------
    input_file : str
        The file path of the CSV file to load.

    flip : bool
        If True, the DataFrame will be transposed. The first row will be set as the header and then removed.
        If False, the CSV will be read with the first row as the header.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing the loaded data, with or without transposition based on the `flip` parameter.
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

    This function extracts sample names, categorical labels, and feature data from an input pandas DataFrame.
    It also converts the categorical labels into one-hot encoded format and computes numeric target labels.
    The processed data, including the class names and feature names, is returned as a dictionary for further use
    in machine learning tasks.

    Parameters:
    -----------
    input_dataframe : pd.DataFrame
        A pandas DataFrame containing the dataset. The first column contains the sample names, the second column
        contains the labels, and the remaining columns contain the feature data.

    Returns:
    --------
    dict
        A dictionary (ms_info) containing:
        - 'class_names': List of unique class names extracted from the labels.
        - 'labels': The original labels as extracted from the input DataFrame.
        - 'X': The features DataFrame (extracted from the input DataFrame).
        - 'y': Numeric target labels corresponding to the one-hot encoded classes.
        - 'samples': The sample names.
        - 'features': List of feature names.
        - 'feature_names': List of feature names (identical to 'features').
    """
    # Extract samples, labels, and features
    samples = input_dataframe.iloc[:, 0]
    labels = input_dataframe.iloc[:, 1]
    data_table = input_dataframe.iloc[:, 2:]
    features = data_table.columns.values.tolist()

    # One-hot encode the labels
    label_table = pd.get_dummies(labels)
    class_names = label_table.columns.to_numpy()

    # Convert labels to numeric form using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    oil_y_num = encoder.fit_transform(labels.to_numpy().reshape(-1, 1))
    y = np.argmax(oil_y_num, axis=1)

    # Construct the final dictionary with all necessary information
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

    # Set random seeds for reproducibility




def build_tf_model(n_features, n_classes, seed, n_neurons):
    """Create and return a TensorFlow model for classification.

    This function sets various random seeds to ensure reproducibility across
    multiple stages (Python, NumPy, TensorFlow). It builds a neural network
    using Keras with specific configurations, including dropout, regularization,
    and learning rate schedules.

    Args:
        n_features (int): Number of input features in the dataset.
        n_classes (int): Number of output classes for classification.
        seed (int): Seed value for controlling randomization across libraries.

    Returns:
        tf.keras.Model: Compiled TensorFlow model ready for training.
    """
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
    TF_NUM_OF_NEURONS = n_neurons # 40  # current from report 32

    tf_model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(TF_NUM_OF_NEURONS, activation=tf.nn.relu),  # input shape required train_x
        tf.keras.layers.Dropout(TF_DROPOUT_RATE, noise_shape=None, seed=drop_out_seed),
        tf.keras.layers.Dense(TF_NUM_OF_NEURONS, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                              activation=tf.nn.relu),
        tf.keras.layers.Dense(n_classes, activation='softmax')])

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(TF_LEARN_RATE,
                                                                 decay_steps=2,  # 1, #2, # train_x.size * 1000,
                                                                 decay_rate=1,  # .5, #1,
                                                                 staircase=False)

    tf_optimizer = tf.keras.optimizers.Adam()  # lr_schedule)

    tf_model.compile(optimizer=tf_optimizer,
                     # loss='categorical_crossentropy',
                     loss='sparse_categorical_crossentropy',
                     metrics=['acc'])

    return tf_model

# class MyKerasClf:
"""
Custom Keras Classifier wrapper to be compatible with scikit-learn's API.

Args:
    n_classes (int): The number of output classes for classification.
    seed (int): Seed value for controlling randomization in the model.

Attributes:
    n_classes (int): The number of output classes.
    seed (int): Seed value for controlling randomization.
    clf (KerasClassifier): The Keras classifier model.

Methods:
    predict(X):
        Predict the class labels for the given input data.
    create_model(learn_rate=0.01, weight_constraint=0):
        Creates the Keras model with the specified parameters.
    fit(X, y, **kwargs):
        Fit the model to the given input data and labels.
    predict_proba(X):
        Predict class probabilities for the given input data.
    get_params(deep=True):
        Get parameters for this estimator.
    set_params(**params):
        Set the parameters of this estimator.
"""
class MyKerasClf(BaseEstimator, ClassifierMixin):
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

    def __init__(self, n_classes, seed, n_neurons=40) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.seed = seed
        self.n_neurons = n_neurons

    def predict(self, X):
        y_pred_nn = self.clf.predict(X)
        return np.array(y_pred_nn).flatten()

    def create_model(self, n_neurons):
        model = build_tf_model(self.input_shape, self.n_classes, self.seed, n_neurons)
        return model

    def fit(self, X, y, **kwargs):
        self.input_shape = X.shape[1]
        self.classes_ = np.unique(y)
        n_neurons = kwargs.get('n_neurons', self.n_neurons)
        # print(f"Fitting with parameters: n_neurons={n_neurons}")
        self.clf = KerasClassifier(model=self.create_model(n_neurons=n_neurons),  verbose=0, epochs=220, batch_size=100)

        self.clf.fit(X, y, **kwargs)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def get_params(self, deep=True):
        return {"n_classes": self.n_classes, "seed": self.seed}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self



def get_models(y):
    """
    Generate a list of machine learning models and their hyperparameter grids for model selection and tuning.

    This function returns a list of tuples, where each tuple consists of a machine learning model's name, the model
    itself (as a scikit-learn estimator), and a dictionary of hyperparameter grids. The models include a variety of
    classifiers such as Support Vector Machine (SVM), AdaBoost, Logistic Regression, K-Nearest Neighbors, Gradient
    Boosting, Random Forest, and a custom Keras-based Artificial Neural Network (ANN). The number of classes in the
    target labels `y` is used to determine the output layer size for the ANN.

    Parameters:
    -----------
    y : array-like
        Target labels from the dataset, used to determine the number of unique classes for the ANN model.

    Returns:
    --------
    list_of_models : list
        A list of tuples, where each tuple contains:
        - A string representing the name of the model.
        - The model as a classifier object.
        - A dictionary of hyperparameter grids for the corresponding model.
    """
    list_of_models = [
        # Support Vector Machine (SVM)
        ('SVM', SVC(probability=True, random_state=seed), {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }),

        # AdaBoost Classifier
        ('AdaBoost', AdaBoostClassifier(random_state=seed), {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 1]
        }),

        # Logistic Regression
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=seed), {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs']
        }),

        # K-Nearest Neighbors (KNN)
        ('KNeighbors', KNeighborsClassifier(), {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }),

        # Gradient Boosting Classifier
        ('GradientBoosting', GradientBoostingClassifier(random_state=seed), {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }),

        # Random Forest Classifier
        ('RandomForest', RandomForestClassifier(n_jobs=-1, random_state=seed), {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }),

        # Custom Keras Classifier (ANN) please use the full search if you have time/resources
        ('ANN', MyKerasClf(n_classes=len(np.unique(y)), seed=seed), {
            'classifier__n_neurons': [30, 40, 50],  # Number of neurons in the dense layers
        })

    ]

    """"
    This can take way to long to run 
    ('ANN', MyKerasClf(n_classes=len(np.unique(y)), seed=seed), {
        'classifier__learn_rate': [0.001, 0.01, 0.1],  # Learning rates to try
        'classifier__weight_constraint': [0, 1, 3, 5],  # Weight constraints
        'classifier__n_neurons': [20, 40, 60],  # Number of neurons in the dense layers
        'classifier__dropout_rate': [0.1, 0.2, 0.3, 0.5],  # Dropout rate for regularization
        'classifier__epochs': [100, 150, 200, 250]  # Number of epochs to train the model
    })"""

    # Example shortened list for testing purposes
    """test_list_of_models_short = [
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=seed), {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs']
        }),
    ]"""


    return list_of_models

def plot_roc(ms_info, pipeline, model_name, output_dir, n_splits=5, n_repeats=10):
    """
    Function: plot_roc
    -------------------
    Generates and saves the ROC curve and AUC (Area Under Curve) for a given model pipeline using repeated
    stratified k-fold cross-validation. The function collects ROC curve points, calculates the AUC, and saves
    the ROC data and plot in both CSV and image formats.

    Parameters:
    -----------
    ms_info : dict
        Dictionary containing the feature matrix 'X', target labels 'y', and 'feature_names'.

    pipeline : Pipeline
        A scikit-learn pipeline containing the model and preprocessing steps.

    model_name : str
        Name of the model, used to save output files.

    output_dir : str
        Directory where the ROC plot and data will be saved.

    n_splits : int, optional (default=5)
        Number of folds for the k-fold cross-validation.

    n_repeats : int, optional (default=10)
        Number of times the k-fold cross-validation is repeated.

    Returns:
    --------
    roc_auc : float
        The area under the ROC curve (AUC), rounded to 3 decimal places.
    """

    # Extract feature matrix, target labels, and feature names from ms_info
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']

    # Initialize cross-validation scheme
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    # Initialize lists to collect true labels and predicted probabilities
    all_y_true = []
    all_y_scores = []

    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        # print(f'Processing fold {fold_idx}')

        # Split data into training and testing sets
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train the pipeline and predict probabilities
        pipeline.fit(X_train, y_train)
        # By default, the ROC curve is often computed for class 1 in binary classification (positive class),
        # treating class 0 as the negative class.
        y_score = pipeline.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
        # Append true labels and scores
        all_y_true.extend(y_test)
        all_y_scores.extend(y_score)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_scores)
    roc_auc = round(auc(fpr, tpr), 3)

    # Save ROC data to a CSV file
    roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
    roc_data.to_csv(f'{output_dir}/roc_data_{model_name}.csv', index=False)

    # Generate ROC curve plot using Plotly
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={roc_auc:.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=2100, height=1500  # Adjust size to 7x5 inches at 300 DPI
    )

    # Add diagonal reference line
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    # Customize plot axes
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    # Customize title and axis font sizes
    fig.update_layout(
        title=dict(font=dict(size=48)),
        xaxis=dict(title=dict(font=dict(size=48)), tickfont=dict(size=40)),
        yaxis=dict(title=dict(font=dict(size=48)), tickfont=dict(size=40)),
        margin=dict(l=80, r=40, t=80, b=40)
    )

    # Save the ROC plot as an image file
    fig.write_image(f'{output_dir}/{model_name}_ROC.png')

    return roc_auc


"""
Function: plot_SHAP
-------------------
This function generates SHAP (SHapley Additive exPlanations) values and visualizations for a given model pipeline. 
It applies normalization, scaling, and feature reduction (if applicable), followed by SHAP value calculation using 
appropriate explainers for different model types. It saves SHAP values, beeswarm plots, and summary plots for 
visualization, as well as selected features to a file.

Parameters:
-----------
X : pd.DataFrame
    Feature data to be used for SHAP analysis.

y : np.ndarray
    Target labels for fitting the model.

features : list
    List of feature names.

model : object
    The model object (e.g., RandomForestClassifier, LogisticRegression).

pipeline : Pipeline
    A scikit-learn pipeline that contains various preprocessing steps and the classifier.

output_dir : str
    Directory where the SHAP plots and data will be saved.

name : str
    Unique name or identifier for the output files.

Returns:
--------
None
"""


def plot_SHAP(X, y, features, model, pipeline, output_dir, name):
    """
    Generate SHAP values and visualizations for a given model pipeline.

    This function calculates SHAP (SHapley Additive exPlanations) values for the model in the pipeline, using the
    appropriate SHAP explainer based on the model type (e.g., TreeExplainer for tree-based models, KernelExplainer
    for SVC). It also applies normalization, scaling, and feature reduction, saves the selected features, SHAP
    values, and generates SHAP beeswarm and summary plots.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature data to be used for SHAP analysis.

    y : np.ndarray
        Target labels for fitting the model.

    features : list
        List of feature names.

    model : object
        The model object (e.g., RandomForestClassifier, LogisticRegression).

    pipeline : Pipeline
        A scikit-learn pipeline containing preprocessing steps and the classifier.

    output_dir : str
        Directory where the SHAP plots, selected features, and SHAP values will be saved.

    name : str
        Unique identifier for the output files.

    Returns:
    --------
    None
        Saves the following outputs to `output_dir`:
        - A text file of selected features.
        - A pickle file of SHAP values and filtered features.
        - Beeswarm and summary plots in PNG format.
        - A combined image of the beeswarm and summary plots.
    """
    # Fit the pipeline to extract SHAP values
    pipeline.fit(X, y)

    # Extract normalization and scaling steps from the pipeline
    normalizer = pipeline.named_steps['normalize']
    X_nor = normalizer.transform(X)

    scaler = pipeline.named_steps['scaler']
    X_scale = scaler.transform(X_nor)

    # Apply feature reduction (if any)
    reduction_step = pipeline.named_steps['Reduction']
    if hasattr(reduction_step, 'support_'):
        print("Applying feature reduction...")
        X_reduced = reduction_step.transform(X_scale)
        support_mask = reduction_step.support_
        selected_features = [features[i] for i, flag in enumerate(support_mask) if flag]
    else:
        X_reduced = X_scale
        selected_features = features

    print(f'{len(y)} samples & {len(selected_features)} selected features: {selected_features}')

    # Save the selected features to a file
    with open(f'{output_dir}/selected_feature_{name}.txt', 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    # Determine the appropriate SHAP explainer based on the model type
    if isinstance(model, MyKerasClf):
        print("Using GradientExplainer for ANN")
        explainer = shap.GradientExplainer(pipeline.named_steps['classifier'].clf.model, X_reduced)
        shap_values = explainer.shap_values(X_reduced)
        shap_values = shap_values[:, :, 0]  # Class 0 (adjust as needed)

    elif isinstance(pipeline.named_steps['classifier'], LogisticRegression):
        print("Using LinearExplainer for LogisticRegression")
        explainer = shap.LinearExplainer(pipeline.named_steps['classifier'], X_reduced, model_output='probability')
        shap_values = explainer.shap_values(X_reduced)

    elif isinstance(pipeline.named_steps['classifier'], (SVC, KNeighborsClassifier, AdaBoostClassifier)):
        print("Using KernelExplainer for SVC, KNN, and AdaBoost")
        explainer = shap.KernelExplainer(pipeline.named_steps['classifier'].predict_proba, X_reduced)
        shap_values = explainer.shap_values(X_reduced)
        shap_values = shap_values[:, :, 0]  # Class 0 (adjust as needed)
        shap_values = np.array(shap_values)

    elif isinstance(pipeline.named_steps['classifier'], (RandomForestClassifier, GradientBoostingClassifier)):
        print("Using TreeExplainer for RandomForest, GradientBoosting, or AdaBoost")
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_reduced)
        if isinstance(pipeline.named_steps['classifier'], RandomForestClassifier):
            shap_values = shap_values[:, :, 0]  # Class 0 (adjust as needed)

    else:
        explainer = shap.Explainer(pipeline.named_steps['classifier'], X_reduced)
        shap_values = explainer(X_reduced)

    # Format SHAP values for visualization
    shap_values = shap.Explanation(
        values=shap_values,
        data=X_reduced,
        feature_names=selected_features
    )

    # Save SHAP values and data for future use
    shap_data = {
        'shap_values': shap_values,
        'features': selected_features,
        'X_filtered': X_reduced
    }
    with open(f'{output_dir}/shap_data_{name}.pkl', 'wb') as f:
        pickle.dump(shap_data, f)

    # Generate and save beeswarm and summary plots
    shap.summary_plot(shap_values, plot_type="dot", feature_names=selected_features, max_display=10, show=False)
    plt.savefig(f'{output_dir}/shap_beeswarm_{name}_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, plot_type="bar", feature_names=selected_features, max_display=10, show=False)
    plt.savefig(f'{output_dir}/shap_summary_plot_{name}_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Combine beeswarm and summary plots into a single image
    bar_plot = Image.open(f'{output_dir}/shap_beeswarm_{name}_plot.png')
    summary_plot = Image.open(f'{output_dir}/shap_summary_plot_{name}_all_folds.png')

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(bar_plot)
    axes[0].axis('off')
    axes[1].imshow(summary_plot)
    axes[1].axis('off')

    fig.suptitle('SHAP Bar Plot and Summary Plot', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_combined_{name}_plots.png', dpi=300)
    plt.close()
    print(f"Combined SHAP plot saved as 'shap_combined_{name}_plots.png'")


def save_best_model(model, output_path):
    """
    Save the trained model to a file using joblib.

    Parameters:
    -----------
    model : scikit-learn compatible model
        The trained model to save.

    output_path : str
        The directory path and filename to save the model.

    Returns:
    --------
    None
    """
    joblib.dump(model, output_path)
    print(f"Model saved successfully to {output_path}")


"""
Function: run_models
--------------------
This function takes in dataset information, a list of machine learning models, and various pipeline configurations 
to perform the following tasks:
1. Create a machine learning pipeline with optional feature reduction, normalization, and log transformation steps.
2. Perform Grid Search Cross Validation to find the best hyperparameters for each model.
3. Save the best model parameters and all model parameters to files.
4. Perform Repeated Stratified K-Fold Cross Validation, collecting and averaging performance metrics (accuracy, recall, etc.).
5. Save the cross-validation scores and generate SHAP plots and ROC curves.
6. Zip the output files for easier access.
"""
def run_models(ms_info, list_of_models, ms_file_name, feature_reduce_choice, normalize_select, log10_select,
               n_splits=5, n_repeats=10, round_scores=True):
    """
    Run various machine learning models with optional feature reduction, normalization, and log transformation.

    This function creates a machine learning pipeline and performs grid search cross-validation to find the best
    hyperparameters for each model. It evaluates the models using repeated stratified K-fold cross-validation and
    saves the results, including SHAP plots and ROC curves.

    Parameters:
    -----------
    ms_info : dict
        A dictionary containing dataset information with keys:
        - 'X': Features dataset (DataFrame).
        - 'y': Target labels (array-like).
        - 'feature_names': List of feature names (list).

    list_of_models : list of tuples
        A list where each element is a tuple (model_name, model_object, param_grid):
        - model_name: A string representing the name of the model.
        - model_object: A scikit-learn compatible model instance.
        - param_grid: A dictionary of hyperparameters for GridSearchCV.

    ms_file_name : str
        The name of the dataset file for reference in output files.

    feature_reduce_choice : str
        The feature reduction method to be used in the pipeline.

    normalize_select : bool
        Whether to apply normalization in the pipeline.

    log10_select : bool
        Whether to apply log10 transformation in the pipeline.

    n_splits : int, optional (default=5)
        Number of folds in Stratified K-Fold Cross Validation.

    n_repeats : int, optional (default=10)
        Number of repetitions for Repeated Stratified K-Fold Cross Validation.

    round_scores : bool, optional (default=True)
        Whether to round performance metrics to 3 decimal places.

    Returns:
    --------
    None
        Outputs various files including best model parameters, cross-validation metrics, SHAP plots, and ROC curves.
    """
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
    current_working_dir = os.getcwd()
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    # Define transformation steps based on flags
    log10_pipe = FunctionTransformer(np.log10) if log10_select else 'passthrough'
    norm_pipe = Normalizer(norm='l1') if normalize_select else 'passthrough'

    for name, model, param_grid in list_of_models:
        print(f'Starting {name}')
        # Create output directory for the current model
        dirpath = Path(os.path.join(current_working_dir, f'output_{name}'))
        dirpath.mkdir(parents=True, exist_ok=True)

        # Build the pipeline with optional transformations and feature reduction
        pipeline = Pipeline([
            ('normalize', norm_pipe),
            ('transform', log10_pipe),
            ('scaler', StandardScaler()),
            ('Reduction', get_feature_reduction(feature_reduce_choice)),
            ('classifier', model)
        ], memory=memory)

        # Perform grid search on the entire dataset
        grid_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=grid_cv, scoring='neg_log_loss')
        grid_search.fit(X, y)
        best_params = grid_search.best_params_

        # Update the pipeline with the best parameters
        pipeline.set_params(**best_params)
        best_pipeline = grid_search.best_estimator_

        # Save the trained model after grid search is done
        model_output_path = os.path.join(dirpath, f"{name}_best_model.pkl")
        print(f'{model_output_path}')
        save_best_model(best_pipeline, model_output_path)

        # Save the best parameters to a JSON file
        with open(f'{dirpath}/best_params_{name}.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        # Extract and save only the best parameters specific to the model
        model_best_params = {k.replace('classifier__', ''): v for k, v in best_params.items() if
                             k.startswith('classifier__')}
        model.set_params(**model_best_params)

        # Save all model parameters to a text file
        with open(f'{dirpath}/all_params_{name}.txt', 'w') as f:
            for key, value in model.get_params().items():
                f.write(f"{key}: {value}\n")

        # Perform Repeated Stratified K-Fold Cross Validation
        outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        all_scores = {'accuracy': [], 'balanced_accuracy': [], 'recall': [], 'f1': [], 'precision': []}

        # Evaluate model performance
        scores = cross_validate(pipeline, X, y, cv=outer_cv,
                                scoring=['accuracy', 'balanced_accuracy', 'recall', 'f1', 'precision'])

        # Calculate average scores per repeat
        repeat_averages = {metric: [] for metric in all_scores.keys()}

        for repeat in range(n_repeats):
            start_idx = repeat * n_splits
            end_idx = start_idx + n_splits

            for metric in all_scores.keys():
                fold_scores = scores[f'test_{metric}'][start_idx:end_idx]
                repeat_avg = np.mean(fold_scores)
                repeat_averages[metric].append(repeat_avg)
                all_scores[metric].extend(fold_scores)

        # Compute mean and standard deviation for each metric
        metrics_summary = {metric: (np.mean(repeat_averages[metric]), np.std(repeat_averages[metric])) for metric in
                           all_scores.keys()}
        if round_scores:
            metrics_summary = {metric: (round(mean, 3), round(std, 3)) for metric, (mean, std) in
                               metrics_summary.items()}

        # Save cross-validation scores for debugging
        scores_df = pd.DataFrame(all_scores)
        if round_scores:
            scores_df = scores_df.round(3)
        scores_df.to_csv(f'{dirpath}/cv_{name}.csv', index=False)

        # Generate SHAP values and ROC curve
        plot_SHAP(X, y, features, model, pipeline, dirpath, name)
        roc_auc = plot_roc(ms_info, pipeline, name, dirpath)

        # Save overall metrics
        with open(f'{dirpath}/metrics_cv_{name}.txt', 'w') as f:
            for metric, (mean, std) in metrics_summary.items():
                f.write(f'{metric.capitalize()}.avg: {mean}\n')
                f.write(f'{metric.capitalize()}.sd: {std}\n')
            f.write(f'AUC: {roc_auc}\n')

        # Zip output files for easier access
        zip_output_dir = Path(current_working_dir) / 'zipFiles'
        zip_output_dir.mkdir(exist_ok=True)
        create_zip_file_output(zip_output_dir / f'{name}_{ms_file_name}', dirpath)

    # Clean up the temporary cache directory
    shutil.rmtree(cachedir)


def resetDirs(list_of_models):
    for name, model, param_grid in list_of_models:
        dirpath = Path(os.path.join(current_working_dir, f'output_{name}'))
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        os.makedirs(dirpath)


seed = 123456


def str2bool(v):
    """Convert a string to a boolean value.

    Args:
        v: The input value to convert.

    Returns:
        bool or None: The converted boolean value or None if the input is None.

    Raises:
        argparse.ArgumentTypeError: If the input is not a valid boolean string.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        elif v.lower() == 'none':
            return None
        else:
            raise argparse.ArgumentTypeError('Boolean value expected or None.')
    raise argparse.ArgumentTypeError('Boolean value expected.')


def str_or_none(v):
    """Convert a string to None if it equals 'none'.

    Args:
        v: The input string.

    Returns:
        str or None: The input string or None if it was 'none'.
    """
    if v.lower() == 'none':
        return None
    return v


def save_input_params(params, output_dir):
    """Save input parameters to a JSON file.

    Args:
        params (dict): The input parameters to save.
        output_dir (str): The directory to save the parameters in.
    """
    params_file = os.path.join(output_dir, 'input_parameters.json')
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Input parameters saved to '{params_file}'.")


def main(ms_input_file, feature_reduce_choice, transpose, norm, log10):
    """Main function to process input files and run models.

    Args:
        ms_input_file (str): Path to the input file.
        feature_reduce_choice: The feature reduction method chosen.
        transpose (bool): Whether to transpose the data.
        norm (bool): Whether to normalize the data.
        log10 (bool): Whether to apply log10 transformation.
    """
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

    ms_file_name = Path(ms_input_file).stem
    df_file = load_data_from_file(ms_input_file, transpose)
    ms_info = load_data_frame(df_file)
    list_of_models = get_models(ms_info['y'])
    resetDirs(list_of_models)

    print(f"------> Starting CV avg SD {ms_input_file} / {feature_reduce_choice}... with {seed}")
    run_models(ms_info, list_of_models, ms_file_name, feature_reduce_choice, norm, log10)


"""
Grid search
Adulteration Can                                               feature reduction , transpose , norm , log10
python ../../GridClassFinal.py Adult_CAN-MALDI_TAG_unnorm_8Sep2024.csv Boruta false true false
Adulteration Soy
python ../../GridClassFinal.py Adult_SOY-MALDI_TAG_unnorm_8Sep2024.csv Boruta false true false
Fresh
python ../../GridClassFinal.py  Freshness_PP_filt_unnorm_9Sep2024.csv Boruta false true false
Grade
python ../../GridClassFinal.py  Grade_PP_filt_unnorm_9Sep2024.csv Boruta false true false

"""

if __name__ == "__main__":
    """
    Main entry point for running regression models with feature reduction.

    This script parses command-line arguments that allow users to specify the input file path, 
    choose feature reduction methods, and set various processing options like transposition, 
    normalization, and logarithmic transformation of the data. Once the arguments are parsed, 
    it calls the `main` function to execute the regression modeling workflow.

    Command-line arguments:
    ------------------------
    ms_input_file : str
        Path to the input CSV file containing the dataset.

    feature_reduce_choice : str, optional
        Choice of feature reduction method. If not provided, feature reduction will be skipped.

    transpose : bool
        Whether or not to transpose the input file (True for transposing, False otherwise).

    norm : bool
        Whether to apply normalization to the data (True for normalization, False otherwise).

    log10 : bool
        Whether to apply a log10 transformation to the input data (True for applying the log10, False otherwise).

    Example usage:
    --------------
        python script_name.py input.csv pca true true false
    """

    parser = argparse.ArgumentParser(description='Run regression models with feature reduction.')
    parser.add_argument('ms_input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('feature_reduce_choice', type=str_or_none, nargs='?', default=None,
                        help='Choice of feature reduction method. Defaults to None.')
    parser.add_argument('transpose', type=str2bool, help='Transpose file (true/false)')
    parser.add_argument('norm', type=str2bool, help='Normalize (true/false)')
    parser.add_argument('log10', type=str2bool, help='Take the log 10 of input in the pipeline (true/false)')
    # Uncomment to allow seed setting
    # parser.add_argument('set_seed', type=str, help='The Seed to use')

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.ms_input_file, args.feature_reduce_choice, args.transpose, args.norm, args.log10)  # , args.set_seed


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Citations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the citation dictionary
citation_dict = {
    "json": "Built-in Python module, does not require a citation.",
    "pandas": "Reback, J., et al. (2023). pandas: powerful Python data analysis toolkit. URL: https://pandas.pydata.org/",
    "numpy": "Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357-362.",
    "joblib": "Joblib: running Python functions as pipeline jobs. Joblib documentation. URL: https://joblib.readthedocs.io/en/latest/",
    "scikit-learn": "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "tensorflow": "TensorFlow Development Team. (2016). TensorFlow: Large-scale machine learning on heterogeneous systems. URL: https://tensorflow.org",
    "shap": "Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems, 30.",
    "pickle": "Built-in Python module, does not require a citation.",
    "scikeras": "SciKeras Development Team. (2021). SciKeras: A Scikit-learn API wrapper for Keras. URL: https://scikeras.readthedocs.io/",
    "boruta": "Kursa, M. B., Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. Journal of Statistical Software, 36(11), 1-13.",
    "plotly": "Plotly Technologies Inc. (2015). Collaborative data science. Montreal, QC: Plotly Technologies Inc. URL: https://plotly.com",
    "matplotlib": "Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95.",
    "PIL": "Clark, A. (2015). Pillow (PIL Fork) Documentation. readthedocs. URL: https://buildmedia.readthedocs.org/media/pdf/pillow/latest/pillow.pdf",
    "Memory": "Joblib: running Python functions as pipeline jobs. Joblib documentation. URL: https://joblib.readthedocs.io/en/latest/",
    "SVC": "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.",
    "roc_curve": "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "auc": "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "balanced_accuracy_score": "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "recall_score": "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "f1_score": "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "precision_score": "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "accuracy_score": "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "warnings": "Built-in Python module, does not require a citation.",
    "argparse": "Built-in Python module, does not require a citation.",
    "os": "Built-in Python module, does not require a citation.",
    "pathlib": "Built-in Python module, does not require a citation.",
    "random": "Built-in Python module, does not require a citation.",
    "shutil": "Built-in Python module, does not require a citation.",
    "zipfile": "Built-in Python module, does not require a citation."
}

# Define the libraries to be processed
libraries = [
    "json", "pandas", "numpy", "joblib", "scikit-learn", "tensorflow", "shap", "pickle",
    "scikeras", "boruta", "plotly", "matplotlib", "PIL", "Memory", "SVC", "roc_curve",
    "auc", "balanced_accuracy_score", "recall_score", "f1_score", "precision_score",
    "accuracy_score", "warnings", "argparse", "os", "pathlib", "random", "shutil", "zipfile"
]


# Function to get the version of a library, with handling for specific functions from scikit-learn
def get_package_version(library):
    try:
        if library in ["roc_curve", "auc", "balanced_accuracy_score", "recall_score", "f1_score", "precision_score",
                       "accuracy_score", "SVC"]:
            return importlib.metadata.version("scikit-learn")  # All are part of scikit-learn
        elif library == "Memory":
            return "Joblib " + importlib.metadata.version("joblib")  # Memory is part of joblib
        elif library == "PIL":
            return "Pillow " + importlib.metadata.version("Pillow")  # PIL is known as Pillow
        else:
            return importlib.metadata.version(library)
    except importlib.metadata.PackageNotFoundError:
        return "Version not found"


# Function to get the Python version
def get_python_version():
    return f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"



# Function to save the citation information to a CSV file
def save_citations_to_csv(filename):
    # Create and write to CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Software", "Package", "Package Version", "Citation"])

        # Write rows for each package
        for library in libraries:
            version = get_package_version(library)
            citation = citation_dict.get(library, "Citation not found")
            writer.writerow([get_python_version(), library, version, citation])

    print(f"Citations saved to {filename}")


""""# Define the main function
def main():
    # Output CSV filename
    filename = "software_citations.csv"
    save_citations_to_csv(filename)


# Standard Python entry point
if __name__ == "__main__":
    main()"""
