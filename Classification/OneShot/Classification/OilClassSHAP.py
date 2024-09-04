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
from sklearn.linear_model import LogisticRegression
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
    samples = input_dataframe.iloc[:, 0]
    labels = input_dataframe.iloc[:, 1]
    data_table = input_dataframe.iloc[:, 2:]
    features = input_dataframe.iloc[:, 2:].columns.values
    features_names = features#.to_numpy().astype(str)

    label_table = pd.get_dummies(labels)
    class_names = label_table.columns.to_numpy()

    encoder = OneHotEncoder(sparse_output=False)
    oil_y_num = encoder.fit_transform(labels.to_numpy().reshape(-1, 1))
    X = data_table
    y = np.argmax(oil_y_num, axis=1)

    ms_info = {
        'class_names': class_names,
        'labels': labels,
        'X': pd.DataFrame(X, columns=features_names),
        'y': y,
        'samples': samples,
        'features': features,
        'feature_names': features_names
    }
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
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=seed),
    }

    best_params = {
        'RandomForest': {'max_depth': "null", 'min_samples_lear': 1, 'min_samples_split': 2,'n_estimators': 100}, # Adult CAN
        'TensorFlow': {'learn_rate': .001, 'weight_constraint': 0}, # Grade
        'SVM': {'C': 10, 'kernel': 'linear'},  #'SVM': {'C': 1, 'kernel': 'rbf'},, 'probability':'True' Adult SOY
        'GradientBoosting': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}, # NONE
        'LogisticRegression': {'C': 1, 'penalty': "l2", 'solver': "lbfgs"}, # Freshness
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


def run_model(ms_info, model, ms_file_name, feature_reduce_choice, normalize_select, log10_select):
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

    pipeline.fit(X, y)

    reduction_step = pipeline.named_steps['Reduction']
    if hasattr(reduction_step, 'support_'):
        support_mask = reduction_step.support_
        selected_features = [features[i] for i, flag in enumerate(support_mask) if flag]
    else:
        selected_features = features  # No reduction step, use all features

    print(f'{len(y)} & {len(selected_features)} = {selected_features}')
    X_filtered = X[selected_features]

    # Use SHAP LinearExplainer for SVC
    if model == 'RandomForest':
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_filtered)
    else:
        explainer = shap.Explainer(pipeline.named_steps['classifier'], X_filtered)
        shap_values = explainer(X_filtered)

    # Ensure SHAP values shape matches the feature data shape
    assert shap_values.values.shape[1] == X_filtered.shape[1], "Mismatch between SHAP values and feature data"


    if len(shap_values.shape) == 3:
        # Multi-class output (e.g., GBC , RF with TreeExplainer)
        shap_values = shap_values[:, :, 0]  # Select class 0 (or modify as needed)
    elif len(shap_values.shape) == 2:
        # Binary output
        pass  # No adjustment needed

    # Generate beeswarm plot for all features
    shap.summary_plot(shap_values, plot_type="dot",  show=False)
    plt.savefig(f'{output_dir}/shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Beeswarm plot saved as 'shap_beeswarm_plot.png'")


    plt.figure()
    shap.summary_plot(shap_values, plot_type="bar", show=False)
    plt.savefig(f'{output_dir}/shap_summary_plot_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_summary_plot_all_folds.png'")

    memory.clear(warn=False)

    bar_plot = Image.open(f'{output_dir}/shap_beeswarm_plot.png')
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



def get_results(model_name, ms_input_file, feature_reduce_choice, transpose_select, norm, log10):
    print(f"Starting ... {model_name} / {ms_input_file}")
    ms_file_name = Path(ms_input_file).stem
    df_file = load_data_from_file(ms_input_file, transpose_select)
    ms_info = load_data_frame(df_file)
    the_model = get_model(model_name, ms_info['y'])
    # <------------------------------------------
    results = run_model(ms_info, the_model, ms_file_name, feature_reduce_choice, norm, log10)
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
    # LogisticRegression SVM
    model_name = 'LogisticRegression'  # 'TensorFlow' #'RandomForest'#'RandomForest' #'GradientBoosting' #'SVM' #'ElasticNet'
    results, ms_info = get_results(model_name, ms_input_file, feature_reduce_choice, transpose_select, norm, log10)
    ms_file_name = Path(ms_input_file).stem

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
