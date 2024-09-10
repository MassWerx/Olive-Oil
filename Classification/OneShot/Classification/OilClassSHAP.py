# https://github.com/scikit-learn-contrib/boruta_py
import json

import pandas as pd
import numpy as np
import joblib
import sklearn
import sys
import os
import tensorflow as tf
import shap

from scikeras.wrappers import KerasClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, Normalizer
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from pathlib import Path
from tempfile import mkdtemp
import argparse
import random
import shutil
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from joblib import Memory

# Graphics
import plotly.express as px

import matplotlib.pyplot as plt
from PIL import Image

from zipfile import ZipFile
from os.path import basename

import pickle
import importlib.metadata

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

    current_working_dir = os.getcwd()
    output_dir = os.path.join(current_working_dir, 'output')
    samples = input_dataframe.iloc[:, 0]
    labels = input_dataframe.iloc[:, 1]
    data_table = input_dataframe.iloc[:, 2:]
    features = input_dataframe.iloc[:, 2:].columns.values.tolist()
    features_names = features  #.to_numpy().astype(str)

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
        # Adult Can n_estimators 200 / Adult Soy n_estimators 100 / Grade n_estimators 300
        'RandomForest': {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300},
        # Fresh
        'SVM': {'C': 10, 'kernel': 'rbf'},
        # Not used
        'LogisticRegression': {'C': 1, 'penalty': "l2", 'solver': "lbfgs"},
        # Not used
        'TensorFlow': {'learn_rate': .001, 'weight_constraint': 0},
        # Not used
        'GradientBoosting': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100},  # NONE
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


def plot_roc(ms_info, pipeline, model_name, n_splits=5, n_repeats=10):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    current_working_dir = os.getcwd()
    output_dir = os.path.join(current_working_dir, 'output')
    all_y_true = []
    all_y_scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f'Step = {fold_idx}')
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_scores.extend(y_score)

    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_scores)
    roc_auc = round(auc(fpr, tpr), 3)
    # ROC
    # Generate the plot with markers
    roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
    roc_data.to_csv(f'{output_dir}/roc_data_{model_name}.csv', index=False)
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


def run_model(ms_info, model, ms_file_name, feature_reduce_choice, normalize_select, log10_select):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
    class_names = ms_info['class_names']

    # Define a threshold for filtering
    threshold = 1e-5


    print(f'X shape {X.shape}')
    current_working_dir = os.getcwd()
    output_dir = os.path.join(current_working_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    with open(f'{output_dir}/{ms_file_name}_class_names.txt', 'w') as f:
        # Write class names with numerical order
        f.write("Class Names:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{i}: {class_name}\n")

    print(f'Starting {model}')
    # only need to find the best parameters.
    # outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)

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

    # Extract the normalization step
    normalizer = pipeline.named_steps['normalize']

    # Apply normalization to the data
    X_nor = normalizer.transform(X)

    # Now extract the StandardScaler from the pipeline
    scaler = pipeline.named_steps['scaler']
    # Apply the scaler to the normalized data
    X_scale = scaler.transform(X_nor)

    reduction_step = pipeline.named_steps['Reduction']
    if hasattr(reduction_step, 'support_'):
        print("Got reduction")
        # Apply the reduction step
        X_reduced = reduction_step.transform(X_scale)

        # Get the support mask to determine selected features
        support_mask = reduction_step.support_
        selected_features = [features[i] for i, flag in enumerate(support_mask) if flag]
    else:
        # No reduction step, use all normalized features
        X_reduced = X_scale
        selected_features = features

    print(features)
    print(selected_features)

    print(f'{len(y)} & {len(selected_features)} = {selected_features}')
    # Save the selected features to a file
    with open(f'{output_dir}/selected_feature.txt', 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    # X_reduced = pd.DataFrame(X_reduced, columns=selected_features)

    # Use SHAP explainers based on model type
    # Grade
    if isinstance(model, MyKerasClf):
        print("Using KernelExplainer for ANN")
        explainer = shap.GradientExplainer(pipeline.named_steps['classifier'].clf.model, X_reduced)
        shap_values = explainer.shap_values(X_reduced)
        shap_values = shap_values[:, :, 0]  # Select class 0 (or modify as needed)

    # Freshness
    elif isinstance(pipeline.named_steps['classifier'], LogisticRegression):
        print("\nUsing LinearExplainer for LogisticRegression with probabilities\n")
        # For Logistic Regression model using LinearExplainer
        explainer = shap.LinearExplainer(pipeline.named_steps['classifier'], X_reduced, model_output='probability')
        shap_values = explainer.shap_values(X_reduced)

    # Adult Soy
    elif isinstance(pipeline.named_steps['classifier'], SVC):
        print("\nUsing KernelExplainer for SVC\n")
        # For SVM model
        explainer = shap.KernelExplainer(pipeline.named_steps['classifier'].predict_proba, X_reduced)
        shap_values = explainer.shap_values(X_reduced)
        shap_values = shap_values[:, :, 0]  # Select class 0 (or modify as needed)
        shap_values = np.array(shap_values)


    # Adult Can
    elif isinstance(pipeline.named_steps['classifier'], RandomForestClassifier):
        print("Using TreeExplainer for RF")
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_reduced)
        """
        Even though you’re working with binary classification, the SHAP library’s handling of 
        RandomForestClassifier can introduce complexity by generating 3D SHAP values.
        """
        shap_values = shap_values[:, :, 0]  # Select class 0 (or modify as needed)

    else:
        explainer = shap.Explainer(pipeline.named_steps['classifier'], X_reduced)
        shap_values = explainer(X_reduced)

    # Convert shap_values to the required format
    shap_values = shap.Explanation(
        values=shap_values,
        data=X_reduced,
        feature_names=selected_features
    )

    # Assuming you want to store SHAP values, feature names, and model predictions
    shap_data = {
        'shap_values': shap_values,
        'features': selected_features,
        'X_filtered': X_reduced  # Input features used for SHAP
    }
    # Save to a pickle file
    with open(f'{output_dir}/shap_data.pkl', 'wb') as f:
        pickle.dump(shap_data, f)

    # Generate beeswarm plot for all features
    shap.summary_plot(shap_values, plot_type="dot", feature_names=selected_features, max_display=10, show=False)
    plt.savefig(f'{output_dir}/shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Beeswarm plot saved as 'shap_beeswarm_plot.png'")

    plt.figure()
    shap.summary_plot(shap_values, plot_type="bar", feature_names=selected_features, max_display=10, show=False)
    plt.savefig(f'{output_dir}/shap_summary_plot_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_summary_plot_all_folds.png'")

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

    # memory.clear(warn=False)  # we want to reuse it.

    return pipeline


def get_results(model_name, ms_input_file, feature_reduce_choice, transpose_select, norm, log10):
    print(f"Starting ... {model_name} / {ms_input_file}")
    ms_file_name = Path(ms_input_file).stem
    df_file = load_data_from_file(ms_input_file, transpose_select)
    ms_info = load_data_frame(df_file)
    the_model = get_model(model_name, ms_info['y'])
    # <------------------------------------------
    pipeline = run_model(ms_info, the_model, ms_file_name, feature_reduce_choice, norm, log10)
    # plot_roc(ms_info, pipeline, model_name)

    # results = run_model_union(ms_info, the_model, ms_file_name, feature_reduce_choice, norm, log10)
    # <------------------------------------------


seed = 123456


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


"""
Keys - Adulteration (CAN) - RF, Adulteration (SOY) - RF, Freshness - SVM, Grade - RF
models - LogisticRegression, SVM, 'TensorFlow', 'RandomFortest'
Grade
python ../OilClassSHAP.py Grade_PP_filt_unnorm_9Sep2024.csv RandomForest Boruta false true false
Fresh
python ../OilClassSHAP.py Freshness_PP_unnorm_3Sep2024.csv SVM Boruta false true false
Adult Soy
python ../OilClassSHAP.py Adult_SOY-MALDI_TAG_unnorm_30Aug2024.csv RandomForest Boruta false true false
Adult Can
python ../OilClassSHAP.py Adult_CAN-MALDI_TAG_unnorm_29Aug2024.csv RandomForest Boruta false true false
"""


def main(model_name, ms_input_file, feature_reduce_choice, transpose, norm, log10):
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
        "model": model_name,
        "feature_reduce_choice": feature_reduce_choice,
        "transpose": transpose,
        "norm": norm,
        "log10": log10,
    }
    save_input_params(input_params, output_dir)

    transpose_select = transpose
    feature_reduce_choice = feature_reduce_choice  # None #'Boruta' #'Boruta' #'Boruta'
    # LogisticRegression SVM # 'TensorFlow' #'RandomForest'#'RandomForest' #'GradientBoosting' #'SVM' #'ElasticNet'

    get_results(model_name, ms_input_file, feature_reduce_choice, transpose_select, norm, log10)
    # plot_roc(pipeline,ms_info,model_name)
    ms_file_name = Path(ms_input_file).stem

    # Get the list of all installed packages and their versions
    installed_packages = importlib.metadata.distributions()
    # Specify the output file name
    ver_output_file = f'{output_dir}/package_versions.txt'

    # Write the package versions to the file
    with open(ver_output_file, "w") as f:
        for package in installed_packages:
            package_name = package.metadata["Name"]
            package_version = package.metadata["Version"]
            f.write(f"{package_name} using {package_version}\n")

    print(f"Package versions have been written to {ver_output_file}")

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
    parser.add_argument('model_name', type=str, help='Model Name.')
    parser.add_argument('feature_reduce_choice', type=str_or_none, nargs='?', default=None,
                        help='Choice of feature reduction method. Defaults to None.')
    parser.add_argument('transpose', type=str2bool, help='Transpose file (true/false)')
    parser.add_argument('norm', type=str2bool, help='Normalize (true/false)')
    parser.add_argument('log10', type=str2bool, help='Take the log 10 of input in the pipeline (true/false)')
    args = parser.parse_args()

    main(args.model_name, args.ms_input_file, args.feature_reduce_choice, args.transpose, args.norm,
         args.log10)  # , args.set_seed)
