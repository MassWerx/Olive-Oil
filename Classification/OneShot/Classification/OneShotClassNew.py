# https://github.com/scikit-learn-contrib/boruta_py

import pandas as pd
import numpy as np
import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
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

    print(f'shape of X values {X.shape} with type {X.dtypes.mode()[0]}')
    print(f'shape of y values {y.shape} with type {y.dtype}')
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
        # 'TensorFlow': MyKerasClf(n_classes=len(np.unique(y)), seed=seed) # Example for TensorFlow
    }

    best_params = {
        # norm {'classifier__max_depth': None, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100}
        # 27   {'classifier__max_depth': None, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 10, 'classifier__n_estimators': 300}
        # norm
        # 'RandomForest': {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500},
        'RandomForest': {'oob_score':True, 'max_features':'sqrt', 'n_estimators':500},
        'TensorFlow': {'learn_rate': .001, 'weight_constraint': 1},
        'SVM': {'C': 1, 'kernel': 'rbf'},
        'GradientBoosting': {'learning_rate': 0.1, 'max_depth': 5},
    }

    # Set input, name , model, best hyper parameters
    model = ml_algo_model[model_name]
    best_params_algo = best_params[model_name]
    model.set_params(**best_params_algo)

    # Write out params
    # Get hyperparameters
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
    output_dir = os.path.join(current_working_dir, 'output')
    result_metrics, result_roc, result_shap = results
    shap_values_combined, X_test_combined, selected_features_final = result_shap

    # Calculate mean SHAP values across all samples
    mean_shap_values = np.mean(shap_values_combined, axis=0)

    # Get indices of features with non-zero SHAP values
    # nonzero_indices = np.where(mean_shap_values != 0)[0]
    # We only want greater than 1.
    nonzero_indices = np.where(np.abs(mean_shap_values) >= 0.000001)[0]

    # Filter features and their SHAP values
    selected_features_filtered = [selected_features_final[i] for i in nonzero_indices]
    shap_values_combined_filtered = shap_values_combined[:, nonzero_indices]

    # Create the summary plot for the entire model
    plt.figure()
    shap.summary_plot(shap_values_combined_filtered, X_test_combined[:, nonzero_indices],
                      feature_names=selected_features_filtered, show=False)
    plt.savefig(f'{output_dir}/shap_summary_plot_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create the bar plot for the entire model
    plt.figure()
    shap.summary_plot(shap_values_combined_filtered, X_test_combined[:, nonzero_indices],
                      feature_names=selected_features_filtered, plot_type="bar", show=False)
    plt.savefig(f'{output_dir}/shap_bar_plot_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()

    top_n_features = 200
    sorted_idx = np.argsort(np.abs(mean_shap_values[nonzero_indices]))[::-1][:top_n_features]
    sorted_features = np.array(selected_features_filtered)[sorted_idx]
    sorted_mean_shap_values = mean_shap_values[nonzero_indices][sorted_idx]

    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['red' if val > 0 else 'blue' for val in sorted_mean_shap_values]
    plt.barh(sorted_features, sorted_mean_shap_values, color=colors)
    plt.xlabel('Mean SHAP value')
    plt.title('SHAP Bar Plot')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar_plot_custom_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create dependence plots for the top features
    top_features = sorted_idx[:3]  # Get indices of top 3 features

    for feature in top_features:
        plt.figure()
        shap.dependence_plot(feature, shap_values_combined_filtered, X_test_combined[:, nonzero_indices],
                             feature_names=selected_features_filtered, show=False)
        plt.savefig(f'{output_dir}/shap_dependence_plot_{selected_features_filtered[feature]}.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

    bar_plot = Image.open(f'{output_dir}/shap_bar_plot_custom_all_folds.png')
    summary_plot = Image.open(f'{output_dir}/shap_summary_plot_all_folds.png')

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(bar_plot)
    axes[0].axis('off')

    axes[1].imshow(summary_plot)
    axes[1].axis('off')

    fig.suptitle('Shap Bar Plot and Summary Plot', fontsize=16)

    plt.tight_layout()

    plt.savefig(f'{output_dir}/shap_combined_plots.png', dpi=300)
    plt.show()


def safe_log10(num):
    return np.log10(np.clip(num, a_min=1e-9, a_max=None))  # Clip values to avoid log10(0)


def run_model(ms_info, model_name, ms_file_name, feature_reduce_choice, normalize_select, log10_select):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
    #X = pd.DataFrame(X, columns=features)

    print(f'X shape {X.shape}')
    current_working_dir = os.getcwd()
    output_dir = os.path.join(current_working_dir, 'output')

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    print(f'Starting {model_name}')
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    # Save the outer fold indices

    if log10_select:
        log10_pipe = FunctionTransformer(safe_log10)
    else:
        log10_pipe = None

    if normalize_select:
        norm_pipe = Normalizer(norm='l1')
    else:
        norm_pipe = None

    pipeline = Pipeline([
        ('normalize', norm_pipe),
        ('transform', log10_pipe),  # FunctionTransformer(np.log10)), # log10_select
        ('scaler', StandardScaler()),
        ('Reduction', get_feature_reduction(feature_reduce_choice)),
        ('classifier', model_name)
    ], memory=memory)

    all_y_true = []
    all_y_scores = []
    all_balanced_accuracies = []
    all_recalls = []
    all_f1_scores = []
    all_precisions = []

    # Store SHAP values
    shap_values_list = []
    X_test_list = []
    fold_feature_sets = []
    selected_features_union = set()

    for train_idx, test_idx in outer_cv.split(X, y):
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
        if 'Reduction' in pipeline.named_steps and hasattr(pipeline.named_steps['Reduction'], 'get_support'):
            support_mask = pipeline.named_steps['Reduction'].get_support()
            selected_features = [features[i] for i, flag in enumerate(support_mask) if flag]
        else:
            selected_features = features  # No reduction step, use all features

        fold_feature_sets.append(selected_features)
        selected_features_union.update(selected_features)

        # Filter X_test to only include selected features
        X_test_filtered = X_test[selected_features]
        X_train_filtered = X_train[selected_features]

        # Compute SHAP values
        # Create SHAP TreeExplainer
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'], X_train_filtered)
        # explainer = shap.Explainer(pipeline.named_steps['classifier'], X_train_filtered)
        #explainer = shap.KernelExplainer(pipeline.named_steps['classifier'], X_train_filtered)
        shap_values = explainer(X_test_filtered)

        # Select SHAP values for the positive class (class 1)
        if shap_values.values.ndim == 3:
            shap_values_class_1 = shap_values.values[..., 1]
        else:
            shap_values_class_1 = shap_values.values

        shap_values_list.append((shap_values_class_1, selected_features))
        X_test_list.append(X_test_filtered)

    # Check consistency of features used across folds
    first_fold_features = fold_feature_sets[0]
    for fold_index, feature_set in enumerate(fold_feature_sets[1:], start=1):
        if set(feature_set) != set(first_fold_features):
            print(f"Warning: Feature set mismatch in fold {fold_index} compared to fold 0.")
            print(f"Fold 0 features: {first_fold_features}")
            print(f"Fold {fold_index} features: {feature_set}")

    # Align SHAP values to the union of all selected features
    aligned_shap_values_list = []
    for shap_values, selected_features in shap_values_list:
        aligned_shap_values = np.zeros((shap_values.shape[0], len(selected_features_union)))
        feature_index_map = {feature: i for i, feature in enumerate(selected_features_union)}
        for i, feature in enumerate(selected_features):
            if feature in feature_index_map:
                aligned_shap_values[:, feature_index_map[feature]] = shap_values[:, i]
        aligned_shap_values_list.append(aligned_shap_values)

    shap_values_combined = np.concatenate(aligned_shap_values_list, axis=0)
    X_test_combined = np.concatenate(X_test_list, axis=0)

    selected_features_final = list(selected_features_union)

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

    return (mean_balanced_accuracy, std_balanced_accuracy, mean_recall, std_recall, mean_f1, std_f1, mean_precision,
            std_precision), (roc_auc, fpr, tpr, thresholds), (
        shap_values_combined, X_test_combined, selected_features_final)

def run_model_debug(ms_info, model_name, ms_file_name, feature_reduce_choice, normalize_select, log10_select):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
    #X = pd.DataFrame(X, columns=features)

    # Check for constant or near-constant columns
    constant_columns = X.nunique() <= 1
    print("Constant or near-constant columns:")
    print(X.columns[constant_columns])

    # Calculate variance for each column
    low_variance_columns = X.var() < 1e-6
    print("Columns with low variance:")
    print(X.columns[low_variance_columns])




    print(f'X shape {X.shape}')
    current_working_dir = os.getcwd()
    output_dir = os.path.join(current_working_dir, 'output')

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    print(f'Starting {model_name}')
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    # Save the outer fold indices

    if log10_select:
        log10_pipe = FunctionTransformer(safe_log10)
    else:
        log10_pipe = None

    if normalize_select:
        norm_pipe = Normalizer(norm='l1')
    else:
        norm_pipe = None

    pipeline = Pipeline([
        ('normalize', norm_pipe),
        ('transform', log10_pipe),  # FunctionTransformer(np.log10)), # log10_select
        ('scaler', StandardScaler()),
        ('Reduction', get_feature_reduction(feature_reduce_choice)),
        ('classifier', model_name)
    ], memory=memory)

    all_y_true = []
    all_y_scores = []
    all_balanced_accuracies = []
    all_recalls = []
    all_f1_scores = []
    all_precisions = []

    # Store SHAP values
    shap_values_list = []
    X_test_list = []
    fold_feature_sets = []
    selected_features_union = set()

    for train_idx, test_idx in outer_cv.split(X, y):
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
        if 'Reduction' in pipeline.named_steps and hasattr(pipeline.named_steps['Reduction'], 'get_support'):
            support_mask = pipeline.named_steps['Reduction'].get_support()
            selected_features = [features[i] for i, flag in enumerate(support_mask) if flag]
        else:
            print(f'No reduction step, use all features')
            selected_features = features  # No reduction step, use all features

        fold_feature_sets.append(selected_features)
        selected_features_union.update(selected_features)

        # Filter X_test to only include selected features
        X_test_filtered = X_test[selected_features]
        X_train_filtered = X_train[selected_features]

        # Compute SHAP values
        # Create SHAP TreeExplainer
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'], X_train_filtered)
        # explainer = shap.Explainer(pipeline.named_steps['classifier'], X_train_filtered)
        #explainer = shap.KernelExplainer(pipeline.named_steps['classifier'], X_train_filtered)
        shap_values = explainer(X_test_filtered)



        # Ensure shap_values is a NumPy array
        import numpy as np

        # Check if shap_values is a list (which it typically is for multi-class classification)
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)  # Convert the list to a NumPy array

            # If shap_values is now a 3D array (classes, samples, features), handle it for each class
            for i, class_shap_values in enumerate(shap_values):
                # Check for identical SHAP values across all instances for this class
                identical_shap_values = np.all(class_shap_values == class_shap_values[0, :], axis=0)

                # Print features with identical SHAP values for this class
                print(f"Features with identical SHAP values across all instances for class {i}:",
                      X.columns[identical_shap_values])

                # Optionally, filter out these features for this class
                class_shap_values_filtered = class_shap_values[:, ~identical_shap_values]
                X_filtered = X.iloc[:, ~identical_shap_values]

                # Now plot the filtered SHAP values for this class
                shap.summary_plot(class_shap_values_filtered, X_filtered, plot_type="bar")
        else:
            # If shap_values is already a NumPy array, handle it directly
            identical_shap_values = np.all(shap_values == shap_values[0, :], axis=0)

            # Print features with identical SHAP values

            # Filter and plot as necessary
            shap_values_filtered = shap_values[:, ~identical_shap_values]
            X_filtered = X.iloc[:, ~identical_shap_values]
            shap.summary_plot(shap_values_filtered, X_filtered)

        # Select SHAP values for the positive class (class 1)
        if shap_values.values.ndim == 3:
            shap_values_class_1 = shap_values.values[..., 1]
        else:
            shap_values_class_1 = shap_values.values

        shap_values_list.append((shap_values_class_1, selected_features))
        X_test_list.append(X_test_filtered)

    # Check consistency of features used across folds
    first_fold_features = fold_feature_sets[0]
    for fold_index, feature_set in enumerate(fold_feature_sets[1:], start=1):
        if set(feature_set) != set(first_fold_features):
            print(f"Warning: Feature set mismatch in fold {fold_index} compared to fold 0.")
            print(f"Fold 0 features: {first_fold_features}")
            print(f"Fold {fold_index} features: {feature_set}")

    # Align SHAP values to the union of all selected features
    aligned_shap_values_list = []
    for shap_values, selected_features in shap_values_list:
        aligned_shap_values = np.zeros((shap_values.shape[0], len(selected_features_union)))
        feature_index_map = {feature: i for i, feature in enumerate(selected_features_union)}
        for i, feature in enumerate(selected_features):
            if feature in feature_index_map:
                aligned_shap_values[:, feature_index_map[feature]] = shap_values[:, i]
        aligned_shap_values_list.append(aligned_shap_values)

    shap_values_combined = np.concatenate(aligned_shap_values_list, axis=0)
    X_test_combined = np.concatenate(X_test_list, axis=0)

    selected_features_final = list(selected_features_union)

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

    return (mean_balanced_accuracy, std_balanced_accuracy, mean_recall, std_recall, mean_f1, std_f1, mean_precision,
            std_precision), (roc_auc, fpr, tpr, thresholds), (
        shap_values_combined, X_test_combined, selected_features_final)


def get_results(model_name, ms_input_file, feature_reduce_choice, transpose_select, norm, log10):
    output_dir = os.path.join(current_working_dir, 'output')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"Starting ... {model_name} / {ms_input_file}")
    ms_file_name = Path(ms_input_file).stem
    df_file = load_data_from_file(ms_input_file, transpose_select)
    ms_info = load_data_frame(df_file)
    the_model = get_model(model_name, ms_info['y'])
    results = run_model(ms_info, the_model, ms_file_name, feature_reduce_choice, norm, log10)
    return results, ms_info


seed = 1234546


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower() == 'none':  # Handle 'none' explicitly
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def main(ms_input_file, feature_reduce_choice, transpose, norm, log10):
    print("Starting ... ")
    model_name = 'RandomForest'  # 'TensorFlow' #'RandomForest'#'RandomForest' #'GradientBoosting' #'SVM' #'ElasticNet'
    results, ms_info = get_results(model_name, ms_input_file, feature_reduce_choice, transpose, norm, log10)
    ms_file_name = Path(ms_input_file).stem
    gen_output(results, model_name, ms_file_name)
    plot_roc(results, model_name)
    features = ms_info['feature_names']
    plot_shap_values(results, model_name, ms_file_name)
    # with open(os.path.join(current_working_dir, f'output_{name}/overall_{name}.txt'), 'w') as f:
    #   f.write(f'Overall So Far: {overall_results}\n')
    output_dir = os.path.join(current_working_dir, 'output')
    if not os.path.exists(os.path.join(current_working_dir, 'zipFiles')):
        os.makedirs(os.path.join(current_working_dir, 'zipFiles'))

    create_zip_file_output(os.path.join(current_working_dir, f'zipFiles/{model_name}_{ms_file_name}'), output_dir)

    # python ../../GridClassFinal.py Grade_DART-PP_filter_unnorm_7Mar23.csv                      Boruta true true false


"""
python ../../OneShotClassOld.py Adult_MALDI-TAG_EVOO_EVO-CAN_no-outliers_unnorm_31Oct23.csv Boruta false true true
python ../../OneShotClassOld.py DART-PP-unnorm-filter_1Mar23.csv Boruta true true false
python ../../OneShotClassOld.py Grade_DART-PP_filter_unnorm_7Mar23.csv Boruta true true false
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run regression models with feature reduction.')
    parser.add_argument('ms_input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('feature_reduce_choice', type=str2bool, help='Choice of feature reduction method.')
    parser.add_argument('transpose', type=str2bool, help='Transpose file (true/false)')
    parser.add_argument('norm', type=str2bool, help='Normalize (true/false)')
    parser.add_argument('log10', type=str2bool, help='Take the log 10 of input in the pipeline (true/false)')
    args = parser.parse_args()

    main(args.ms_input_file, args.feature_reduce_choice, args.transpose, args.norm, args.log10)  #, args.set_seed)
