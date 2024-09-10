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

from joblib import Memory

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
    with ZipFile(f'{output_file_name}.zip', 'w') as zipObj:
        for folderName, subfolders, filenames in os.walk(base_dir):
            for filename in filenames:
                file_path = os.path.join(folderName, filename)
                zipObj.write(file_path, basename(file_path))


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


def save_features_for_fold(features, fold_idx, output_dir):
    features_df = pd.DataFrame(features)
    features_df.to_csv(os.path.join(output_dir, f'features_fold_{fold_idx}.csv'), index=False)


def load_data_from_file(input_file, flip):
    if flip:
        df = pd.read_csv(input_file, header=None).T
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    else:
        df = pd.read_csv(input_file, header=0)
    return df


def load_data_frame(input_dataframe):
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


class MyKerasClf:
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


def get_models(y):
    list_of_models = [
        ('SVM', SVC(probability=True, random_state=seed), {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }),
        ('AdaBoost', AdaBoostClassifier(random_state=seed), {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 1]
        }),
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=seed), {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs']
        }),
        ('KNeighbors', KNeighborsClassifier(), {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }),
        ('GradientBoosting', GradientBoostingClassifier(random_state=seed), {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }),
        ('RandomForest', RandomForestClassifier(n_jobs=-1, random_state=seed), {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }),
        ('ANN', MyKerasClf(n_classes=len(np.unique(y)), seed=seed), {
            'classifier__learn_rate': [0.001, 0.01],
            'classifier__weight_constraint': [0, 1]
        })
    ]

    list_of_models_short = [
        ('KNeighbors', KNeighborsClassifier(), {
            'classifier__n_neighbors': [3],
        }),
    ]
    return list_of_models


def safe_log10(num):
    return np.log10(np.clip(num, a_min=1e-9, a_max=None))  # Clip values to avoid log10(0)


def run_models_cv(ms_info, list_of_models, ms_file_name, feature_reduce_choice, normalize_select, log10_select):
    X = ms_info['X']
    y = ms_info['y']
    current_working_dir = os.getcwd()
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    overall_results = []

    if log10_select:
        log10_pipe = FunctionTransformer(safe_log10)
    else:
        log10_pipe = None

    if normalize_select:
        norm_pipe = Normalizer(norm='l1')
    else:
        norm_pipe = None

    for name, model, param_grid in list_of_models:
        print(f'Starting {name}')

        pipeline = Pipeline([
            ('normalize', norm_pipe),
            ('transform', log10_pipe),  # FunctionTransformer(np.log10)), # log10_select
            ('scaler', StandardScaler()),
            ('Reduction', get_feature_reduction(feature_reduce_choice)),
            ('classifier', model)
        ], memory=memory)

        # Perform grid search using the entire data
        grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=grid_cv, scoring='neg_log_loss')
        grid_search.fit(X, y)
        best_params = grid_search.best_params_

        # Save the best parameters
        dirpath = Path(os.path.join(current_working_dir, f'output_{name}'))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        with open(f'{dirpath}/best_params_{name}.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        # Use the best parameters for the model
        pipeline.set_params(**best_params)

        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
        all_scores = {'accuracy': [], 'balanced_accuracy': [], 'recall': [], 'f1': [], 'precision': []}

        scores = cross_validate(pipeline, X, y, cv=outer_cv,
                                scoring=['accuracy', 'balanced_accuracy', 'recall', 'f1', 'precision'])

        all_scores['accuracy'].extend(scores['test_accuracy'])
        all_scores['balanced_accuracy'].extend(scores['test_balanced_accuracy'])
        all_scores['recall'].extend(scores['test_recall'])
        all_scores['f1'].extend(scores['test_f1'])
        all_scores['precision'].extend(scores['test_precision'])

        # Calculate mean and standard deviation for metrics
        mean_balanced_accuracy = np.mean(all_scores['balanced_accuracy'])
        std_balanced_accuracy = np.std(all_scores['balanced_accuracy'])
        mean_recall = np.mean(all_scores['recall'])
        std_recall = np.std(all_scores['recall'])
        mean_f1 = np.mean(all_scores['f1'])
        std_f1 = np.std(all_scores['f1'])
        mean_precision = np.mean(all_scores['precision'])
        std_precision = np.std(all_scores['precision'])
        mean_score = np.mean(all_scores['accuracy'])
        std_score = np.std(all_scores['accuracy'])

        # Save cross-validation scores for debugging
        scores_df = pd.DataFrame(all_scores)
        scores_df.to_csv(f'{dirpath}/cv_{name}.csv', index=False)

        with open(f'{dirpath}/metrics_cv_{name}.txt', 'w') as f:
            f.write(f'Bal.Acc.avg: {mean_balanced_accuracy}\n')
            f.write(f'Bal.Acc.sd: {std_balanced_accuracy}\n')
            f.write(f'Recall.avg: {mean_recall}\n')
            f.write(f'Recall.sd: {std_recall}\n')
            f.write(f'Precision.avg: {mean_precision}\n')
            f.write(f'Precision.sd: {std_precision}\n')
            f.write(f'F1.avg: {mean_f1}\n')
            f.write(f'F1.sd: {std_f1}\n')
            f.write(f'Accuracy.avg: {mean_score}\n')
            f.write(f'Accuracy.sd: {std_score}\n')

        if not os.path.exists(os.path.join(current_working_dir, 'zipFiles')):
            os.makedirs(os.path.join(current_working_dir, 'zipFiles'))

        create_zip_file_output(os.path.join(current_working_dir, f'zipFiles/{name}_{ms_file_name}'), dirpath)

    shutil.rmtree(cachedir)


def plot_roc(ms_info, pipeline, model_name, output_dir, n_splits=5, n_repeats=10):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

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

    return roc_auc


def plot_SHAP(model, pipeline, X_reduced, selected_features, output_dir, name):
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
    elif isinstance(pipeline.named_steps['classifier'], (SVC, KNeighborsClassifier, AdaBoostClassifier)):
        print("\nUsing KernelExplainer for most\n")
        # For SVM model
        explainer = shap.KernelExplainer(pipeline.named_steps['classifier'].predict_proba, X_reduced)
        shap_values = explainer.shap_values(X_reduced)
        shap_values = shap_values[:, :, 0]  # Select class 0 (or modify as needed)
        shap_values = np.array(shap_values)

    # Adult Can
    elif isinstance(pipeline.named_steps['classifier'], (RandomForestClassifier, GradientBoostingClassifier)):
        print("Using TreeExplainer for RF, GB, or AdaBoost")
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_reduced)
        """
        Even though you’re working with binary classification, the SHAP library’s handling of 
        RandomForestClassifier can introduce complexity by generating 3D SHAP values.
        """
        if isinstance(pipeline.named_steps['classifier'], RandomForestClassifier):
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
    with open(f'{output_dir}/shap_data_{name}.pkl', 'wb') as f:
        pickle.dump(shap_data, f)

    # Generate beeswarm plot for all features
    shap.summary_plot(shap_values, plot_type="dot", feature_names=selected_features, max_display=10, show=False)
    plt.savefig(f'{output_dir}/shap_beeswarm_{name}_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Beeswarm plot saved as 'shap_beeswarm_plot.png'")

    plt.figure()
    shap.summary_plot(shap_values, plot_type="bar", feature_names=selected_features, max_display=10, show=False)
    plt.savefig(f'{output_dir}/shap_summary_plot_{name}_all_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Summary plot for all features saved as 'shap_summary_plot_all_folds.png'")

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
    print("Summary plot for all features saved as 'shap_combined_plots.png'")


def run_models_cv_avg_sd(ms_info, list_of_models, ms_file_name, feature_reduce_choice, normalize_select, log10_select,
                         n_splits=5, n_repeats=10, round_scores=True):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
    current_working_dir = os.getcwd()
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    overall_results = []

    log10_pipe = FunctionTransformer(np.log10) if log10_select else 'passthrough'
    norm_pipe = Normalizer(norm='l1') if normalize_select else 'passthrough'

    for name, model, param_grid in list_of_models:
        print(f'Starting {name}')

        pipeline = Pipeline([
            ('normalize', norm_pipe),
            ('transform', log10_pipe),
            ('scaler', StandardScaler()),
            ('Reduction', get_feature_reduction(feature_reduce_choice)),
            ('classifier', model)
        ], memory=memory)

        # Perform grid search using the entire data
        grid_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=grid_cv, scoring='neg_log_loss')
        grid_search.fit(X, y)
        best_params = grid_search.best_params_

        # Use the best parameters for the model
        pipeline.set_params(**best_params)

        # Save the best parameters
        dirpath = Path(os.path.join(current_working_dir, f'output_{name}'))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        with open(f'{dirpath}/best_params_{name}.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        # Extract the best parameters specific to the model
        model_best_params = {k.replace('classifier__', ''): v for k, v in best_params.items() if
                             k.startswith('classifier__')}

        # Set the best parameters on the model and save all model parameters
        model.set_params(**model_best_params)
        model_hyperparameters = model.get_params()

        # Save all model parameters to a file
        with open(f'{dirpath}/all_params_{name}.txt', 'w') as f:
            for key, value in model_hyperparameters.items():
                f.write(f"{key}: {value}\n")

        outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        all_scores = {'accuracy': [], 'balanced_accuracy': [], 'recall': [], 'f1': [], 'precision': []}

        scores = cross_validate(pipeline, X, y, cv=outer_cv,
                                scoring=['accuracy', 'balanced_accuracy', 'recall', 'f1', 'precision'])

        # Process scores for each repeat
        repeat_averages = {metric: [] for metric in all_scores.keys()}

        for repeat in range(n_repeats):
            start_idx = repeat * n_splits
            end_idx = start_idx + n_splits

            for metric in all_scores.keys():
                fold_scores = scores[f'test_{metric}'][start_idx:end_idx]
                repeat_avg = np.mean(fold_scores)
                repeat_averages[metric].append(repeat_avg)
                all_scores[metric].extend(fold_scores)  # Store the raw fold scores

        # Calculate mean and standard deviation for each metric
        mean_balanced_accuracy = np.mean(repeat_averages['balanced_accuracy'])
        std_balanced_accuracy = np.std(repeat_averages['balanced_accuracy'])
        mean_recall = np.mean(repeat_averages['recall'])
        std_recall = np.std(repeat_averages['recall'])
        mean_f1 = np.mean(repeat_averages['f1'])
        std_f1 = np.std(repeat_averages['f1'])
        mean_precision = np.mean(repeat_averages['precision'])
        std_precision = np.std(repeat_averages['precision'])
        mean_score = np.mean(repeat_averages['accuracy'])
        std_score = np.std(repeat_averages['accuracy'])

        if round_scores:
            mean_balanced_accuracy = round(mean_balanced_accuracy, 3)
            std_balanced_accuracy = round(std_balanced_accuracy, 3)
            mean_recall = round(mean_recall, 3)
            std_recall = round(std_recall, 3)
            mean_f1 = round(mean_f1, 3)
            std_f1 = round(std_f1, 3)
            mean_precision = round(mean_precision, 3)
            std_precision = round(std_precision, 3)
            mean_score = round(mean_score, 3)
            std_score = round(std_score, 3)

        # Save cross-validation scores for debugging
        scores_df = pd.DataFrame(all_scores)
        if round_scores:
            scores_df = scores_df.round(3)
        scores_df.to_csv(f'{dirpath}/cv_{name}.csv', index=False)

        roc_auc = plot_roc(ms_info, pipeline, name, dirpath)

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

        print(f'{len(y)} & {len(selected_features)} = {selected_features}')
        # Save the selected features to a file
        with open(f'{dirpath}/selected_feature_{name}.txt', 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")

        plot_SHAP(model, pipeline, X_reduced, selected_features, dirpath, name)

        with open(f'{dirpath}/metrics_cv_{name}.txt', 'w') as f:
            f.write(f'Bal.Acc.avg: {mean_balanced_accuracy}\n')
            f.write(f'Bal.Acc.sd: {std_balanced_accuracy}\n')
            f.write(f'Recall.avg: {mean_recall}\n')
            f.write(f'Recall.sd: {std_recall}\n')
            f.write(f'Precision.avg: {mean_precision}\n')
            f.write(f'Precision.sd: {std_precision}\n')
            f.write(f'F1.avg: {mean_f1}\n')
            f.write(f'F1.sd: {std_f1}\n')
            f.write(f'Accuracy.avg: {mean_score}\n')
            f.write(f'Accuracy.sd: {std_score}\n')
            f.write(f'AUC: {roc_auc}\n')

        if not os.path.exists(os.path.join(current_working_dir, 'zipFiles')):
            os.makedirs(os.path.join(current_working_dir, 'zipFiles'))

        create_zip_file_output(os.path.join(current_working_dir, f'zipFiles/{name}_{ms_file_name}'), dirpath)

    shutil.rmtree(cachedir)


def run_models_cv_score(ms_info, list_of_models, ms_file_name, feature_reduce_choice, normalize_select, log10_select):
    X = ms_info['X']
    y = ms_info['y']
    current_working_dir = os.getcwd()
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    overall_results = []

    if log10_select:
        log10_pipe = FunctionTransformer(safe_log10)
    else:
        log10_pipe = None

    if normalize_select:
        norm_pipe = Normalizer(norm='l1')
    else:
        norm_pipe = None

    for name, model, param_grid in list_of_models:
        print(f'Starting {name}')

        pipeline = Pipeline([
            ('normalize', norm_pipe),
            ('transform', log10_pipe),  # FunctionTransformer(np.log10)), # log10_select
            ('scaler', StandardScaler()),
            ('Reduction', get_feature_reduction(feature_reduce_choice)),
            ('classifier', model)
        ], memory=memory)

        # Perform grid search using the entire data
        grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=grid_cv, scoring='accuracy')
        grid_search.fit(X, y)
        best_params = grid_search.best_params_

        # Use the best parameters for the model
        pipeline.set_params(**best_params)

        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)

        accuracy_scores = cross_val_score(pipeline, X, y, cv=outer_cv, scoring='accuracy')
        print("Accuracy scores done")

        balanced_accuracy_scores = cross_val_score(pipeline, X, y, cv=outer_cv, scoring='balanced_accuracy')
        print("Balanced accuracy scores done")

        recall_scores = cross_val_score(pipeline, X, y, cv=outer_cv, scoring='recall')
        print("Recall scores done")

        f1_scores = cross_val_score(pipeline, X, y, cv=outer_cv, scoring='f1')
        print("F1 scores done")

        precision_scores = cross_val_score(pipeline, X, y, cv=outer_cv, scoring='precision')
        print("Precision scores done")

        # Calculate mean and standard deviation for metrics
        mean_balanced_accuracy = np.mean(balanced_accuracy_scores)
        std_balanced_accuracy = np.std(balanced_accuracy_scores)
        mean_recall = np.mean(recall_scores)
        std_recall = np.std(recall_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_precision = np.mean(precision_scores)
        std_precision = np.std(precision_scores)
        mean_score = np.mean(accuracy_scores)
        std_score = np.std(accuracy_scores)

        dirpath = Path(os.path.join(current_working_dir, f'output_{name}'))

        # Save cross-validation scores for debugging
        scores_data = {
            'accuracy_scores': accuracy_scores,
            'balanced_accuracy_scores': balanced_accuracy_scores,
            'recall_scores': recall_scores,
            'f1_scores': f1_scores,
            'precision_scores': precision_scores
        }
        scores_df = pd.DataFrame(scores_data)
        scores_df.to_csv(f'{dirpath}/cv_scores_{name}.csv', index=False)

        with open(f'{dirpath}/metrics_cv_score{name}.txt', 'w') as f:
            f.write(f'Bal.Acc.avg: {mean_balanced_accuracy}\n')
            f.write(f'Bal.Acc.sd: {std_balanced_accuracy}\n')
            f.write(f'Recall.avg: {mean_recall}\n')
            f.write(f'Recall.sd: {std_recall}\n')
            f.write(f'Precision.avg: {mean_precision}\n')
            f.write(f'Precision.sd: {std_precision}\n')
            f.write(f'F1.avg: {mean_f1}\n')
            f.write(f'F1.sd: {std_f1}\n')
            f.write(f'Accuracy.avg: {mean_score}\n')
            f.write(f'Accuracy.sd: {std_score}\n')

        if not os.path.exists(os.path.join(current_working_dir, 'zipFiles')):
            os.makedirs(os.path.join(current_working_dir, 'zipFiles'))

        create_zip_file_output(os.path.join(current_working_dir, f'zipFiles/{name}_{ms_file_name}'), dirpath)

    shutil.rmtree(cachedir)


def run_models_loop(ms_info, list_of_models, ms_file_name, feature_reduce_choice, log10_select, normalize_select):
    X = ms_info['X']
    y = ms_info['y']
    current_working_dir = os.getcwd()

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    overall_results = []

    for name, model, param_grid in list_of_models:
        print(f'Starting {name}')

        dirpath = Path(os.path.join(current_working_dir, f'output_{name}'))

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('Reduction', get_feature_reduction(feature_reduce_choice)),
            ('classifier', model)
        ], memory=memory)

        # Perform grid search using the entire data
        grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=grid_cv, scoring='accuracy')
        grid_search.fit(X, y)
        best_params = grid_search.best_params_

        # Use the best parameters for the model
        pipeline.set_params(**best_params)

        outer_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

        all_accuracy_scores = []
        all_balanced_accuracy_scores = []
        all_recall_scores = []
        all_f1_scores = []
        all_precision_scores = []

        for repeat in range(10):
            print(f'Repeat {repeat + 1}/10')

            accuracy_scores = cross_val_score(pipeline, X, y, cv=outer_cv, scoring='accuracy')
            balanced_accuracy_scores = cross_val_score(pipeline, X, y, cv=outer_cv, scoring='balanced_accuracy')
            recall_scores = cross_val_score(pipeline, X, y, cv=outer_cv, scoring='recall')
            f1_scores = cross_val_score(pipeline, X, y, cv=outer_cv, scoring='f1')
            precision_scores = cross_val_score(pipeline, X, y, cv=outer_cv, scoring='precision')

            print(f'This is the accuracy score {accuracy_scores}')
            all_accuracy_scores.append(np.mean(accuracy_scores))
            all_balanced_accuracy_scores.append(np.mean(balanced_accuracy_scores))
            all_recall_scores.append(np.mean(recall_scores))
            all_f1_scores.append(np.mean(f1_scores))
            all_precision_scores.append(np.mean(precision_scores))

        # Save cross-validation scores for debugging
        scores_data = {
            'accuracy_scores': all_accuracy_scores,
            'balanced_accuracy_scores': all_balanced_accuracy_scores,
            'recall_scores': all_recall_scores,
            'f1_scores': all_f1_scores,
            'precision_scores': all_precision_scores
        }
        scores_df = pd.DataFrame(scores_data)
        scores_df.to_csv(f'{dirpath}/loop_cross_val_scores_{name}.csv', index=False)

        # Calculate mean and standard deviation for metrics
        mean_balanced_accuracy = np.mean(all_balanced_accuracy_scores)
        std_balanced_accuracy = np.std(all_balanced_accuracy_scores)
        mean_recall = np.mean(all_recall_scores)
        std_recall = np.std(all_recall_scores)
        mean_f1 = np.mean(all_f1_scores)
        std_f1 = np.std(all_f1_scores)
        mean_precision = np.mean(all_precision_scores)
        std_precision = np.std(all_precision_scores)
        mean_score = np.mean(all_accuracy_scores)
        std_score = np.std(all_accuracy_scores)

        with open(f'{dirpath}/metrics_loop_{name}.txt', 'w') as f:
            f.write(f'Bal.Acc.avg: {mean_balanced_accuracy}\n')
            f.write(f'Bal.Acc.sd: {std_balanced_accuracy}\n')
            f.write(f'Recall.avg: {mean_recall}\n')
            f.write(f'Recall.sd: {std_recall}\n')
            f.write(f'Precision.avg: {mean_precision}\n')
            f.write(f'Precision.sd: {std_precision}\n')
            f.write(f'F1.avg: {mean_f1}\n')
            f.write(f'F1.sd: {std_f1}\n')
            f.write(f'Accuracy.avg: {mean_score}\n')
            f.write(f'Accuracy.sd: {std_score}\n')

        if not os.path.exists(os.path.join(current_working_dir, 'zipFiles')):
            os.makedirs(os.path.join(current_working_dir, 'zipFiles'))

        create_zip_file_output(os.path.join(current_working_dir, f'zipFiles/{name}_{ms_file_name}'), dirpath)

    shutil.rmtree(cachedir)


def run_models_org(ms_info, list_of_models, ms_file_name, feature_reduce_choice):
    X = ms_info['X']
    y = ms_info['y']

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    overall_results = []

    for name, model, param_grid in list_of_models:
        print(f'Starting {name}')

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('Reduction', get_feature_reduction(feature_reduce_choice)),
            ('classifier', model)
        ], memory=memory)

        # Perform grid search using the entire data
        grid_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=grid_cv, scoring='accuracy')
        grid_search.fit(X, y)
        best_params = grid_search.best_params_

        # Use the best parameters for the model
        pipeline.set_params(**best_params)

        all_roc_data = []
        all_splits_data = []  # Store data for each split
        all_features_data = []
        all_orig_data = []
        all_orig_data.append(
            ["accuracy_score", "balanced_accuracy_score", "recall_score", "f1_score", "precision_score"])
        all_repeat_mean_balanced_accuracies = []
        all_repeat_std_balanced_accuracies = []
        all_repeat_mean_recalls = []
        all_repeat_std_recalls = []
        all_repeat_mean_f1_scores = []
        all_repeat_std_f1_scores = []
        all_repeat_mean_precisions = []
        all_repeat_std_precisions = []
        all_repeat_mean_accuracies = []
        all_repeat_std_accuracies = []

        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)

        for repeat in range(10):
            print(f'Repeat {repeat + 1}/10')
            repeat_accuracy = []
            repeat_balanced_accuracies = []
            repeat_recalls = []
            repeat_f1_scores = []
            repeat_precisions = []

            #scores = cross_val_score(pipeline, X, y, cv=outer_cv)

            for train_idx, test_idx in outer_cv.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                best_model = pipeline.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)

                #y_true_repeat.extend(y_test) # we are not doing accuracy_score(y_true_repeat, y_pred_repeat)
                #y_pred_repeat.extend(y_pred)

                repeat_accuracy.append(accuracy_score(y_test, y_pred))
                repeat_balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
                repeat_recalls.append(recall_score(y_test, y_pred))
                repeat_f1_scores.append(f1_score(y_test, y_pred))
                repeat_precisions.append(precision_score(y_test, y_pred))

                all_orig_data.append([accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred),
                                      recall_score(y_test, y_pred), f1_score(y_test, y_pred),
                                      precision_score(y_test, y_pred)])

            all_repeat_mean_accuracies.append(np.mean(repeat_accuracy))
            all_repeat_std_accuracies.append(np.std(repeat_accuracy))
            all_repeat_mean_balanced_accuracies.append(np.mean(repeat_balanced_accuracies))
            all_repeat_std_balanced_accuracies.append(np.std(repeat_balanced_accuracies))
            all_repeat_mean_recalls.append(np.mean(repeat_recalls))
            all_repeat_std_recalls.append(np.std(repeat_recalls))
            all_repeat_mean_f1_scores.append(np.mean(repeat_f1_scores))
            all_repeat_std_f1_scores.append(np.std(repeat_f1_scores))
            all_repeat_mean_precisions.append(np.mean(repeat_precisions))
            all_repeat_std_precisions.append(np.std(repeat_precisions))

        # Calculate mean and standard deviation for metrics
        mean_balanced_accuracy = np.mean(all_repeat_mean_balanced_accuracies)
        std_balanced_accuracy = np.std(all_repeat_mean_balanced_accuracies)
        mean_recall = np.mean(all_repeat_mean_recalls)
        std_recall = np.std(all_repeat_mean_recalls)
        mean_f1 = np.mean(all_repeat_mean_f1_scores)
        std_f1 = np.std(all_repeat_mean_f1_scores)
        mean_precision = np.mean(all_repeat_mean_precisions)
        std_precision = np.std(all_repeat_mean_precisions)
        mean_score = np.mean(all_repeat_mean_accuracies)
        std_score = np.std(all_repeat_mean_accuracies)

        dirpath = Path(os.path.join(current_working_dir, f'output_{name}'))

        """
        all_accuracies_data.append(accuracy_score(y_test, y_pred))  # Store features for each split
        all_balanced_accuracies_data.append(balanced_accuracy_score(y_test, y_pred))
        all_recalls_data.append(recall_score(y_test, y_pred))
        all_f1_scores_data.append(f1_score(y_test, y_pred))
        all_precisions_data.append(precision_score(y_test, y_pred))
        """

        all_precisions_data_df = pd.DataFrame(all_orig_data)
        all_precisions_data_df.to_csv(f'{dirpath}/values_orig_{name}.csv', index=False)

        # Combine all features data into a single DataFrame and save to CSV
        all_features_data_df = pd.DataFrame(all_features_data)
        all_features_data_df.to_csv(f'{dirpath}/features_data_orig_{name}.csv', index=False)

        # Combine ROC data
        """fpr_all, tpr_all, roc_auc_all = zip(*all_roc_data)
        mean_fpr = np.mean(fpr_all, axis=0)
        mean_tpr = np.mean(tpr_all, axis=0)
        mean_roc_auc = np.mean(roc_auc_all)"""

        #roc_data = pd.DataFrame({'fpr': mean_fpr, 'tpr': mean_tpr})
        #roc_data.to_csv(f'{dirpath}/roc_data_{name}.csv', index=False)

        with open(f'{dirpath}/metrics_orig_{name}.txt', 'w') as f:
            #f.write(f'Mean balanced accuracy: {mean_balanced_accuracy}\n')
            f.write(f'Bal.Acc.avg: {mean_balanced_accuracy}\n')
            f.write(f'Bal.Acc.sd: {std_balanced_accuracy}\n')
            f.write(f'Recall.avg: {mean_recall}\n')
            f.write(f'Recall.sd: {std_recall}\n')
            f.write(f'Precision.avg: {mean_precision}\n')
            f.write(f'Precision.sd: {std_precision}\n')
            f.write(f'F1.avg: {mean_f1}\n')
            f.write(f'F1.sd: {std_f1}\n')
            f.write(f'acu.mean: {mean_score}\n')
            f.write(f'acu.sd: {std_score}\n')

            # f.write(f'Mean AUC: {mean_roc_auc}\n')
            # f.write(f'Best parameters: {grid_search.best_params_}\n')

        with open(f'{dirpath}/overall_orig_{name}.txt', 'w') as f:
            f.write(f'Overall So Far: {overall_results}\n')

        if not os.path.exists(os.path.join(current_working_dir, 'zipFiles')):
            os.makedirs(os.path.join(current_working_dir, 'zipFiles'))

        create_zip_file_output(os.path.join(current_working_dir, f'zipFiles/{name}_{ms_file_name}'), dirpath)

    shutil.rmtree(cachedir)


def resetDirs(list_of_models):
    for name, model, param_grid in list_of_models:
        dirpath = Path(os.path.join(current_working_dir, f'output_{name}'))
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        os.makedirs(dirpath)


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

    ms_file_name = Path(ms_input_file).stem
    df_file = load_data_from_file(ms_input_file, transpose)
    ms_info = load_data_frame(df_file)
    list_of_models = get_models(ms_info['y'])
    resetDirs(list_of_models)

    """print(f"------> Starting orig {ms_input_file} / {feature_reduce_choice}... with {seed}")
    run_models_org(ms_info, list_of_models, ms_file_name, feature_reduce_choice)"""

    """print(f"------> Starting CV {ms_input_file} / {feature_reduce_choice}... with {seed}")
    run_models_cv(ms_info, list_of_models, ms_file_name, feature_reduce_choice, norm, log10)"""

    print(f"------> Starting CV avg SD {ms_input_file} / {feature_reduce_choice}... with {seed}")
    run_models_cv_avg_sd(ms_info, list_of_models, ms_file_name, feature_reduce_choice, norm, log10)

    # print(f"------> Starting CV_Score ... with {seed}")
    # run_models_cv_score(ms_info, list_of_models, ms_file_name, feature_reduce_choice, norm, log10)

    """print(f"-------> Starting Loop ... with {seed}")
    run_models_loop(ms_info, list_of_models, ms_file_name, feature_reduce_choice, norm, log10)"""


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

"""if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run regression models with feature reduction.')
    parser.add_argument('ms_input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('feature_reduce_choice', type=str_or_none, nargs='?', default=None,
                        help='Choice of feature reduction method. Defaults to None.')
    parser.add_argument('transpose', type=str2bool, help='Transpose file (true/false)')
    parser.add_argument('norm', type=str2bool, help='Normalize (true/false)')
    parser.add_argument('log10', type=str2bool, help='Take the log 10 of input in the pipeline (true/false)')
    # parser.add_argument('set_seed', type=str, help='The Seed to use')
    args = parser.parse_args()

    main(args.ms_input_file, args.feature_reduce_choice, args.transpose, args.norm, args.log10)  #, args.set_seed)"""

import csv
import importlib.metadata
import sys

# Define the citation dictionary
citation_dict = {
    "json": "Built-in Python module, does not require a citation.",
    "pandas": "Reback, J., et al. (2023). pandas: powerful Python data analysis toolkit. URL: https://pandas.pydata.org/",
    "numpy": "Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357-362.",
    "joblib": "Joblib: running Python functions as pipeline jobs. Joblib documentation. URL: https://joblib.readthedocs.io/en/latest/",
    "scikit-learn": "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "tensorflow": "TensorFlow Development Team. (2015). TensorFlow: Large-scale machine learning on heterogeneous systems. URL: https://tensorflow.org",
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
        if library in ["roc_curve", "auc", "balanced_accuracy_score", "recall_score", "f1_score", "precision_score", "accuracy_score", "SVC"]:
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


# Define the main function
def main():
    # Output CSV filename
    filename = "software_citations.csv"
    save_citations_to_csv(filename)


# Standard Python entry point
if __name__ == "__main__":
    main()
