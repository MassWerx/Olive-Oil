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
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, recall_score, f1_score, precision_score, \
    accuracy_score
from sklearn.svm import SVC

from joblib import Memory
from xgboost import XGBRegressor

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
    if feature_reduce_choice == 'Boruta':
        rf_reducer = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_
        boruta = BorutaPy(rf_reducer, n_estimators='auto', verbose=0, random_state=seed)
        reduction_fun = boruta
        print('Using Boruta')
    """else:
        print("Please select a valid Feature Reduction Method or None")
        quit()"""
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
    features = input_dataframe.iloc[:, 2:].columns
    features_names = features.to_numpy().astype(str)

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
            'classifier__n_estimators': [100, 200, 300],
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
        ('AdaBoost', AdaBoostClassifier(random_state=seed), {
            'classifier__n_estimators': [50],
            'classifier__learning_rate': [0.01]
        })
    ]
    return list_of_models


def safe_log10(num):
    return np.log10(np.clip(num, a_min=1e-9, a_max=None))  # Clip values to avoid log10(0)


def run_models_cv(ms_info, list_of_models, ms_file_name, feature_reduce_choice, normalize_select, log10_select):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
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


def run_models_cv_avg_sd(ms_info, list_of_models, ms_file_name, feature_reduce_choice, normalize_select, log10_select,
                  n_splits=5, n_repeats=10, round_scores=True):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
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
            ('transform', log10_pipe),
            ('scaler', StandardScaler()),
            ('Reduction', get_feature_reduction(feature_reduce_choice)),
            ('classifier', model)
        ], memory=memory)

        # Perform grid search using the entire data
        grid_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=grid_cv, scoring='accuracy')
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


def run_models_cv_score(ms_info, list_of_models, ms_file_name, feature_reduce_choice, normalize_select, log10_select):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
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
    features = ms_info['feature_names']
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
    features = ms_info['feature_names']
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

                # Store features used for this split
                if hasattr(best_model.named_steps['Reduction'], 'support_'):
                    selected_features = [features[i] for i in range(len(features)) if
                                         best_model.named_steps['Reduction'].support_[i]]
                elif hasattr(best_model.named_steps['Reduction'], 'selected_features_'):
                    selected_features = [features[i] for i in range(len(features)) if
                                         best_model.named_steps['Reduction'].selected_features_[i]]
                else:
                    selected_features = features  # Fallback if no feature selection

                all_features_data.append(selected_features)

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

"""
              ms_input_file,                                       feature_reduce_choice,  transpose, norm, log10
python ../../GridClassFinal.py Adult_CAN-MALDI_TAG_unnorm_29Aug2024.csv none false true false # log10 was true
python ../../GridClassFinal.py DART-PP-unnorm-filter_1Mar23.csv                            Boruta true true false
python ../../GridClassFinal.py Grade_DART-PP_filter_unnorm_7Mar23.csv                      Boruta true true false
"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run regression models with feature reduction.')
    parser.add_argument('ms_input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('feature_reduce_choice', type=str, help='Choice of feature reduction method.')
    parser.add_argument('transpose', type=str2bool, help='Transpose file (true/false)')
    parser.add_argument('norm', type=str2bool, help='Normalize (true/false)')
    parser.add_argument('log10', type=str2bool, help='Take the log 10 of input in the pipeline (true/false)')
    # parser.add_argument('set_seed', type=str, help='The Seed to use')
    args = parser.parse_args()

    main(args.ms_input_file, args.feature_reduce_choice, args.transpose, args.norm, args.log10)  #, args.set_seed)
