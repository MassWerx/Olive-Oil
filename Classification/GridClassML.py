import pandas as pd
import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
import os
from pathlib import Path
from tempfile import mkdtemp
import argparse
import random
import shutil
import plotly.express as px
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, recall_score, f1_score, precision_score
from sklearn.svm import SVC
import tensorflow as tf
from joblib import Memory

seed = 123456


def get_feature_reduction(feature_reduce_choice):
    reduction_fun = None
    if feature_reduce_choice == 'Boruta':
        rf_reducer = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_
        boruta = BorutaPy(rf_reducer, n_estimators='auto', verbose=2, random_state=seed)
        reduction_fun = boruta
        print('Using Boruta')
    else:
        print("Please select a valid Feature Reduction Method or None")
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
        'X': X.astype(float),
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
        ('GradientBoosting', GradientBoostingClassifier(random_state=seed), {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
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
        ('SVM', SVC(probability=True, random_state=seed), {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }),
        ('AdaBoost', AdaBoostClassifier(random_state=seed), {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 1]
        }),
        ('RandomForest', RandomForestClassifier(n_jobs=-1, random_state=seed), {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }),
        ('TensorFlow', MyKerasClf(n_classes=len(np.unique(y)), seed=1), {
            'classifier__learn_rate': [0.001, 0.01],
            'classifier__weight_constraint': [0, 1]
        })
    ]
    return list_of_models


def run_models(ms_info, list_of_models, ms_file_name, feature_reduce_choice):
    X = ms_info['X']
    y = ms_info['y']
    current_working_dir = os.getcwd()
    output_dir = os.path.join(current_working_dir, 'output')
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)

    # Save the outer fold indices
    outer_fold_indices = [(train_idx, test_idx) for train_idx, test_idx in outer_cv.split(X, y)]

    # Save the inner fold indices for the first outer fold (for simplicity)
    first_outer_train_idx, first_outer_test_idx = outer_fold_indices[0]
    inner_folds = list(inner_cv.split(X.iloc[first_outer_train_idx], y[first_outer_train_idx]))

    inner_fold_indices_flat = []
    for fold_idx, (train_idx, val_idx) in enumerate(inner_folds):
        inner_fold_indices_flat.extend([fold_idx] * len(val_idx))
    inner_fold_indices_flat = np.array(inner_fold_indices_flat)

    ps = PredefinedSplit(inner_fold_indices_flat)

    # Save the fold indices
    outer_fold_indices = [(train_idx, test_idx) for train_idx, test_idx in outer_cv.split(X, y)]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for name, model, param_grid in list_of_models:
        print(f'Starting {name}')

        pipeline = Pipeline([
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

        for fold_idx, (train_idx, test_idx) in enumerate(outer_fold_indices):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=ps, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_score = best_model.predict_proba(X_test)[:, 1]

            all_y_true.extend(y_test)
            all_y_scores.extend(y_score)
            all_balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
            all_recalls.append(recall_score(y_test, y_pred))
            all_f1_scores.append(f1_score(y_test, y_pred))
            all_precisions.append(precision_score(y_test, y_pred))

            # Save features for the current fold
            # features_for_fold = best_model.named_steps['Reduction'].transform(X_test)
            # save_features_for_fold(features_for_fold, fold_idx, output_dir)

        # Calculate mean and standard deviation for metrics
        mean_balanced_accuracy = np.mean(all_balanced_accuracies)
        std_balanced_accuracy = np.std(all_balanced_accuracies)
        mean_recall = np.mean(all_recalls)
        std_recall = np.std(all_recalls)
        mean_f1 = np.mean(all_f1_scores)
        std_f1 = np.std(all_f1_scores)
        mean_precision = np.mean(all_precisions)
        std_precision = np.std(all_precisions)

        fpr, tpr, thresholds = roc_curve(all_y_true, all_y_scores)
        roc_auc = auc(fpr, tpr)

        # Save metrics and ROC data
        model_output_dir = os.path.join(output_dir, name)
        os.makedirs(model_output_dir, exist_ok=True)

        roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        roc_data.to_csv(os.path.join(model_output_dir, f'roc_data_{name}.csv'), index=False)

        with open(os.path.join(model_output_dir, f'metrics_{name}.txt'), 'w') as f:
            f.write(f'Mean balanced accuracy: {mean_balanced_accuracy}\n')
            f.write(f'Standard deviation of balanced accuracy: {std_balanced_accuracy}\n')
            f.write(f'Mean recall: {mean_recall}\n')
            f.write(f'Standard deviation of recall: {std_recall}\n')
            f.write(f'Mean F1 score: {mean_f1}\n')
            f.write(f'Standard deviation of F1 score: {std_f1}\n')
            f.write(f'Mean precision: {mean_precision}\n')
            f.write(f'Standard deviation of precision: {std_precision}\n')
            f.write(f'AUC: {roc_auc}\n')
            f.write(f'Best parameters: {grid_search.best_params_}\n')

        # ROC plot
        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={roc_auc:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        fig.write_image(os.path.join(model_output_dir, f'{name}_ROC.png'))
        fig.show()


def main(ms_input_file, feature_reduce_choice):
    print("Starting ... ")
    ms_file_name = Path(ms_input_file).stem
    df_file = load_data_from_file(ms_input_file, False)
    ms_info = load_data_frame(df_file)
    list_of_models = get_models(ms_info['y'])
    run_models(ms_info, list_of_models, ms_file_name, feature_reduce_choice)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run regression models with feature reduction.')
    parser.add_argument('ms_input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('feature_reduce_choice', type=str, help='Choice of feature reduction method.')
    args = parser.parse_args()

    main(args.ms_input_file, args.feature_reduce_choice)
