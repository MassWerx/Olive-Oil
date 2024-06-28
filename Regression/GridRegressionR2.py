import argparse
import os
import shutil
from os.path import basename
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZipFile

import numpy as np
import pandas as pd
from boruta import BorutaPy
from joblib import Memory
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.feature_selection import f_regression, SelectKBest, r_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor

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
    if flip:
        df = pd.read_csv(input_file, header=None).T
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    else:
        df = pd.read_csv(input_file, header=0)
    return df


def load_data_frame(input_dataframe):
    train_oil_file = input_dataframe.copy()
    samples = train_oil_file.iloc[:, 0]
    labels = train_oil_file.iloc[:, 1]
    data_table = train_oil_file.iloc[:, 2:]
    features = train_oil_file.iloc[:, 2:].columns
    features_names = features.to_numpy().astype(str)
    label_table = pd.get_dummies(labels)
    class_names = label_table.columns.to_numpy()

    X = data_table
    y = labels.to_numpy()

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


"""('Penalized Logistic Regression', LogisticRegression(penalty='l2', solver='liblinear', random_state=seed),
 {
     'regressor__C': [0.001, 0.01, 0.1, 1, 10, 100],
 }),"""


def get_models():
    list_of_models = [
        ('AdaBoost', AdaBoostRegressor(random_state=seed), {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.1, 1]
        }),
        ('Gradient Boosting', GradientBoostingRegressor(random_state=seed), {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 4, 5]
        }),
        ('Extreme Gradient Boosting', XGBRegressor(random_state=seed), {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 4, 5],
            'regressor__gamma': [0, 0.1, 0.2],
            'regressor__subsample': [0.6, 0.8, 1.0],
            'regressor__colsample_bytree': [0.6, 0.8, 1.0],
            'regressor__reg_alpha': [0, 0.001, 0.01, 0.1],
            'regressor__reg_lambda': [0, 0.001, 0.01, 0.1]
        }),
        ('ElasticNet', ElasticNet(max_iter=100000, random_state=seed), {
            'regressor__alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            'regressor__l1_ratio': np.arange(0.1, 1.0, 0.1),
        }),
        ('Lasso', Lasso(max_iter=100000, random_state=seed), {
            'regressor__alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
        }),
        ('Ridge Regression', Ridge(random_state=seed), {
            'regressor__alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
        }),
        ('Random Forest', RandomForestRegressor(random_state=seed), {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        })
    ]

    return list_of_models


class SpearmanReduction(BaseEstimator, TransformerMixin):
    def __init__(self, p_value_threshold=0.05):
        self.p_value_threshold = p_value_threshold
        self.selected_features_ = None

    def fit(self, X, y):
        selected_features = np.zeros(X.shape[1], dtype=bool)  # Initialize boolean array
        for i in range(X.shape[1]):
            rho, p_value = spearmanr(X[:, i], y)
            if p_value < self.p_value_threshold:
                selected_features[i] = True
        self.selected_features_ = selected_features
        return self

    def transform(self, X):
        return X[:, self.selected_features_]


class PScoreReduction(BaseEstimator, TransformerMixin):
    def __init__(self, p_value_threshold=0.05):
        self.p_value_threshold = p_value_threshold
        self.selected_features_ = None

    def fit(self, X, y):
        f_values, p_values = f_regression(X, y)
        self.selected_features_ = p_values < self.p_value_threshold
        return self

    def transform(self, X):
        return X[:, self.selected_features_]


def get_feature_reduction(feature_reduce_choice):
    reduction_fun = None
    match feature_reduce_choice:
        case None:
            print("No Feature Reduction")
            reduction_fun = 'passthrough'
        case 'Boruta':
            np.int = np.int32
            np.float = np.float64
            np.bool = np.bool_
            rf_reducer = RandomForestRegressor(n_jobs=-1, max_depth=5)
            print('Using Boruta')
            boruta = BorutaPy(rf_reducer, n_estimators='auto', verbose=0, random_state=seed)
            reduction_fun = boruta
        case 'PScore':
            print('Using P-Score Reduction')
            reduction_fun = PScoreReduction(p_value_threshold=0.05)
        case 'Spearman':
            print('Using Spearman Correlation Reduction')
            reduction_fun = SpearmanReduction(p_value_threshold=0.05)
        case _:
            print("Please select a valid Feature Reduction Method")
            quit()
    return reduction_fun


import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp
from joblib import Memory

def run_models(ms_info, list_of_models, ms_file_name, feature_reduce_choice):
    X = ms_info['X']
    y = ms_info['y']
    features = ms_info['feature_names']
    class_names = ms_info['class_names']
    class_labels = ms_info['labels'].tolist()
    seed = 123456

    # Store the full input X and y for debugging
    input_data = pd.DataFrame(X, columns=features)
    input_data['y'] = y
    input_data.to_csv(os.path.join(current_working_dir, f'input_data_{ms_file_name}.csv'), index=False)
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    outer_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=seed)

    # Save the outer fold indices for all repeats
    outer_fold_indices = [(train_idx, test_idx) for train_idx, test_idx in outer_cv.split(X, y)]

    overall_results = []

    for name, model, param_grid in list_of_models:
        print(f'Starting {name}')

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('Reduction', get_feature_reduction(feature_reduce_choice)),
            ('regressor', model)
        ], memory=memory)

        all_y_true = []
        all_y_preds = []
        all_mse = []
        all_mae = []
        all_r2_scores = []
        r2_score_repeat = []
        all_pearson_corrs = []
        all_spearman_corrs = []
        all_splits_data = []  # Store data for each split
        all_features_data = []  # Store features for each split

        for repeat in range(10):
            print(f'Repeat {repeat + 1}/10')
            r2_repeat = []
            for fold_idx in range(5):
                outer_fold = outer_fold_indices[repeat * 5 + fold_idx]
                train_idx, test_idx = outer_fold
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                inner_cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=seed)
                inner_folds = list(inner_cv.split(X_train, y_train))
                inner_fold_indices_flat = []
                for fold_inner_idx, (train_inner_idx, val_idx) in enumerate(inner_folds):
                    inner_fold_indices_flat.extend([fold_inner_idx] * len(val_idx))
                inner_fold_indices_flat = np.array(inner_fold_indices_flat)

                ps = PredefinedSplit(inner_fold_indices_flat)

                grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=ps, scoring='r2')
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                r2_score_pre_fold = r2_score(y_test, y_pred)
                all_y_true.extend(y_test)
                all_y_preds.extend(y_pred)
                all_mse.append(mean_squared_error(y_test, y_pred))
                all_mae.append(mean_absolute_error(y_test, y_pred))
                all_r2_scores.append(r2_score_pre_fold)
                r2_repeat.append(r2_score_pre_fold)
                pearson_corr = pearsonr(y_test, y_pred)[0]
                all_pearson_corrs.append(pearson_corr)
                spearman_corr = spearmanr(y_test, y_pred)[0]
                all_spearman_corrs.append(spearman_corr)

                # Store y_test and y_pred
                split_data = pd.DataFrame({
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'r2_score': r2_score_pre_fold,
                    'repeat_idx': repeat,
                    'split_idx': fold_idx
                })
                all_splits_data.append(split_data)

                # Store features used for this split
                if hasattr(best_model.named_steps['Reduction'], 'support_'):
                    selected_features = [features[i] for i in range(len(features)) if
                                         best_model.named_steps['Reduction'].support_[i]]
                elif hasattr(best_model.named_steps['Reduction'], 'selected_features_'):
                    selected_features = [features[i] for i in range(len(features)) if
                                         best_model.named_steps['Reduction'].selected_features_[i]]
                else:
                    selected_features = features  # Fallback if no feature selection

                features_data = pd.DataFrame({
                    'repeat_idx': repeat,
                    'split_idx': fold_idx,
                    'features': selected_features
                })
                all_features_data.append(features_data)
                r2_score_repeat.append(r2_repeat)



        mean_MSE = np.mean(all_mse)
        std_MSE = np.std(all_mse)
        mean_MAE = np.mean(all_mae)
        std_MAE = np.std(all_mae)
        mean_R2 = np.mean(all_r2_scores)
        std_R2 = np.std(all_r2_scores)


        # np.array(r2_score_repeat) converts this list of lists into a numpy array where each row corresponds to a repeat and each column corresponds to a fold.
        r2_score_repeat = np.array(r2_score_repeat)
        # calculates the mean across each row (repeat), resulting in mean_R2_rep, which contains the mean R^2 score for each repeat.
        mean_R2_rep = np.mean(r2_score_repeat, axis=1)
        std_R2_rep = np.std(r2_score_repeat, axis=1)

        mean_Pearson = np.mean(all_pearson_corrs)
        std_Pearson = np.std(all_pearson_corrs)
        mean_Spearman = np.mean(all_spearman_corrs)
        std_Spearman = np.std(all_spearman_corrs)

        dirpath = Path(os.path.join(current_working_dir, f'output_{name}'))
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        os.makedirs(dirpath)

        # Combine all splits data into a single DataFrame and save to CSV
        all_splits_data_df = pd.concat(all_splits_data, ignore_index=True)
        all_splits_data_df.to_csv(os.path.join(current_working_dir, f'output_{name}/splits_data_{name}.csv'), index=False)

        # Combine all features data into a single DataFrame and save to CSV
        all_features_data_df = pd.concat(all_features_data, ignore_index=True)
        all_features_data_df.to_csv(os.path.join(current_working_dir, f'output_{name}/features_data_{name}.csv'), index=False)

        with open(os.path.join(current_working_dir, f'output_{name}/metrics_{name}.txt'), 'w') as f:
            f.write(f'Mean MSE: {mean_MSE}\n')
            f.write(f'Standard deviation MSE: {std_MSE}\n')
            f.write(f'Mean MAE: {mean_MAE}\n')
            f.write(f'Standard deviation MAE: {std_MAE}\n')
            f.write(f'Mean R2: {mean_R2}\n')
            f.write(f'Standard deviation for R2: {std_R2}\n')
            f.write(f'Mean R2 per repeat: {mean_R2_rep}\n')
            f.write(f'Standard deviation for R2 per repeat: {std_R2_rep}\n')
            f.write(f'Mean Pearson Correlation: {mean_Pearson}\n')
            f.write(f'Standard deviation for Pearson Correlation: {std_Pearson}\n')
            f.write(f'Mean Spearman Correlation: {mean_Spearman}\n')
            f.write(f'Standard deviation for Spearman Correlation: {std_Spearman}\n')
            f.write(f'Best parameters: {grid_search.best_params_}\n')

        with open(os.path.join(current_working_dir, f'output_{name}/overall_{name}.txt'), 'w') as f:
            f.write(f'Overall So Far: {overall_results}\n')

        if not os.path.exists(os.path.join(current_working_dir, 'zipFiles')):
            os.makedirs(os.path.join(current_working_dir, 'zipFiles'))

        create_zip_file_output(os.path.join(current_working_dir, f'zipFiles/{name}_{ms_file_name}'), dirpath)


seed = 1234546


def main(ms_input_file, feature_reduce_choice):
    print("Starting ... ")
    ms_file_name = Path(ms_input_file).stem
    df_file = load_data_from_file(ms_input_file, False)
    ms_info = load_data_frame(df_file)
    list_of_models = get_models()
    run_models(ms_info, list_of_models, ms_file_name, feature_reduce_choice)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run regression models with feature reduction.')
    parser.add_argument('ms_input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('feature_reduce_choice', type=str, help='Choice of feature reduction method.')
    args = parser.parse_args()

    main(args.ms_input_file, args.feature_reduce_choice)
