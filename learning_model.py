from sklearn.inspection import permutation_importance
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.preprocessing import OneHotEncoder
from rfpimp import *
import pandas as pd


puzzle_data_path_export = ''
features_path_export = ''

def learning_method(data, puzzle_features):

    one_hot_encoder = OneHotEncoder(sparse=False)
    puzzle_encoded = pd.DataFrame(one_hot_encoder.fit_transform(data[['puzzle']]))
    puzzle_encoded.columns = 'puzzle_no_' + puzzle_encoded.columns.astype(str)
    data = data.join(puzzle_encoded, rsuffix='user_')


    X = data.drop('completed', axis=1)
    y = data['completed']
    print(f'(GLOBAL) Entradas positivas: {len(y[y == 1])}. Entradas negativas: {len(y[y==0])}')
    groups = data['group']

    recall = []
    f1 = []
    precision = []
    auc = []
    confussion_matrices = []

    recall_train = []
    f1_train = []
    precision_train = []
    auc_train = []
    leave_one_out = LeaveOneGroupOut()

    importances = []
    importances_std = []
    for i, (train_index, test_index) in enumerate(leave_one_out.split(X, y, groups)):
        group_to_exclude = X.loc[test_index]['group'].unique()[0]
        X_with_lvl_features = add_puzzle_features(X, puzzle_features, group_to_exclude)
        X_with_lvl_features.to_csv(f'C:/Users/joser/OneDrive/TFG/Data/Debug/with_level/{i}.csv')

        X_train = X_with_lvl_features.loc[train_index]
        X_test = X_with_lvl_features.loc[test_index]
        if X_test['user'].nunique() < 20:
            continue
        X_train.drop(['user', 'group', 'full_user', 'puzzle'], axis=1, inplace=True)
        X_test.drop(['user', 'group', 'full_user', 'puzzle'], axis=1, inplace=True)
        X_droped = X_with_lvl_features.drop(['user', 'group', 'full_user', 'puzzle'], axis=1)
        y_train = y.loc[train_index]
        y_test = y.loc[test_index]


        # n_estimators=200, max_depth=10, min_samples_split=4, class_weight={0: 2, 1: 1}
        rf_model = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_split=7, min_samples_leaf=4, class_weight='balanced')

        # RF Model

        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        test_data = X_with_lvl_features.loc[test_index]

        # Feature importance calculation
        result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=0)


        result = permutation_importance(
            rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
        )

        forest_importances = pd.Series(result.importances_mean, index=X_droped.columns)

        filtered_values = forest_importances[forest_importances.index.str.startswith('puzzle_no_')]
        sum_of_filtered = filtered_values.sum()
        forest_importances.drop(filtered_values.index, inplace=True)
        forest_importances['puzzle'] = sum_of_filtered


        imp_std = pd.Series(result.importances_std, index=X_droped.columns)
        filtered_values = imp_std[imp_std.index.str.startswith('puzzle_no_')]
        sum_of_filtered = filtered_values.sum()
        imp_std.drop(filtered_values.index, inplace=True)
        imp_std['puzzle'] = sum_of_filtered
        imp_std = imp_std.to_numpy()

        importances.append(forest_importances)
        importances_std.append(imp_std)


        y_train_pred = rf_model.predict(X_train)
        recall_train.append(recall_score(y_train, y_train_pred))
        f1_train.append(f1_score(y_train, y_train_pred))
        precision_train.append(precision_score(y_train, y_train_pred))
        y_train_proba = rf_model.predict_proba(X_train)[:, 1]
        auc_train.append(roc_auc_score(y_train, y_train_proba))

        print(
            f'Iteration number {i}, test group {test_data["group"].unique()[0]}, test users {test_data["user"].nunique()}, '
            f'train rows {len(X_with_lvl_features.loc[train_index])}, test rows: {len(X_with_lvl_features.loc[test_index])}')
        print(
            f'recall: {recall[-1]}, f1: {f1[-1]}, precision: {precision[-1]}')
        cm = confusion_matrix(y_test, y_pred)
        confussion_matrices.append(cm)
        print(f'(TEST) Entradas positivas: {len(y_test[y_test == 1])}. Entradas negativas: {len(y_test[y_test==0])}')
        print(f'confussion matrix: TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TP: {cm[1,1]}')

        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        print(f'Área bajo la curva ROC: {auc_test}')
        auc.append(auc_test)

        print()

    acc_mean = np.mean(recall)
    f1_mean = np.mean(f1)
    precision_mean = np.mean(precision)
    auc_test_mean = np.mean(auc)

    cm_mean = sum(confussion_matrices) / len(confussion_matrices)

    # print the overall test and train scores
    print(f"Scores for manual crossvalidation")
    print(f"Overall recall: {acc_mean}")
    print(f"Overall F1 score: {f1_mean}")
    print(f"Overall precision score: {precision_mean}")
    print(f"Overall AUC score: {auc_test_mean}")
    print(f"Overall CM: TN: {cm_mean[0,0]}, FP: {cm_mean[0,1]}, FN: {cm_mean[1,0]}, TP: {cm_mean[1,1]}')")

    print(f"Overall TRAIN recall: {np.mean(recall_train)}")
    print(f"Overall TRAIN F1 score: {np.mean(f1_train)}")
    print(f"Overall TRAIN precision score: {np.mean(precision_train)}")
    print(f"Overall AUC-ROC score: {np.mean(auc_train)}")

    importances_mean = sum(importances) / len(importances)
    importances_std_mean = sum(importances_std) / len(importances_std)
    fig, ax = plt.subplots()
    importances_mean.plot.bar(yerr=importances_std_mean, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


def add_puzzle_features(df, df_puzzles, group_to_exclude):
    copy_df = df.copy()
    # dataframe with the data of all the groups but the one to be exluded for calculations
    filtered_puzzles_df = df_puzzles[df_puzzles['group'] != group_to_exclude]
    for puzzle in copy_df['puzzle'].unique():
        specific_puzzle_rows = filtered_puzzles_df[filtered_puzzles_df['puzzle'] == puzzle]
        specific_completed_puzzle_rows = specific_puzzle_rows[specific_puzzle_rows['is_completed'] == 1]

        puzzle_avg_time = 0 if len(specific_completed_puzzle_rows) == 0 else specific_completed_puzzle_rows[
            'time'].median()
        puzzle_avg_submissions = 0 if len(specific_completed_puzzle_rows) == 0 else specific_completed_puzzle_rows[
            'n_times_submitted'].median()

        puzzle_success_rate = 0 if len(specific_puzzle_rows) == 0 else \
            len(specific_completed_puzzle_rows) / len(specific_puzzle_rows)

        # we update the main dataframe with the data collected
        copy_df.loc[copy_df['puzzle'] == puzzle, 'puzzle_avg_time'] = puzzle_avg_time
        copy_df.loc[copy_df['puzzle'] == puzzle, 'puzzle_avg_submissions'] = puzzle_avg_submissions
        copy_df.loc[copy_df['puzzle'] == puzzle, 'puzzle_success_rate'] = puzzle_success_rate

    return copy_df

data_csv = pd.read_csv(features_path_export, sep=',')
df_puzzle_data = pd.read_csv(puzzle_data_path_export)

learning_method(data_csv, df_puzzle_data)