import os

import numpy as np
import pandas
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.experiments.datasets import BerkaPlusInfo, RaboPlusInfo, get_datasets_info

DATASET_STATS_RESULTS_DIR = './results/dataset_stats/'
DIFFICULTY_RESULTS_DIR = './results/ml_difficulty/'

FEATURES = ['in_degree', 'out_degree', 'unique_in_degree', 'unique_out_degree', 'in_strength', 'out_strength']

UNLABELED_DATASETS = {RaboPlusInfo.name, BerkaPlusInfo.name}

N_FOLDS = 5
RANDOM_STATE = 42


def log_transform(x):
    return np.log10(x + 1)


def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    idx = np.argmax(f1)
    return precision[idx], recall[idx], f1[idx]


def cross_validate(X, y):
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos

    clf = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric='aucpr',
        verbosity=0,
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = {'auc': [], 'avg_precision': [], 'precision': [], 'recall': [], 'f1': []}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        p, r, f1 = best_f1_threshold(y_test, y_prob)
        fold_scores['auc'].append(roc_auc_score(y_test, y_prob))
        fold_scores['avg_precision'].append(average_precision_score(y_test, y_prob))
        fold_scores['precision'].append(p)
        fold_scores['recall'].append(r)
        fold_scores['f1'].append(f1)

    return {k: round(float(np.mean(v)), 4) for k, v in fold_scores.items()}


def main():
    os.makedirs(DIFFICULTY_RESULTS_DIR, exist_ok=True)

    rows = []
    for dataset_info in get_datasets_info():
        if dataset_info.name in UNLABELED_DATASETS:
            continue

        df = pandas.read_csv(DATASET_STATS_RESULTS_DIR + dataset_info.name + '_stats.csv')
        y = df['is_ml'].values.astype(bool)
        X = log_transform(df[FEATURES].values)
        ml_prevalence = y.mean()

        print(f'\n{dataset_info.name}  (ML prevalence: {ml_prevalence:.2%})')
        scores = cross_validate(X, y)
        rows.append({'dataset': dataset_info.name, 'ml_prevalence': round(ml_prevalence, 4), **scores})
        print(f'  AUC={scores["auc"]:.4f}  AP={scores["avg_precision"]:.4f}'
              f'  P={scores["precision"]:.4f}  R={scores["recall"]:.4f}  F1={scores["f1"]:.4f}')

    results = pandas.DataFrame(rows)
    output_path = DIFFICULTY_RESULTS_DIR + 'ml_difficulty.csv'
    results.to_csv(output_path, index=False)
    print(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    main()
