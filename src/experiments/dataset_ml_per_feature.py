import os

import numpy as np
import pandas
from src.experiments.datasets import BerkaPlusInfo, RaboPlusInfo, get_datasets_info

DATASET_STATS_RESULTS_DIR = './results/dataset_stats/'
RESULTS_DIR = './results/ml_per_feature/'

FEATURES = ['in_degree', 'out_degree', 'unique_in_degree', 'unique_out_degree', 'in_strength', 'out_strength']

UNLABELED_DATASETS = {RaboPlusInfo.name, BerkaPlusInfo.name}


def log_transform(x):
    return np.log10(x + 1)


def fdr(x, y):
    ml, normal = x[y], x[~y]
    return (ml.mean() - normal.mean()) ** 2 / (ml.var() + normal.var())


def multivariate_fdr(X, y):
    ml, normal = X[y], X[~y]
    diff = ml.mean(axis=0) - normal.mean(axis=0)
    S_w = (np.cov(ml.T) * (len(ml) - 1) + np.cov(normal.T) * (len(normal) - 1)) / (len(X) - 2)
    return float(diff @ np.linalg.inv(S_w) @ diff)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for dataset_info in get_datasets_info():
        if dataset_info.name in UNLABELED_DATASETS:
            continue

        df = pandas.read_csv(DATASET_STATS_RESULTS_DIR + dataset_info.name + '_stats.csv')
        y = df['is_ml'].values.astype(bool)
        X = log_transform(df[FEATURES].values)
        ml_prevalence = y.mean()

        print(f'\n{dataset_info.name}  (ML prevalence: {ml_prevalence:.2%})')

        for i, feature in enumerate(FEATURES):
            score = fdr(X[:, i], y)
            rows.append({'dataset': dataset_info.name, 'feature': feature, 'fdr': round(score, 4)})
            print(f'  {feature}: FDR={score:.4f}')

        combined = multivariate_fdr(X, y)
        rows.append({'dataset': dataset_info.name, 'feature': 'combined', 'fdr': round(combined, 4)})
        print(f'  combined: FDR={combined:.4f}')

    results = pandas.DataFrame(rows)
    output_path = RESULTS_DIR + 'ml_per_feature.csv'
    results.to_csv(output_path, index=False)
    print(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    main()
