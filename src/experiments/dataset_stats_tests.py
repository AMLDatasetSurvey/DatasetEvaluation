import os

import numpy as np
import pandas
from scipy.stats import ks_2samp, wasserstein_distance

from src.experiments.datasets import get_datasets_info, RaboPlusInfo

DATASET_STATS_RESULTS_DIR = './results/dataset_stats/'
TESTS_RESULTS_DIR = './results/statistical_tests/'

FEATURES = ['in_degree', 'out_degree', 'unique_in_degree', 'unique_out_degree', 'in_strength', 'out_strength']

SAMPLE_SIZE = 50_000
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def load_feature(dataset_name, feature):
    return pandas.read_csv(
        DATASET_STATS_RESULTS_DIR + dataset_name + '_stats.csv', usecols=[feature]
    )[feature].values


def log_transform(x):
    return np.log10(x + 1)


def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def bootstrap_wasserstein_ci(a, b, rng):
    distances = [
        wasserstein_distance(
            rng.choice(a, size=BOOTSTRAP_SAMPLE_SIZE, replace=True),
            rng.choice(b, size=BOOTSTRAP_SAMPLE_SIZE, replace=True),
        )
        for _ in range(BOOTSTRAP_ITERATIONS)
    ]
    return float(np.percentile(distances, 2.5)), float(np.percentile(distances, 97.5))


def run_tests(dataset_info, rabobank_features, rng):
    results = []
    for feature in FEATURES:
        if feature not in rabobank_features:
            continue

        synthetic = log_transform(load_feature(dataset_info.name, feature))
        real = rabobank_features[feature]

        synthetic_sample = rng.choice(synthetic, size=min(SAMPLE_SIZE, len(synthetic)), replace=False)
        real_sample = rng.choice(real, size=min(SAMPLE_SIZE, len(real)), replace=False)

        ks_stat, p_value = ks_2samp(synthetic_sample, real_sample)
        w_dist = wasserstein_distance(synthetic_sample, real_sample)
        ci_low, ci_high = bootstrap_wasserstein_ci(synthetic, real, rng)
        d = cohens_d(synthetic_sample, real_sample)

        row = {
            'dataset': dataset_info.name,
            'feature': feature,
            'ks_statistic': round(ks_stat, 4),
            'p_value': round(p_value, 4),
            'wasserstein_distance': round(w_dist, 4),
            'wasserstein_ci_low': round(ci_low, 4),
            'wasserstein_ci_high': round(ci_high, 4),
            'cohens_d': round(d, 4),
        }
        results.append(row)
        print(f"  {feature}: KS={ks_stat:.4f} (p={p_value:.4e}), "
              f"W={w_dist:.4f} [{ci_low:.4f}, {ci_high:.4f}], d={d:.4f}")

    return results


def main():
    os.makedirs(TESTS_RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    rabobank_raw = {
        feature: log_transform(load_feature(RaboPlusInfo.name, feature))
        for feature in FEATURES
    }

    all_results = []
    for dataset_info in get_datasets_info():
        if dataset_info.name == RaboPlusInfo.name:
            continue
        print(f'\n{dataset_info.name}')
        all_results.extend(run_tests(dataset_info, rabobank_raw, rng))

    summary = pandas.DataFrame(all_results)
    output_path = TESTS_RESULTS_DIR + 'statistical_tests.csv'
    summary.to_csv(output_path, index=False)
    print(f'\nResults saved to {output_path}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
