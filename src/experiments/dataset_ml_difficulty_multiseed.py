import json
import os

import numpy as np
import pandas
from scipy.stats import t
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.experiments.datasets import get_datasets_info
from src.experiments.dataset_ml_difficulty import (
    DATASET_LABELS,
    DATASET_STATS_RESULTS_DIR,
    FEATURES,
    N_FOLDS,
    UNLABELED_DATASETS,
    VAL_SIZE,
    best_f1_threshold,
    format_ml_percent,
    log_transform,
)

MULTISEED_RESULTS_DIR = './results/ml_difficulty_multiseed/'
SEEDS = list(range(10))
CONFIDENCE = 0.95


def cross_validate_seed(X, y, seed):
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos

    clf = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        n_jobs=1,
        random_state=seed,
        eval_metric='aucpr',
        verbosity=0,
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    aucs, aps, f1s = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train,
            test_size=VAL_SIZE, stratify=y_train, random_state=seed,
        )

        clf.fit(X_fit, y_fit)

        threshold = best_f1_threshold(y_val, clf.predict_proba(X_val)[:, 1])
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= threshold).astype(int)

        aucs.append(float(roc_auc_score(y_test, y_prob_test)))
        aps.append(float(average_precision_score(y_test, y_prob_test)))
        f1s.append(float(f1_score(y_test, y_pred_test)))

    return {
        'auc': float(np.mean(aucs)),
        'avg_precision': float(np.mean(aps)),
        'f1': float(np.mean(f1s)),
    }


def ci_halfwidth(values, confidence=CONFIDENCE):
    n = len(values)
    if n < 2:
        return 0.0
    sem = float(np.std(values, ddof=1)) / np.sqrt(n)
    return float(t.ppf(0.5 + confidence / 2, df=n - 1)) * sem


def summarise(values):
    arr = np.asarray(values, dtype=float)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'ci': ci_halfwidth(arr),
        'lo': float(arr.mean() - ci_halfwidth(arr)),
        'hi': float(arr.mean() + ci_halfwidth(arr)),
    }


def write_main_latex(rows, path, confidence=CONFIDENCE):
    pct = int(round(confidence * 100))
    lines = [
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        rf'\textbf{{Dataset}} & \textbf{{\%ML Accounts}} & \textbf{{AUC}} & \textbf{{AP}} & \textbf{{F1}} \\',
        r'\midrule',
    ]
    for r in rows:
        label = DATASET_LABELS.get(r['dataset'], r['dataset'])
        lines.append(
            f'{label} & {format_ml_percent(r["ml_prevalence"])} & '
            f'${r["auc_mean"]:.3f} \\pm {r["auc_ci"]:.4f}$ & '
            f'${r["ap_mean"]:.3f} \\pm {r["ap_ci"]:.4f}$ & '
            f'${r["f1_mean"]:.3f} \\pm {r["f1_ci"]:.4f}$ \\\\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        '',
        f'% mean $\\pm$ {pct}% CI half-width over {len(SEEDS)} seeds (t-interval, df={len(SEEDS) - 1}); each run is {N_FOLDS}-fold stratified CV.',
        '',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    os.makedirs(MULTISEED_RESULTS_DIR, exist_ok=True)

    rows = []
    per_seed_rows = []
    for dataset_info in get_datasets_info():
        if dataset_info.name in UNLABELED_DATASETS:
            continue

        df = pandas.read_csv(DATASET_STATS_RESULTS_DIR + dataset_info.name + '_stats.csv')
        y = df['is_ml'].values.astype(bool)
        X = log_transform(df[FEATURES].values)
        ml_prevalence = float(y.mean())

        print(f'\n{dataset_info.name}  (ML prevalence: {ml_prevalence:.2%})')

        aucs, aps, f1s = [], [], []
        for seed in SEEDS:
            scores = cross_validate_seed(X, y, seed)
            aucs.append(scores['auc'])
            aps.append(scores['avg_precision'])
            f1s.append(scores['f1'])
            per_seed_rows.append({
                'dataset': dataset_info.name,
                'seed': seed,
                'auc': round(scores['auc'], 4),
                'avg_precision': round(scores['avg_precision'], 4),
                'f1': round(scores['f1'], 4),
            })
            print(f'  seed {seed:>3}: AUC={scores["auc"]:.4f}  AP={scores["avg_precision"]:.4f}  F1={scores["f1"]:.4f}')

        auc_s = summarise(aucs)
        ap_s = summarise(aps)
        f1_s = summarise(f1s)
        rows.append({
            'dataset': dataset_info.name,
            'ml_prevalence': round(ml_prevalence, 4),
            'n_seeds': len(SEEDS),
            'auc_mean': round(auc_s['mean'], 4),
            'auc_std': round(auc_s['std'], 4),
            'auc_ci': round(auc_s['ci'], 4),
            'auc_lo': round(auc_s['lo'], 4),
            'auc_hi': round(auc_s['hi'], 4),
            'ap_mean': round(ap_s['mean'], 4),
            'ap_std': round(ap_s['std'], 4),
            'ap_ci': round(ap_s['ci'], 4),
            'ap_lo': round(ap_s['lo'], 4),
            'ap_hi': round(ap_s['hi'], 4),
            'f1_mean': round(f1_s['mean'], 4),
            'f1_std': round(f1_s['std'], 4),
            'f1_ci': round(f1_s['ci'], 4),
            'f1_lo': round(f1_s['lo'], 4),
            'f1_hi': round(f1_s['hi'], 4),
        })
        print(f'  AUC={auc_s["mean"]:.4f} ± {auc_s["ci"]:.4f}  '
              f'AP={ap_s["mean"]:.4f} ± {ap_s["ci"]:.4f}  '
              f'F1={f1_s["mean"]:.4f} ± {f1_s["ci"]:.4f}')

    summary_df = pandas.DataFrame(rows)
    summary_path = MULTISEED_RESULTS_DIR + 'ml_difficulty_multiseed.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'\nSummary saved to {summary_path}')

    per_seed_df = pandas.DataFrame(per_seed_rows)
    per_seed_path = MULTISEED_RESULTS_DIR + 'ml_difficulty_multiseed_per_seed.csv'
    per_seed_df.to_csv(per_seed_path, index=False)
    print(f'Per-seed results saved to {per_seed_path}')

    main_tex_path = MULTISEED_RESULTS_DIR + 'ml_difficulty_multiseed.tex'
    write_main_latex(rows, main_tex_path)
    print(f'Main LaTeX table saved to {main_tex_path}')

    config = {
        'seeds': SEEDS,
        'n_folds': N_FOLDS,
        'val_size': VAL_SIZE,
        'features': FEATURES,
        'confidence': CONFIDENCE,
    }
    config_path = MULTISEED_RESULTS_DIR + 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Config saved to {config_path}')


if __name__ == '__main__':
    main()
