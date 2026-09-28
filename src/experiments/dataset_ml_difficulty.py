import os

import numpy as np
import pandas
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.experiments.datasets import BerkaPlusInfo, RaboPlusInfo, get_datasets_info

DATASET_STATS_RESULTS_DIR = './results/dataset_stats/'
DIFFICULTY_RESULTS_DIR = './results/ml_difficulty/'

FEATURES = ['in_degree', 'out_degree', 'unique_in_degree', 'unique_out_degree', 'in_strength', 'out_strength']

UNLABELED_DATASETS = {RaboPlusInfo.name, BerkaPlusInfo.name}

N_FOLDS = 5
RANDOM_STATE = 42

DATASET_LABELS = {
    'amlsim_1m': 'AMLSim',
    'amlworld': 'AMLworld',
    'samld': 'SAML-D',
}


def log_transform(x):
    return np.log10(x + 1)


def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    idx = np.argmax(f1[:-1])  # last point has no corresponding threshold
    return thresholds[idx]


def cross_validate(X, y):
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos

    clf = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        n_jobs=1,
        random_state=RANDOM_STATE,
        eval_metric='aucpr',
        verbosity=0,
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)

        threshold = best_f1_threshold(y_train, clf.predict_proba(X_train)[:, 1])
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()
        fold_rows.append({
            'fold': fold_idx,
            'auc': round(float(roc_auc_score(y_test, y_prob_test)), 4),
            'avg_precision': round(float(average_precision_score(y_test, y_prob_test)), 4),
            'precision': round(float(precision_score(y_test, y_pred_test)), 4),
            'recall': round(float(recall_score(y_test, y_pred_test)), 4),
            'f1': round(float(f1_score(y_test, y_pred_test)), 4),
            'threshold': round(float(threshold), 6),
            'train_pos': int(y_train.sum()),
            'train_neg': int(len(y_train) - y_train.sum()),
            'test_pos': int(y_test.sum()),
            'test_neg': int(len(y_test) - y_test.sum()),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        })

    metric_keys = ['auc', 'avg_precision', 'precision', 'recall', 'f1']
    means = {k: round(float(np.mean([r[k] for r in fold_rows])), 4) for k in metric_keys}
    stds = {f'{k}_std': round(float(np.std([r[k] for r in fold_rows])), 4) for k in metric_keys}
    return {**means, **stds}, fold_rows


def format_ml_percent(x):
    pct = x * 100
    return f'{pct:.1f}\\%' if pct < 1 else f'{pct:.0f}\\%'


def write_main_latex(rows, path):
    lines = [
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'\textbf{Dataset} & \textbf{\%ML Accounts} & \textbf{AUC} & \textbf{AP} & \textbf{F1} \\',
        r'\midrule',
    ]
    for r in rows:
        label = DATASET_LABELS.get(r['dataset'], r['dataset'])
        lines.append(
            f'{label} & {format_ml_percent(r["ml_prevalence"])} & '
            f'${r["auc"]:.3f} \\pm {r["auc_std"]:.4f}$ & '
            f'${r["avg_precision"]:.3f} \\pm {r["avg_precision_std"]:.4f}$ & '
            f'${r["f1"]:.3f} \\pm {r["f1_std"]:.4f}$ \\\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def write_folds_latex(fold_rows, path):
    header = (
        r'\textbf{Dataset} & \textbf{Fold} & \textbf{AUC} & \textbf{AP} & '
        r'\textbf{P} & \textbf{R} & \textbf{F1} & \textbf{Thr} & '
        r'\textbf{Train+} & \textbf{Train-} & \textbf{Test+} & \textbf{Test-} & '
        r'\textbf{TP} & \textbf{TN} & \textbf{FP} & \textbf{FN} \\'
    )
    lines = [
        r'\begin{tabular}{llcccccccrrrrrrrr}',
        r'\toprule',
        header,
        r'\midrule',
    ]
    prev_dataset = None
    for r in fold_rows:
        if prev_dataset is not None and prev_dataset != r['dataset']:
            lines.append(r'\midrule')
        label = DATASET_LABELS.get(r['dataset'], r['dataset'])
        lines.append(
            f'{label} & {r["fold"]} & '
            f'${r["auc"]:.3f}$ & ${r["avg_precision"]:.3f}$ & '
            f'${r["precision"]:.3f}$ & ${r["recall"]:.3f}$ & ${r["f1"]:.3f}$ & '
            f'${r["threshold"]:.3f}$ & '
            f'{r["train_pos"]} & {r["train_neg"]} & {r["test_pos"]} & {r["test_neg"]} & '
            f'{r["tp"]} & {r["tn"]} & {r["fp"]} & {r["fn"]} \\\\'
        )
        prev_dataset = r['dataset']
    lines += [r'\bottomrule', r'\end{tabular}', '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    os.makedirs(DIFFICULTY_RESULTS_DIR, exist_ok=True)

    rows = []
    fold_rows_all = []
    for dataset_info in get_datasets_info():
        if dataset_info.name in UNLABELED_DATASETS:
            continue

        df = pandas.read_csv(DATASET_STATS_RESULTS_DIR + dataset_info.name + '_stats.csv')
        y = df['is_ml'].values.astype(bool)
        X = log_transform(df[FEATURES].values)
        ml_prevalence = y.mean()

        print(f'\n{dataset_info.name}  (ML prevalence: {ml_prevalence:.2%})')
        scores, fold_rows = cross_validate(X, y)
        rows.append({'dataset': dataset_info.name, 'ml_prevalence': round(ml_prevalence, 4), **scores})
        for r in fold_rows:
            fold_rows_all.append({'dataset': dataset_info.name, **r})
        print(f'  AUC={scores["auc"]:.4f}±{scores["auc_std"]:.4f}  AP={scores["avg_precision"]:.4f}±{scores["avg_precision_std"]:.4f}'
              f'  P={scores["precision"]:.4f}±{scores["precision_std"]:.4f}  R={scores["recall"]:.4f}±{scores["recall_std"]:.4f}  F1={scores["f1"]:.4f}±{scores["f1_std"]:.4f}')
        for r in fold_rows:
            print(f'  fold {r["fold"]}: AUC={r["auc"]:.4f}  AP={r["avg_precision"]:.4f}  F1={r["f1"]:.4f}  thr={r["threshold"]:.4f}'
                  f'  train(+/-)={r["train_pos"]}/{r["train_neg"]}  test(+/-)={r["test_pos"]}/{r["test_neg"]}'
                  f'  TP={r["tp"]} TN={r["tn"]} FP={r["fp"]} FN={r["fn"]}')

    results = pandas.DataFrame(rows)
    output_path = DIFFICULTY_RESULTS_DIR + 'ml_difficulty.csv'
    results.to_csv(output_path, index=False)
    print(f'\nResults saved to {output_path}')

    folds_df = pandas.DataFrame(fold_rows_all)
    folds_path = DIFFICULTY_RESULTS_DIR + 'ml_difficulty_folds.csv'
    folds_df.to_csv(folds_path, index=False)
    print(f'Per-fold results saved to {folds_path}')

    main_tex_path = DIFFICULTY_RESULTS_DIR + 'ml_difficulty.tex'
    write_main_latex(rows, main_tex_path)
    print(f'Main LaTeX table saved to {main_tex_path}')

    folds_tex_path = DIFFICULTY_RESULTS_DIR + 'ml_difficulty_folds.tex'
    write_folds_latex(fold_rows_all, folds_tex_path)
    print(f'Per-fold LaTeX table saved to {folds_tex_path}')


if __name__ == '__main__':
    main()
