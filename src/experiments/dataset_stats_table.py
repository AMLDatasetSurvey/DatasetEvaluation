import pandas

from src.experiments.dataset_stats import DATASET_STATS_RESULTS_DIR
from src.experiments.datasets import (
    AML_SIM_1M_NAME, AMLWORLD_NAME, BERKA_PLUS_NAME, RABOBANK_PLUS_NAME, SAMLD_NAME,
)

DATASET_DISPLAY_ORDER = [
    (BERKA_PLUS_NAME, 'Berka'),
    (AML_SIM_1M_NAME, 'AMLSim'),
    (AMLWORLD_NAME, 'AMLworld'),
    (SAMLD_NAME, 'SAML-D'),
    (RABOBANK_PLUS_NAME, 'Rabobank'),
]

FEATURE_ORDER = [
    ('in_degree', 'Nr. credit trans. per account'),
    ('out_degree', 'Nr. debit trans. per account'),
    ('unique_in_degree', 'Nr. distinct credit counterparties per account'),
    ('unique_out_degree', 'Nr. distinct debit counterparties per account'),
    ('in_strength', 'Total credit trans. volume per account'),
    ('out_strength', 'Total debit trans. volume per account'),
]


def fmt(v):
    if pandas.isna(v):
        return r'\np{nan}'
    if abs(v) >= 1000:
        return r'\np{' + str(int(round(v / 1000))) + '}K'
    if float(v).is_integer():
        return r'\np{' + str(int(v)) + '}'
    return r'\np{' + f'{v:.2f}' + '}'


def main():
    lines = [
        r'\begin{tabular}{l|lrrrrrr}',
        r'\hline',
        r'\textbf{Data set} & \textbf{Property} & \textbf{Max} & \textbf{Min} & \textbf{Q25} & \textbf{Median} & \textbf{Q75} & \textbf{Std.} \\',
        r'\hline',
    ]

    for internal_name, display_name in DATASET_DISPLAY_ORDER:
        dataset_stats = pandas.read_csv(DATASET_STATS_RESULTS_DIR + internal_name + '_stats.csv')
        lines.append(r'\multirow{' + str(len(FEATURE_ORDER)) + '}{*}{' + display_name + '}')
        for feature, feature_label in FEATURE_ORDER:
            col = dataset_stats[feature]
            cells = [
                fmt(col.max()),
                fmt(col.min()),
                fmt(col.quantile(0.25)),
                fmt(col.median()),
                fmt(col.quantile(0.75)),
                fmt(col.std()),
            ]
            lines.append('& ' + feature_label + ' & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\hline')

    lines.append(r'\end{tabular}')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
