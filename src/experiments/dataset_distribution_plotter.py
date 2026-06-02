import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde

import src.experiments.datasets as datasets

DATASET_STATS_RESULTS_DIR = './results/dataset_stats/'
DATASET_DISTRIBUTION_VISUALIZATION_DIR = './visualization/dataset_distributions/'

A4_W, A4_H = 8.27, 11.69
FONT_SIZE = 9
SPINE_LW = 2
TICK_WIDTH = 2
TICK_LENGTH = 5

DATASETS_ORDER = [
    datasets.AMLSim1mInfo,
    datasets.AMLWorldInfo,
    datasets.SAMLDInfo,
]

DATASETS_DISPLAY_NAMES = ['AMLSim', 'AMLworld', 'SAML-D']

VARIABLES = [
    ('in_degree', 'Nr. credit trans.'),
    ('out_degree', 'Nr. debit trans.'),
    ('unique_in_degree', 'Nr. distinct credit counterparties'),
    ('unique_out_degree', 'Nr. distinct debit counterparties'),
    ('in_strength', 'Total credit trans. volume'),
    ('out_strength', 'Total debit trans. volume'),
]

NOT_ML_COLOR = '#2166ac'
ML_COLOR = '#d6604d'


def plot_kde_on_ax(ax, series, label, color):
    data = series.dropna()
    data = data[data > 0]
    if len(data) < 2:
        return
    log_data = np.log10(data)
    kde = gaussian_kde(log_data)
    x_log = np.linspace(log_data.min(), log_data.max(), 500)
    y = kde(x_log)
    x_orig = 10 ** x_log
    ax.plot(x_orig, y, label=label, color=color, linewidth=2)
    ax.fill_between(x_orig, y, alpha=0.25, color=color)



def build_combined_figure(all_datasets):
    plt.rc('font', weight='bold', family='Arial', size=FONT_SIZE)

    n_vars = len(VARIABLES)
    n_datasets = len(DATASETS_ORDER)

    fig, axes = plt.subplots(n_vars, n_datasets, figsize=(A4_W, A4_H))

    for col_idx, (dataset_info, display_name) in enumerate(zip(DATASETS_ORDER, DATASETS_DISPLAY_NAMES)):
        dataset_stats = all_datasets[col_idx]
        ml = dataset_stats[dataset_stats['is_ml'] == True]
        not_ml = dataset_stats[dataset_stats['is_ml'] == False]

        for row_idx, (col, label) in enumerate(VARIABLES):
            ax = axes[row_idx][col_idx]
            plot_kde_on_ax(ax, not_ml[col], 'Not ML', NOT_ML_COLOR)
            plot_kde_on_ax(ax, ml[col], 'ML', ML_COLOR)

            ax.set_xscale('log')
            ax.tick_params(axis='both', which='major', width=TICK_WIDTH, length=TICK_LENGTH)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.setp([ax.spines['left'], ax.spines['bottom']], linewidth=SPINE_LW)

            if row_idx == 0:
                ax.set_title(display_name, fontsize=FONT_SIZE + 1, fontweight='bold', family='Arial')
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=FONT_SIZE - 2, labelpad=6, fontweight='bold', family='Arial')
            if row_idx == n_vars - 1:
                ax.set_xlabel('Value (log)', fontsize=FONT_SIZE, labelpad=4, fontweight='bold', family='Arial')
            if row_idx == 0 and col_idx == n_datasets - 1:
                ax.legend(fontsize=FONT_SIZE - 2)

    fig.tight_layout()
    return fig


def main():
    os.makedirs(DATASET_DISTRIBUTION_VISUALIZATION_DIR, exist_ok=True)

    all_datasets = [
        pandas.read_csv(DATASET_STATS_RESULTS_DIR + info.name + '_stats.csv')
        for info in DATASETS_ORDER
    ]

    pdf_path = DATASET_DISTRIBUTION_VISUALIZATION_DIR + 'distributions.pdf'
    with PdfPages(pdf_path) as pdf:
        fig = build_combined_figure(all_datasets)
        pdf.savefig(fig)
        plt.close(fig)

    print(f'Saved {pdf_path}')


if __name__ == '__main__':
    main()
