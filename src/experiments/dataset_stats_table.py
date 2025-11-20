import pandas
pandas.options.display.float_format = '{:.2f}'.format

from src.experiments.dataset_stats import DATASET_STATS_RESULTS_DIR
from src.experiments.datasets import get_datasets_info

def main():
    name = []
    features = []
    max_list = []
    min_list = []
    q25 = []
    q75 = []
    mean_list = []
    std_list = []
    for dataset_info in get_datasets_info():
        dataset_stats = pandas.read_csv(DATASET_STATS_RESULTS_DIR + dataset_info.name + '_stats.csv').sample(frac=1)


        dataset_stats.drop(columns=["account_id", 'is_ml'], inplace=True)
        name += [dataset_info.name] * len(dataset_stats.columns)
        features += list(dataset_stats.columns)
        max_list += list(dataset_stats.max())
        min_list += list(dataset_stats.min())
        q25 += list(dataset_stats.quantile(0.25))
        mean_list += list(dataset_stats.median())
        q75 += list(dataset_stats.quantile(0.75))
        std_list += list(dataset_stats.std())

    summary = pandas.DataFrame({
        'name': name,
        'features': features,
        'max': max_list,
        'min': min_list,
        'q25': q25,
        'median': mean_list,
        'q75': q75,
        'std': std_list
    })
    print(summary.round(3).to_string())

if __name__ == '__main__':
    main()