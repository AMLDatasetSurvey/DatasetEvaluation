from src.experiments import load_datasets, dataset_stats, dataset_timestamp_ploter, dataset_stats_ploter, dataset_stats_table

if __name__ == '__main__':
    load_datasets.main()
    dataset_stats.main()
    dataset_stats_ploter.main()
    dataset_timestamp_ploter.main()
    dataset_stats_table.main()
