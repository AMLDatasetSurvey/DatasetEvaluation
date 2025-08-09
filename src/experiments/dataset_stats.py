import pandas
import tqdm

import datasets

PARSED_TRANSACTIONS_FILE = 'parsed_transactions.csv'
PARSED_ACCOUNTS_FILE = 'parsed_accounts.csv'
DATASET_STATS_RESULTS_DIR = './results/dataset_stats/'

def count_in_out_stats(dataset):
    empty_dataframe = pandas.DataFrame(data=None, columns=list(set(dataset.transactions_by_origin.columns.to_list() + dataset.transactions_by_target.columns.to_list())))
    in_degree = []
    out_degree = []
    unique_in_degree = []
    unique_out_degree = []
    in_strength = []
    out_strength = []
    for _, account in tqdm.tqdm(dataset.accounts.iterrows(), total=len(dataset.accounts)):
        account_id = account['account_id']
        received = dataset.transactions_by_target.loc[[account_id]] if account_id in dataset.transactions_by_target.index else empty_dataframe
        sent = dataset.transactions_by_origin.loc[[account_id]] if account_id in dataset.transactions_by_origin.index else empty_dataframe

        in_degree.append(len(received))
        out_degree.append(len(sent))

        in_strength.append(received['amount'].sum())
        out_strength.append(sent['amount'].sum())

        unique_in_degree.append(len(set(received['origin_id'])))
        unique_out_degree.append(len(set(sent['target_id'])))

    return pandas.DataFrame({'account_id': dataset.accounts['account_id'], 'in_degree': in_degree, 'out_degree': out_degree, 'unique_in_degree': unique_in_degree, 'unique_out_degree': unique_out_degree, 'in_strength': in_strength, 'out_strength': out_strength, 'is_ml': (dataset.accounts['is_ml'].to_list() if 'is_ml' in dataset.accounts.columns else [False] * len(dataset.accounts))}).sort_values(by='is_ml')

def main():
    for dataset in datasets.get_datasets():
        print('Calculating out/in degree stats for ' + dataset.info.name)
        degree_stats = count_in_out_stats(dataset)
        degree_stats.to_csv(DATASET_STATS_RESULTS_DIR + dataset.info.name + '_stats.csv', index=False)


if __name__ == '__main__':
    main()
