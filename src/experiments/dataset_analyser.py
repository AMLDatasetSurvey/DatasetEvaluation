import pandas

import datasets
from aux import uniqueness_ratio

def diversity_of_transaction_amounts_synth():
        print('synthaml')
        data = pandas.read_csv('./data/synth/raw_data/raw_transactions.csv', usecols=['Size'])
        counts = data['Size'].abs().value_counts().sort_values(ascending=False).to_numpy()

        total_frequency = 0
        for ix, count in enumerate(counts):
            total_frequency += count
            if (total_frequency / len(data)) > 0.5:
                print('50%:', ix + 1)
                break

        print('ur: ', uniqueness_ratio(data['Size'].abs()))

def diversity_of_transaction_amounts():
    for dataset_info in datasets.get_datasets_info():
        if dataset_info.name != datasets.SAMLDInfo.name:
            continue

        print(dataset_info.name)
        data = pandas.read_csv(dataset_info.data_dir + 'parsed_transactions.csv', usecols=['amount'])
        counts = data['amount'].value_counts().sort_values(ascending=False).to_numpy()

        total_frequency = 0
        for ix, count in enumerate(counts):
            total_frequency += count
            if (total_frequency / len(data)) > 0.5:
                print('50%:', ix + 1)
                break

        print('ur: ', uniqueness_ratio(data['amount']))



def basic_stats():
    for dataset_info in datasets.get_datasets_info():
        if dataset_info.name == datasets.RABOBANK_PLUS_NAME:
            continue
        print(dataset_info.name)
        if dataset_info.name == datasets.BERKA_PLUS_NAME:
            data = pandas.read_csv(dataset_info.data_dir + 'parsed_transactions.csv')
            print('nr. transactions:', len(data))
            del data
            data = pandas.read_csv(dataset_info.data_dir + 'parsed_accounts.csv')
            print('nr. accounts:', len(data))
        else:
            data = pandas.read_csv(dataset_info.data_dir + 'parsed_transactions.csv', usecols=['is_ml'])
            print('nr. transactions:', len(data))
            print('% of ml transactions', len(data[data['is_ml']])/len(data))
            del data
            data = pandas.read_csv(dataset_info.data_dir + 'parsed_accounts.csv', usecols=['is_ml'])
            print('nr. accounts:', len(data))
            print('% of ml accounts', len(data[data['is_ml']])/len(data))
        print()

if __name__ == '__main__':
    print('Calculating dataset statistics')
    basic_stats()
    print('Calculating diversity of transaction amounts')
    diversity_of_transaction_amounts()
    diversity_of_transaction_amounts_synth()