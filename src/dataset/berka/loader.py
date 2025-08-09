import subprocess
from src.dataset import parsing_tools
from src.dataset.berka import translator
from src.dataset.parsing_tools import ColumnNameGenerator
import os
import pandas
from tqdm import tqdm

DATASET_DIRECTORY = 'berka/'

DATA_DIRECTORY = os.getcwd() + '/data/'
RAW_DATA_DIRECTORY = DATA_DIRECTORY + DATASET_DIRECTORY + 'raw_data/'
PARSED_DATA_DIRECTORY = DATA_DIRECTORY + DATASET_DIRECTORY + 'parsed_data/'

ACCOUNTS_FILE = 'parsed_accounts.csv'
TRANSACTIONS_FILE = 'parsed_transactions.csv'

NAMES_FOR_DATA = [
    ColumnNameGenerator('account_id'),
    ColumnNameGenerator('client_id'),
    ColumnNameGenerator('disp_id'),
    ColumnNameGenerator('trans_id'),
    ColumnNameGenerator('origin_id'),
    ColumnNameGenerator('target_id'),
    ColumnNameGenerator('date')
]

def parse_account_ids_column(dataset):
    dataset['account_id'] = dataset['account_id'].apply(lambda account_id: f'a{account_id}')

def parse_account_column(dataset):
    dataset['account'] = dataset['account'].apply(lambda account_id: f'a{account_id}')

def parse_transaction_ids_column(dataset):
    dataset['trans_id'] = dataset['trans_id'].apply(lambda relation_id: f't{relation_id}')

def parse_account_data():
    internal_accounts = set(pandas.read_csv(RAW_DATA_DIRECTORY + 'raw_accounts.csv', sep=';', usecols=['account_id'])['account_id'])
    transactions = pandas.read_csv(RAW_DATA_DIRECTORY + 'raw_transactions.csv', sep=';', usecols=['account'])
    external_accounts = set(transactions[~transactions['account'].isna() & (transactions['account'] != 0)]['account'].astype(int))
    accounts = pandas.DataFrame({'account_id': list(internal_accounts | external_accounts)})
    parse_account_ids_column(accounts)
    return accounts

def parse_transaction_record(transaction):
    accounts = (transaction['account_id'], transaction['account']) if transaction['operation'] == 'transaction_sent' else (transaction['account'], transaction['account_id'])
    column_names = ['trans_id', 'origin_id', 'target_id', 'timestamp', 'amount']
    data = [transaction['trans_id'], *accounts, transaction['date'], transaction['amount']]
    return pandas.Series(dict(zip(column_names, data)))

def parse_transaction_data():
    dataset = pandas.read_csv(RAW_DATA_DIRECTORY + 'raw_transactions.csv', sep=';',
                              usecols=['trans_id', 'account_id', 'date', 'operation', 'amount', 'account'], dtype={'account': 'Int64'})
    dataset = dataset.drop(304834) #incomplete data (transaction without receiver id)
    translator.translate_transaction_data(dataset)
    dataset = dataset[(dataset['operation'] == 'transaction_sent') | (dataset['operation'] == 'transaction_received')]
    parse_account_column(dataset)
    parse_account_ids_column(dataset)
    parse_transaction_ids_column(dataset)
    dataset = dataset.progress_apply(parse_transaction_record, axis=1)
    return dataset


def save_to_csv(parsed_accounts, parsed_transactions):
    account_data_names = {x for x in NAMES_FOR_DATA if x.old_name in parsed_accounts.columns}
    transactions_data_names = {x for x in NAMES_FOR_DATA if x.old_name in parsed_transactions.columns}

    parsed_accounts. \
        rename(columns=parsing_tools.get_csv_name_map(account_data_names)). \
        to_csv(PARSED_DATA_DIRECTORY + ACCOUNTS_FILE, index=False)
    parsed_transactions. \
        rename(columns=parsing_tools.get_csv_name_map(transactions_data_names)). \
        to_csv(PARSED_DATA_DIRECTORY + TRANSACTIONS_FILE, index=False)

def main():
    tqdm.pandas()
    print('Parsing account data\n')
    parsed_accounts = parse_account_data()
    print('Parsing transaction data\n')
    parsed_transactions = parse_transaction_data()
    print('Saving data to CSV\n')
    save_to_csv(
        parsed_accounts=parsed_accounts,
        parsed_transactions=parsed_transactions,
    )

if __name__ == '__main__':
    main()
