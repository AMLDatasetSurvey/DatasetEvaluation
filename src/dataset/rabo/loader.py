import os
import subprocess

import pandas

from src.dataset.parsing_tools import ColumnNameGenerator
from src.dataset import parsing_tools

DATASET_DIRECTORY = 'rabo/'
DATABASE_NAME = 'rabo'

DATA_DIRECTORY = os.getcwd() + '/data/'
RAW_DATA_DIRECTORY = DATA_DIRECTORY + DATASET_DIRECTORY + 'raw_data/'
PARSED_DATA_DIRECTORY = DATA_DIRECTORY + DATASET_DIRECTORY + 'parsed_data/'

RAW_DATA_FILE = 'raw_transactions.csv'
ACCOUNTS_FILE = 'parsed_accounts.csv'
TRANSACTIONS_FILE = 'parsed_transactions.csv'

NAMES_FOR_ACCOUNT_DATA = [
        ColumnNameGenerator('account_id')
    ]

NAMES_FOR_TRANSACTION_DATA = \
    [
        ColumnNameGenerator('start_id', 'origin_id'),
        ColumnNameGenerator('end_id', 'target_id'),
        ColumnNameGenerator('total', 'amount'),
    ]


def parse_raw_data():
    transactions = pandas.read_csv(RAW_DATA_DIRECTORY + RAW_DATA_FILE, sep=';')

    # clean duplicated transaction pairs
    transactions = transactions.groupby(['start_id', 'end_id']).agg(
        {'total': 'sum', 'count': 'sum', 'year_from': 'min', 'year_to': 'max'}).reset_index()

    senders = transactions['start_id'].unique()
    receivers = transactions['end_id'].unique()
    accounts = pandas.DataFrame({'account_id': list(set(senders) | set(receivers))})
    return accounts, transactions


def save_to_csv(accounts, transactions):
    accounts.to_csv(PARSED_DATA_DIRECTORY + ACCOUNTS_FILE, index=None)
    transactions.rename(columns=parsing_tools.get_csv_name_map(NAMES_FOR_TRANSACTION_DATA)).to_csv(PARSED_DATA_DIRECTORY + TRANSACTIONS_FILE, index_label='transaction_id')


def main():
    accounts, transactions = parse_raw_data()
    save_to_csv(accounts, transactions)


if __name__ == '__main__':
    main()
