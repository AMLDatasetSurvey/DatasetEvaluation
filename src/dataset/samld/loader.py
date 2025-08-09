import subprocess
import os
from src.dataset.parsing_tools import ColumnNameGenerator
from src.dataset import parsing_tools
from tqdm import tqdm
import pandas
from pandas import isna

DATASET_DIRECTORY = 'samld/'

DATA_DIRECTORY = os.getcwd() +  '/data/'
RAW_DATA_DIRECTORY = DATA_DIRECTORY + DATASET_DIRECTORY + 'raw_data/'
PARSED_DATA_DIRECTORY = DATA_DIRECTORY + DATASET_DIRECTORY + 'parsed_data/'

ACCOUNTS_FILE = 'parsed_accounts.csv'
TRANSACTIONS_FILE = 'parsed_transactions.csv'

NAMES_FOR_TRANSACTION_DATA = \
    [
        ColumnNameGenerator('Sender_account', 'origin_id'),
        ColumnNameGenerator('Receiver_account', 'target_id'),
        ColumnNameGenerator('Amount', 'amount'),
        ColumnNameGenerator('Is_laundering', 'is_ml'),
        ColumnNameGenerator('Date', 'timestamp'),
    ]

def parse_transaction_data():
    transactions = pandas.read_csv(RAW_DATA_DIRECTORY + 'raw_transactions.csv', usecols=['Date', 'Sender_account', 'Receiver_account', 'Amount', 'Is_laundering'])
    ml_transactions = transactions[transactions['Is_laundering'] == 1]
    ml_accounts = set(ml_transactions['Sender_account']) | set(ml_transactions['Receiver_account'])
    all_accounts = set(transactions['Sender_account']) | set(transactions['Receiver_account'])
    non_ml_accounts = all_accounts - ml_accounts
    accounts = pandas.DataFrame({'account_id': list(non_ml_accounts) + list(ml_accounts), 'is_ml': [False] * len(non_ml_accounts) + [True] * len(ml_accounts)})
    transactions['Is_laundering'] = transactions['Is_laundering'].map({1: True, 0: False})
    return accounts, transactions


def save_to_csv(parsed_accounts, parsed_transactions):
    parsed_accounts.to_csv(PARSED_DATA_DIRECTORY + ACCOUNTS_FILE, index=False)

    parsed_transactions. \
        rename(columns=parsing_tools.get_csv_name_map(NAMES_FOR_TRANSACTION_DATA)). \
        to_csv(PARSED_DATA_DIRECTORY + TRANSACTIONS_FILE, index=False)

def main():
    accounts, transactions = parse_transaction_data()
    print('Saving data to CSV\n')
    save_to_csv(accounts, transactions)

if __name__ == '__main__':
    main()
