import os
from src.dataset.parsing_tools import ColumnNameGenerator
from src.dataset import parsing_tools
from tqdm import tqdm
import pandas

DATASET_DIRECTORY = 'amlworld/'

DATA_DIRECTORY = os.getcwd() + '/data/'
RAW_DATA_DIRECTORY = DATA_DIRECTORY + DATASET_DIRECTORY + 'raw_data/'
PARSED_DATA_DIRECTORY = DATA_DIRECTORY + DATASET_DIRECTORY + 'parsed_data/'

ACCOUNTS_FILE = 'parsed_accounts.csv'
TRANSACTIONS_FILE = 'parsed_transactions.csv'

NAMES_FOR_TRANSACTION_DATA = \
    [
        ColumnNameGenerator('Timestamp', 'timestamp'),
        ColumnNameGenerator('Account', 'origin_id'),
        ColumnNameGenerator('Account.1', 'target_id'),
        ColumnNameGenerator('Amount Paid', 'amount'),
        ColumnNameGenerator('Is Laundering', 'is_ml'),
    ]


def parse_transaction_data():
    transactions = pandas.read_csv(RAW_DATA_DIRECTORY + 'raw_transactions.csv')
    transactions = transactions[['Timestamp', 'Account', 'Account.1', 'Amount Paid', 'Is Laundering']]
    ml_transactions = transactions[transactions['Is Laundering'] == 1]
    ml_accounts = set(ml_transactions['Account']) | set(ml_transactions['Account.1'])
    all_accounts = set(transactions['Account']) | set(transactions['Account.1'])
    non_ml_accounts = all_accounts - ml_accounts
    accounts = pandas.DataFrame({'account_id': list(non_ml_accounts) + list(ml_accounts), 'is_ml': [False] * len(non_ml_accounts) + [True] * len(ml_accounts)})
    transactions['Is Laundering'] = transactions['Is Laundering'].map({1: True, 0: False})
    return accounts, transactions

def save_to_csv(parsed_accounts, parsed_transactions):
    parsed_accounts.to_csv(PARSED_DATA_DIRECTORY + ACCOUNTS_FILE, index=False)

    parsed_transactions. \
        rename(columns=parsing_tools.get_csv_name_map(NAMES_FOR_TRANSACTION_DATA)). \
        to_csv(PARSED_DATA_DIRECTORY + TRANSACTIONS_FILE, index=False)



def main():
    tqdm.pandas()
    print('Parsing transaction data\n')
    parsed_accounts, parsed_transactions = parse_transaction_data()
    print('Saving data to CSV\n')
    save_to_csv(parsed_accounts, parsed_transactions)

if __name__ == '__main__':
    main()
