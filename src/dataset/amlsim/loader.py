import subprocess
import os
from src.dataset.parsing_tools import ColumnNameGenerator
from src.dataset import parsing_tools
from tqdm import tqdm
import pandas

DATASET_DIRECTORY = 'amlsim/'

DATA_DIRECTORY = os.getcwd() +  '/data/'
RAW_DATA_DIRECTORY = DATA_DIRECTORY + DATASET_DIRECTORY + 'raw_data/'
PARSED_DATA_DIRECTORY = DATA_DIRECTORY + DATASET_DIRECTORY + 'parsed_data/'

ACCOUNTS_FILE = 'parsed_accounts.csv'
TRANSACTIONS_FILE = 'parsed_transactions.csv'

NAMES_FOR_ACCOUNT_DATA = \
    [
        ColumnNameGenerator('ACCOUNT_ID', 'account_id'),
        ColumnNameGenerator('IS_FRAUD', 'is_ml')
    ]

NAMES_FOR_TRANSACTION_DATA = \
    [
        ColumnNameGenerator('SENDER_ACCOUNT_ID', 'origin_id'),
        ColumnNameGenerator('RECEIVER_ACCOUNT_ID', 'target_id'),
        ColumnNameGenerator('TX_AMOUNT', 'amount'),
        ColumnNameGenerator('IS_FRAUD', 'is_ml'),
        ColumnNameGenerator('TIMESTAMP', 'timestamp'),
    ]

def parse_account_data():
    accounts = pandas.read_csv(RAW_DATA_DIRECTORY + 'raw_accounts.csv', usecols=['ACCOUNT_ID', 'IS_FRAUD'])
    return accounts
    
def parse_transaction_data():
    transactions = pandas.read_csv(RAW_DATA_DIRECTORY + 'raw_transactions.csv', usecols=['IS_FRAUD', 'SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'TIMESTAMP'])
    return transactions


def save_to_csv(parsed_accounts, parsed_transactions):
    parsed_accounts. \
        rename(columns=parsing_tools.get_csv_name_map(NAMES_FOR_ACCOUNT_DATA)). \
        to_csv(PARSED_DATA_DIRECTORY + ACCOUNTS_FILE, index=False)

    parsed_transactions. \
        rename(columns=parsing_tools.get_csv_name_map(NAMES_FOR_TRANSACTION_DATA)). \
        to_csv(PARSED_DATA_DIRECTORY + TRANSACTIONS_FILE, index=False)

def main():
    tqdm.pandas()
    print('Parsing account data\n')
    parsed_accounts = parse_account_data()
    print('Parsing transaction data\n')
    parsed_transactions = parse_transaction_data()
    print('Saving data to CSV\n')
    save_to_csv(parsed_accounts, parsed_transactions)

if __name__ == '__main__':
    main()
