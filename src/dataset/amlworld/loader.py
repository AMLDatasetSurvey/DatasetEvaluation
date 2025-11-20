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

CONVERSION_RATE = {
  "US Dollar": 1.0168,
  "Euro": 1.0,
  "UK Pound": 0.83571,
  "Bitcoin": 0.0000437,
  "Yen": 135.2176,
  "Yuan": 6.8664,
  "Canadian Dollar": 1.3092,
  "Rupee": 79.9275,
  "Australian Dollar": 1.4698,
  "Ruble": 61.77,
  "Shekel": 3.428,
  "Brazil Real": 5.3175,
  "Mexican Peso": 21.1283,
  "Swiss Franc": 0.9732,
  "Saudi Riyal": 3.82
}

def parse_transaction_data():
    transactions = pandas.read_csv(RAW_DATA_DIRECTORY + 'raw_transactions.csv')
    print('transactions loaded')
    transactions = transactions[['Timestamp', 'Account', 'Account.1', 'Amount Paid', 'Is Laundering', 'Payment Currency']]
    ml_transactions = transactions[transactions['Is Laundering'] == 1]
    ml_accounts = set(ml_transactions['Account']) | set(ml_transactions['Account.1'])
    all_accounts = set(transactions['Account']) | set(transactions['Account.1'])
    non_ml_accounts = all_accounts - ml_accounts
    accounts = pandas.DataFrame({'account_id': list(non_ml_accounts) + list(ml_accounts), 'is_ml': [False] * len(non_ml_accounts) + [True] * len(ml_accounts)})
    transactions['Is Laundering'] = transactions['Is Laundering'].map({1: True, 0: False})
    transactions['Amount Paid'] = transactions['Amount Paid'] * (1/transactions['Payment Currency'].map(CONVERSION_RATE))
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
