import os.path
import pandas

PARSED_TRANSACTIONS_FILE = 'parsed_transactions.csv'
PARSED_ACCOUNTS_FILE = 'parsed_accounts.csv'

RESULTS_DIR = 'results/'

AML_SIM_1M_NAME = 'amlsim_1m'
BERKA_PLUS_NAME = 'berka_plus'
RABOBANK_PLUS_NAME = 'rabobank_plus'
AMLWORLD_NAME = 'amlworld'
SAMLD_NAME = 'samld'

DATA_DIR = os.getcwd() + '/data/'
AML_SIM_1M_DIR = DATA_DIR + 'amlsim/parsed_data/'
RABO_DIR = DATA_DIR + 'rabo/parsed_data/'
BERKA_DIR = DATA_DIR + 'berka/parsed_data/'
AMLWORLD_DIR = DATA_DIR + 'amlworld/parsed_data/'
SAMLD_DIR = DATA_DIR + 'samld/parsed_data/'

class DatasetInfo:
    def __init__(self, name, data_dir):
        self.name = name
        self.data_dir = data_dir

class Dataset:
    def __init__(self, dataset_info):
        self.info = dataset_info
        self.accounts, self.transactions, self.transactions_by_origin, self.transactions_by_target = self.load_dataset()

    def load_dataset(self):

        accounts = pandas.read_csv(self.info.data_dir + PARSED_ACCOUNTS_FILE)
        transactions = pandas.read_csv(self.info.data_dir + PARSED_TRANSACTIONS_FILE)

        transactions_by_origin = transactions.set_index('origin_id').sort_index()
        transactions_by_target = transactions.set_index('target_id').sort_index()

        return accounts, transactions, transactions_by_origin, transactions_by_target

AMLSim1mInfo = DatasetInfo(AML_SIM_1M_NAME, AML_SIM_1M_DIR)
BerkaPlusInfo = DatasetInfo(BERKA_PLUS_NAME, BERKA_DIR)
RaboPlusInfo = DatasetInfo(RABOBANK_PLUS_NAME, RABO_DIR)
AMLWorldInfo = DatasetInfo(AMLWORLD_NAME, AMLWORLD_DIR)
SAMLDInfo = DatasetInfo(SAMLD_NAME, SAMLD_DIR)

def get_datasets_info():
    return [AMLSim1mInfo, AMLWorldInfo, BerkaPlusInfo, RaboPlusInfo, SAMLDInfo]

def get_datasets():
    datasets = list(map(Dataset, get_datasets_info()))
    return datasets

if __name__ == '__main__':
    get_datasets()