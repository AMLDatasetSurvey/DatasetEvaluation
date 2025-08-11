### Initial setup
```
git clone https://github.com/AMLDatasetSurvey/DatasetEvaluation.git
cd DatasetEvaluation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Adding datasets for analysis

- AMLSim:
  - Get the AMLSim dataset used in the study by following [this link](https://www.dropbox.com/scl/fo/7g35w7wk7gglve627we3k/AHjP6pnCmV8M62L7RxTFtkU?rlkey=6ksx339ac9117onfx3l0g3fji&e=1&dl=0) and downloading the "1Mvertices-100Medges.7z" file.
  - Extract the accounts.csv and transactions.csv files and rename them to raw_accounts.csv and raw_transactions.csv respectively.
  - Move the files to data/amlsim/raw_data/
- AMLworld
  - Get the AMLWorld dataset used in the study by following [this link](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml?select=HI-Large_Trans.csv) and downloading the "HI-Large_Trans.csv" file.
  - Rename the file to raw_transactions.csv
  - Move the file to data/amlworld/raw_data/
- Berka
  - Get the Berka dataset by following [this link](https://web.archive.org/web/20070214120527/http://lisp.vse.cz/pkdd99/DATA/data_berka.zip).
  - Extract the account.asc and trans.asc files and rename them raw_accounts.csv and raw_transactions.csv respectively.
  - Move the files to data/berka/raw_data/
- Rebobank:
  - Get the Rabobank dataset by following [this link](https://github.com/akratiiet/RaboBank_Dataset).
  - Rename the file to raw_transactions.csv
  - Move the file to data/rabo/raw_data/
- SAML-D
  - Get the SAML-D dataset by following [this link](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml) and downloading the SAML-D.csv file.
  - Rename the file to raw_transactions.csv
  - Move the file to data/samld/raw_data/
- SynthAML
  - Get the SynthAML transactions dataset by following [this link](https://springernature.figshare.com/ndownloader/files/39841711).
  - Rename the file to raw_transactions.csv
  - Move the file to data/synth/raw_data/     

### Run
On the root directory of the project run:
```
export PYTHONPATH="$PYTHONPATH:$PWD"
python src/main.py
```
