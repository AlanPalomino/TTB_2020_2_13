clear#!usr/bin/env python3
# _*_ coding: utf-8 _*_ #
#
#___________________________________________________________________________
#|                                                                         |
#|    TTB__2020_1_13 server code:                                          |
#|      Código para distribución de carga en el servidor                   |
#|                                                                         |
#|                                                                         |
#|_________________________________________________________________________|


# ===================== Librerias Utilizadas ====================== #
from sklearn.model_selection import train_test_split
from TT_utilities import Case, Record, NL_METHODS
from entropy import spectral_entropy
from multiprocessing import Pool
from scipy.stats import stats
from hurst import compute_Hc
from torch import nn, optim
from arff2pandas import a2p
from pylab import rcParams 
from pathlib import Path

import torch.nn.functional as F
import TT_utilities
import pandas as pd
import numpy as np
import pickle
import torch
import copy
import sys
import os
import re


CSV_COLS = [
    'case',
    'record',
    'condition',
    'cond_id',
    'length',
]
for m in NL_METHODS:
    CSV_COLS.extend([
        m['tag'] + '_mean',
        m['tag'] + '_variance',
        m['tag'] + '_skewness',
        m['tag'] + '_spectral_entropy',
    ])


COND_ID = dict(
    atrial_fibrillation=0,
    congestive_heartfailure=1,
    myocardial_infarction=2,
    control=3
)

# ===================== Funciones para LSTM ======================= #
def create_dataset(df):
  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features


class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x


def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history


def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses


## CUSTOM LOADING SNIPPETS

def load_dummy() -> pd.DataFrame:
    with open('Test/linear_healthy.pkl', 'rb') as pf:
        DF = pickle.load(pf)
    with open('Test/linear_sick.pkl', 'rb') as pf:
        DF = DF.append(pickle.load(pf))
    return DF[ DF.length > 500*6]


def load_mimic() -> pd.DataFrame:
    cases = unpickle_data()
    columns = ['case', 'record', 'condition', 'cond_id', 'length', 'fs', 'rr']
    DF = pd.DataFrame(columns=columns)
    for c in cases:
        for r in c:
            data = [
                c._case_name,
                r.name,
                c.pathology,
                COND_ID.get(c.pathology),
                len(r.rr),
                r.fs,
                r.rr
            ]
            S = pd.Series(data=data, index=columns)
            DF = DF.append(S)
    return DF


# ===================== Funciones y Métodos ======================= #
def hurst_eval(rr):
    H, _, _ = compute_Hc(rr)
    return H


def generate_csv():
    cases_list = unpickle_data()
    csv_name = 'complete_data.csv'
    FULL_CSV = pd.DataFrame(columns=CSV_COLS)
    for c in cases_list:
        print(f"    > Case {c._case_name}")
        for r in c:
            print(f"\t\t + RECORD {r.name}", end="")
            values = list()
            for k, v in r.N_LINEAR.items():
                s = stats.describe(v)
                values.extend([
                    s[2],                                       # Mean
                    s[3],                                       # Variance
                    s[4],                                       # Skewness
                    spectral_entropy(v, sf=r.fs, method='fft')  # Spectral Entropy
                ])
            row_data = [
                c._case_name,               # Case
                r.name,                     # Record
                c.pathology,                # Condition
                COND_ID[c.pathology],       # Condition ID
                len(r.rr),                  # RR Length
            ] + values
            FULL_CSV = FULL_CSV.append(
                pd.Series(
                    data=row_data,
                    index=CSV_COLS
                ), ignore_index=True
            )
            print("[v]")
    FULL_CSV.to_csv(csv_name, index=False)


def unpickle_data():
    p_paths = list(Path('./Pickled').glob('*.pkl')) 
    UNPICKLED = list()
    for pkl in p_paths:
        with pkl.open('rb') as pf:
            UNPICKLED.append(
                pickle.load(pf)
            )
    return UNPICKLED


def process_case(case_path: Path):
    c = Case(case_path)
    c.process()
    print(f"\n\n\t\tCASE {c._case_name} has {len(c)} RECORDS\n\n")
    if len(c) > 0:
        with open(f'Pickled/case_{c._case_name}.pkl', 'wb') as pf:
            pickle.dump(c, pf)


def gen_name(path):
    c_name = re.search('p[0-9]{6}', str(path))[0]
    return path.joinpath(c_name)


def pickle_data():
    RECORD_DIRS = list(Path("./Data").glob("*_p0*")) 
    RECORD_DIRS = [gen_name(p) for p in RECORD_DIRS]

    p = Pool()
    p.map(process_case, RECORD_DIRS)
    p.close()


def help():
    global OPTS
    print("""
    SERVER SCRIPT OPTIONS
        Exclusive options for use in server!!
          """)
    for opt in OPTS:
        print(f"{', '.join(opt['opts'])} :")
        print(f"\t{opt['desc']}\n")


def test_unpickle(parent):
    unpickled_cases = list()
    for d in parent.glob('*.pkl'):
        with d.open('rb') as pf:
            unpickled_cases.append(
                pickle.load(pf)
            )
    return unpickled_cases


def save_test():
    TEST_DIRS = list(Path('.').glob('Test_*ws/'))
    for td in TEST_DIRS:
        t_cases = test_unpickle(td)

        pdir = "Test/"

        csv_name = pdir + td.stem + '.csv'
        pkl_name = pdir + td.stem + '.pkl'

        csv_data = pd.DataFrame(columns=CSV_COLS)
        pkl_data = pd.DataFrame(columns=CSV_COLS[:5])

        for c in t_cases:
            for r in c:
                # Process for CSV
                values = list()
                row_data = [
                    c._case_name,
                    r.name,
                    c.pathology,
                    COND_ID[c.pathology],
                    len(r.rr_int),
                ]
                for k, v in r.N_LINEAR.items():
                    s = stats.describe(v)
                    row_data.extend([
                        s[2],
                        s[3],
                        s[4],
                        spectral_entropy(v, sf=r.fs, method='fft')
                    ])
                csv_data = csv_data.append(
                    pd.Series(
                        data=row_data,
                        index=CSV_COLS,
                    ), ignore_index=True
                )
                # Process for pickle
                pkl_row = {
                    'case': c._case_name,
                    'record': r.name,
                    'condition': c.pathology,
                    'cond_id': COND_ID[c.pathology],
                    'length': len(r.rr_int)
                }
                pkl_row.update(r.N_LINEAR)
                pkl_data = pkl_data.append(pd.DataFrame(pkl_row))

        # DATA IS SAVED IN BOTH FORMATS
        csv_data.to_csv(csv_name, index=False)
        with open(pkl_name, 'wb') as pf:
            pickle.dump(pkl_data, pf)


def test_case(ddir: Path):
    c = Case(ddir)
    c.process()
    if len(c) != 0:
        with open(f'Test_{TT_utilities.RR_WLEN}ws/case_{c._case_name}.pkl', 'wb') as pf:
            pickle.dump(c, pf)
        print(f'\n\n\tTEST CASE with {len(c)} records processed and saved to: case_{c._case_name}.pkl\n\n')


def run_test():
    n = int(sys.argv[2])

    af_dirs = list(Path('Data/').glob('atrial_fibrillation_p*'))[:n]
    mi_dirs = list(Path('Data/').glob('myocardial_infarction_p*'))[:n]
    ch_dirs = list(Path('Data/').glob('congestive_heartfailure_p*'))[:n]

    try:
        os.mkdir(f"Test_{TT_utilities.RR_WLEN}ws/")
    except FileExistsError:
        print("REWRITING TEST VALUES")

    data_dirs = [ gen_name(d) for d in af_dirs + mi_dirs + ch_dirs]

    p = Pool()
    p.map(test_case, data_dirs)
    p.close()


def lstm_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RANDOM_SEED = 42

    CLASS_NORMAL = 1

    class_names = ['control', 'atrial_fibrillation', 'congestive_heartfailure', 'myocardial_infarction']

    # READING TESTS
    full_df = load_dummy()
    # TESTS ENDING

    MIN_LEN = full_df['length'].min()
    print(full_df.groupby('condition')['case'].count())
    print(f'With minimal length: {MIN_LEN}')
    # Equalize data on a separate dataframe with identifiers
      
    data = [
      list([row['cond_id']]) + list(row['rr'][:MIN_LEN]) for i, row in full_df.iterrows()
    ]
    signal_df = pd.DataFrame(data)
    for id in [0, 1]:
        # Data Preprocessing - Separating control signals
        control_df = signal_df[signal_df[0] == 3]
        sickly_df = signal_df[signal_df[0] == id]

        # Training collections generation
        train_df, val_df = train_test_split(
            control_df,
            test_size=0.15,
            random_state=RANDOM_SEED
            )
        val_df, test_df = train_test_split(
            val_df,
            test_size=0.33,
            random_state=RANDOM_SEED
            )

        train_dataset, seq_len, n_features = create_dataset(train_df)
        val_dataset, _, _ = create_dataset(val_df)
        test_control_dataset, _, _ = create_dataset(test_df)
        test_sickly_dataset, _, _ = create_dataset(sickly_df)
        
        print(" > Datasets setup finished")

        # Training anew or getting previously used model
        
        try:
            model = torch.load(f'model{id}.pth')
        except FileNotFoundError:
            print(' ¿ Starting Autoencoder Model')
            model = RecurrentAutoencoder(seq_len, n_features, 128)
            model = model.to(device)
            print(' ! Training Model...')
            model, history = train_model(
                model,
                train_dataset,
                val_dataset,
                n_epochs=150
            )

            # Saving model and history
            MODEL_PATH = f'model{id}.pth'
            torch.save(model, MODEL_PATH)
            with open(f'model{id}_history.pkl', 'wb') as f:
                pickle.dump(history, f)
            print(' > Model finished and saved')

        def plot_prediction(data, model, title, ax):
          predictions, pred_losses = predict(model, [data])

          ax.plot(data, label='true')
          ax.plot(predictions[0], label='reconstructed')
          ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
          ax.legend()

  
        # Calculation and saving of loss data
        
        _, losses = predict(model, train_dataset)
        with open('model_losses.pkl', 'wb') as f:
            pickle.dump(losses, f)

        print(" ? Predicting...")
        predictions, pred_losses, predict(model, test_control_dataset)
        THRESHOLD = 26
        correct = sum(l <= THRESHOLD for l in pred_losses)

        sickly_dataset = test_sickly_dataset[:len(test_control_dataset)]
        predictions, pred_losses = predict(model, anomaly_dataset())


def dummy_process(jsonfiles: list) -> pd.DataFrame:
    def get_ids(row: pd.Series) -> pd.Series:
        try:
            row['condition'] = COND_ID.get(row['conditon'])[0]
            row['cond_id'] = COND_ID.get(row['conditon'])[1]
        except KeyError as e:
            print(row)
            raise KeyError(e)
        return row

    # Data es read from json paths
    data = list()
    for jf in jsonfiles:
        with jf.open() as file:
            mixed = json.load(file)
            data.extend([reg for reg in mixed if reg['approved']])
    data = pd.DataFrame(data)
    data = data.apply(get_ids, axis=1)
    # Dataframe adjustment and ordering
    data["rr"] = data.apply(lambda row: np.array(row["rr"])/row["fs"], axis=1)
    data["rr"] = data["rr"].apply(lambda signal: signal[np.where(signal < 2)])
    data["length"] = data["rr"].apply(lambda signal: len(signal))
    data["case"] = data["record"]
    # We filter for only relevant columns
    data = data[['case', 'record', 'condition', 'cond_id', 'length', 'rr']]
    print(f" < Vectorizing...", end="")
    data_DF = vectorize_df(data)
    print("\tDONE >")

    return data_DF


def vectorize_df(data: pd.DataFrame):
    def gen_vectors(row: pd.Series) -> pd.Series:
        rr = row['rr']
        # Linear
        tags = ['mean', 'variance', 'skewness', 'kurtosis']
        for tag, t in zip(tags, linearWindowing(rr)):
            row[tag] = t
        # Non Linear
        tags = ['appen', 'sampen', 'hfd', 'dfa', 'pratio', 'hurst']
        for m, t in zip(tags, nonLinearWindowing(rr)):
            row[tag] = t
        return row
    return data.apply(gen_vectors, axis=1)


def full_test(ddir: Path):
    c = Case(ddir)
    c.process(mode="full")
    if len(c) != 0:
        with open(f'csv_datasets/sample_case{c._case_name}.pkl', 'wb') as pf:
            pickle.dump(c, pf)
        print(f'\n\n\tTEST CASE with {len(c)} records processed and saved to: sample_case{c._case_name}.pkl\n\n')


def sample_graphs():
    # PATHOLOGICAL CASES PROCESING
    af_dirs = list(Path('Data/').glob('atrial_fibrillation_p*'))[:4]
    mi_dirs = list(Path('Data/').glob('myocardial_infarction_p*'))[:4]
    ch_dirs = list(Path('Data/').glob('congestive_heartfailure_p*'))[:4]
    
    data_dirs = [ gen_name(d) for d in af_dirs+mi_dirs+ch_dirs]
    
    p = Pool()
    p.map(full_test, data_dirs)
    p.close()
    
    # PATHOLOGICAL DATAFRAME GENERATION
    
    CASES = list()
    for ddir in list(Path('csv_datasets/').glob('sample_case*')):
        with ddir.open('rb') as pf:
            CASES.append(
                pickle.load(pf)
            )
    
    columns = [
        'case', 'record', 'condition', 'condition_id', 'length',
        'rr', 'mean', 'variance', 'skewness', 'kurtosis',
        'appen', 'sampen', 'hfd', 'dfa', 'pratio', 'hurst'
    ]
    
    PATHOLOGIC_DF = pd.DataFrame(columns=columns)
    for c in CASES:
        for r in c:
            PATHOLOGIC_DF = PATHOLOGIC_DF.append(
                pd.Series(
                    data=[
                        c._case_name,
                        r.name,
                        c.pathology,
                        COND_ID[c.pathology],
                        len(r.rr),
                        r.rr,
                        r.LINEAR['mean'],
                        r.LINEAR['var'],
                        r.LINEAR['skew'],
                        r.LINEAR['kurt'],
                        r.N_LINEAR['ae'],
                        r.N_LINEAR['se'],
                        r.N_LINEAR['hfd'],
                        r.N_LINEAR['dfa'],
                        r.N_LINEAR['psd'],
                        r.N_LINEAR['hst']
                    ],
                    index=columns
                ),
                ignore_index=True
            )
    
    with open('Sample_Pathologic.pkl', 'wb') as pf:
        pickle.dump(PATHOLOGIC_DF, pf)
    
    # CONTROL CASES PROCESSING
    h_jsonfiles = [
        Path('Data_Jsons/normal-sinus-rhythm-rr-interval-database-1.0.0.json'),
        Path('Data_Jsons/nn-cases-healthy-control.json')
    ]
    HEALTHY_DF = dummpy_process(h_jsonfiles)
    
    with open('Sample_Healthy.pkl', 'wb') as pf:
        pickle.dump(HEALTHY_DF, pf)


def main(argv):
    global OPTS
    for opt in OPTS:
        print(f'is {argv} in {opt["opts"]}')
        if argv in opt['opts']:
            opt['func']()
            break
    else:
        print("""
        No valid parameter detected
        Check bellow for valid options:
              """)
        help()


OPTS = [
    {
        'opts': ['-h', '--help'],
        'desc': 'Prints valid options to use the script.',
        'func': help
    },{
        'opts': ['-pd', '--pickle_data'],
        'desc': 'Processes and pickles downloaded data',
        'func': pickle_data
    },{
        'opts': ['-gc', '--generate_csv'],
        'desc': 'Unpickles data and generates the corresponding csv.',
        'func': generate_csv
    },{
        'opts': ['-rt', '--run_test'],
        'desc': 'Run selected test with [n] number of cases per pathology.',
        'func': run_test
    },{
        'opts': ['-st', '--save_test'],
        'desc': 'Recounts each data compendium and generates a csv file.',
        'func': save_test
    },{
        'opts': ['-sg', '--sample_graphs'],
        'desc': 'Gets uniform data for posterior plotting by pickled DataFrames.',
        'func': sample_graphs
    }
]

# ===================== Ejecución principal ======================= #
if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1])

