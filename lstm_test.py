from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from arff2pandas import a2p
from torch import nn, optim
from pylab import rcParams
from matplotlib import rc
import pandas as pd
import numpy as np
import torch
import copy
import pickle

from server import unpickle_data, COND_ID

## COPIED TIME SERIES SNIPPETS
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


## CUSTOM SNIPPETS

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


## MAIN CODE START
if __name__ == '__main__':
    
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
    # Data Preprocessing - Separating control signals
    control_df = signal_df[signal_df[0] == 3]
    sickly_df = signal_df[signal_df[0] != 3]

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
        model = torch.load('model.pth')
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
        MODEL_PATH = 'model.pth'
        torch.save(model, MODEL_PATH)
        with open('model_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        print(' > Model finished and saved')
    
    # Calculation and saving of loss data
    _, losses = predict(model, train_dataset)
    with open('model_losses.pkl', 'wb') as f:
        pickle.dump(losses, f)
    
    print(" ? Predicting...")
    predictions, pred_losses, predict(model, test_normal_dataset)
    THRESHOLD = 26
    correct = sum(l <= THRESHOLD for l in pred_losses)

    sickly_dataset = test_sickly_dataset[:len(test_control_dataset)]
    predictions, pred_losses = predict(model, anomaly_dataset())
    
























