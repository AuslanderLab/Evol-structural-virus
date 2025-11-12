import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# List amino acids, map to indices (random, may be improved)
AMINO_ACIDS = 'IVLMAGPFWYCTSQNKRHED-' ##sorted by functional groups and polarity
# (Nonpolar aliphatic (hydrophobicity high→low):IVLMAGP, Aromatic (hydrophobicity high→low):FWY, Polar uncharged (hydrophobicity high→low):
# CTSQN Positively charged (hydrophobicity high→low):KRH Negatively charged (hydrophobicity high→low):ED

AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

## some params
SEQ_LEN = 6
VOCAB_SIZE = len(AMINO_ACIDS)  # 20
batch_size = 16
epochs = 10
lr = 1e-3


# ------------ functions and classes ------------

def filter_valid_proteins(sequences):
    """
    Filters a list of protein sequences, keeping only those composed of valid amino acids.

    Parameters
    ----------
    sequences : list of str
        List of protein sequences (strings of amino acid letters).

    Returns
    -------
    list of str
        List containing only sequences that have valid amino acid letters.
        Valid amino acids:IVLMAGPFWYCTSQNKRHED.
    """
    valid_aas = set(AMINO_ACIDS)
    filtered = []
    for seq in sequences:
        seq_set = set(seq.upper())
        if seq_set.issubset(valid_aas):
            filtered.append(seq)
    return filtered

def read_fasta_first_k(fasta_path,k=1000000000000):

    sequences = []
    with open(fasta_path, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    sequences.append(seq[:k])
                    seq = ''
            else:
                seq += line
        # Add last
        if seq:
            sequences.append(seq[:k])
    return sequences

class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein classification tasks.

    Each item returns:
      - Encoded amino acid indices (tensor)
      - Label (1 <-->, 0 <--> negative)

    Parameters
    ----------
    positive_seqs : list of str
        List of positive protein sequences (of len SEQ_LEN)
    negative_seqs : list of str
        List of negative protein sequences (of len SEQ_LEN).

    Attributes
    ----------
    sequences : list of str
        All protein sequences (positive + negative)
    labels : list of int
        Corresponding binary labels for each sequence
    """
    def __init__(self, positive_seqs, negative_seqs):
        self.sequences = positive_seqs + negative_seqs
        self.labels = [1] * len(positive_seqs) + [0] * len(negative_seqs)
        assert all(len(seq) == SEQ_LEN for seq in self.sequences), "All sequences must be of length "+str(SEQ_LEN)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_idx = torch.tensor([AA_TO_IDX[aa] for aa in seq], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return seq_idx, label


class ProteinModel(nn.Module):
    """
       LSTM-based neural net for protein sequence classification.

       Architecture:
         - Embedding layer: Converts AA to dense vectors
         - LSTM layer: models sequence of AA and long-term dependencies
         - Classifier: Fully connected layers with ReLU activation
         - Sigmoid: Outputs probability of positive class (can replace with softmax)

       Parameters
       ----------
       vocab_size : int
           Number of unique amino acids in the vocabulary.
       embed_dim : int, optional (default=64)
           Dimension of amino acid embeddings
       hidden_dim : int, optional (default=128) --> higher gets very slow and worse
           Hidden dimension of the LSTM
       """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) #word embedding - AA representation
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True) ##LSTM cell (one cell is best, can play with the hidden.embed dim params)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        hn = hn.squeeze(0)
        out = self.classifier(hn)
        return out.squeeze(1)


def train(model, dataloader, criterion, optimizer, device):
    """
      Trains the protein model per epoch.

      Parameters
      ----------
      model : ProteinModel
          Model to train
      dataloader : DataLoader
          PyTorch DataLoader for the training data.
      criterion : loss function
          Loss function (e.g., BCELoss or BCEWithLogitsLoss).
      optimizer : torch.optim.Optimizer
          Optimizer (e.g., Adam).
      device : torch.device
          Device to run training on ('cpu' or 'cuda').

      Returns
      -------
      float
          Average training loss over the dataset.
      """
    model.train()
    total_loss = 0
    for seqs, labels in dataloader:
        seqs = seqs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
    return total_loss / len(dataloader.dataset)



class ProteinDataset_testing(Dataset):
    """
    PyTorch Dataset for inference on protein sequences (without labels). Good to keep train,test seperate.

    Parameters
    ----------
    sequences : list of str
        Protein sequences (of len SEQ_LEN).
    """
    def __init__(self, sequences):
        self.sequences = sequences
        assert all(len(seq) == SEQ_LEN for seq in self.sequences), "All sequences must be if length "+SEQ_LEN

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_idx = torch.tensor([AA_TO_IDX[aa] for aa in seq], dtype=torch.long)
        return seq_idx

def predict(model, sequences, device):
    """
    Applies model to a list of protein sequences for prediction of short/long protein

    Parameters
    ----------
    model : ProteinModel
        Trained model used for prediction
    sequences : list of str
        Protein sequences to predict using model
    device : torch.device
        Device for running

    Returns
    -------
    probs : list of float
        Predicted probabilities of positive class.
    preds : list of int
        Binary predictions (0/1) using threshold of 0.5.
    """
    dataset = ProteinDataset_testing(sequences)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for seqs in loader:
            seqs = seqs.to(device)
            outputs = model(seqs)  # outputs probabilities (sigmoid)
            probs.extend(outputs.cpu().numpy())
            preds.extend((outputs >= 0.5).cpu().numpy().astype(int))
    return probs, preds


# ------------ training model ------------

###ToDo: add more parvo to balance and increase the sizes of data?
alld=read_fasta_first_k('./intermediate_files/aligned-psi-blast-relative-all.fa')
count_gaps = [sum([alld[i][k]=='-' for i in range(len(alld))]) for k in range(len(alld[0]))]
windows_consider = [i for i in range(len(count_gaps)-SEQ_LEN+1) if max(count_gaps[i:i+SEQ_LEN])<len(alld)/3]

positive_seqs=read_fasta_first_k('./intermediate_files/adeno.fa')
negative_seqs=read_fasta_first_k('./intermediate_files/parvo.fa')

train_pos=positive_seqs[:10]
train_neg=negative_seqs[:10]

positive_val_seqs = positive_seqs[10:15]
negative_val_seqs = negative_seqs[10:15]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
aurocs = []; window = []
## run over windows
##remove windows with many gaps across anything
for w in windows_consider:

    train_dataset = ProteinDataset([i[w:w+SEQ_LEN] for i in train_pos], [i[w:w+SEQ_LEN] for i in train_neg] )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ProteinModel(VOCAB_SIZE).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}: Train loss={train_loss:.4f}")


    # Predict on positive and negative test sequences separately
    pos_probs, pos_preds = predict(model, [i[w:w+SEQ_LEN] for i in positive_val_seqs], device)
    neg_probs, neg_preds = predict(model, [i[w:w+SEQ_LEN] for i in negative_val_seqs], device)

    # calc AUROC on validation
    labels = [1 for i in range(len(pos_probs))] + [0 for i in range(len(neg_probs))]
    scores = pos_probs + neg_probs
    auroc = roc_auc_score(labels, scores)
    aurocs.append(auroc)
    window.append((w,w+SEQ_LEN))
start_w=[window[i][0] for i in range(len(window))]
stop_w=[window[i][1] for i in range(len(window))]

# postprocess the auroc and the windows 
d1=pd.DataFrame(list(zip(start_w, stop_w, aurocs)), columns=["start","stop", "auroc"])
d1.to_csv("./results/check_df.csv", index=False)
d1 = d1.dropna(subset=["start", "stop", "auroc"])
sub_d1 = d1.dropna(subset=["start", "stop", "auroc"]).apply(
    lambda row: pd.Series(row["auroc"], index=range(int(row["start"]), int(row["stop"]) + 1)),
    axis=1
)
df=sub_d1.stack().reset_index(level=0, drop=True)
df=df.to_frame()

# plot preliminary
d1=pd.DataFrame(aurocs, index=window)
d1.plot(kind="area", figsize=(20,6))
plt.xticks(ticks=range(len(d1.index)), labels=d1.index, rotation=90)
plt.savefig("./results/draft_performance.png")
