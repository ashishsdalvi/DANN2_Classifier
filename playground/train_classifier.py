import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Sampler
import os 
import time
import h5py
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split




class PredictionsDataset(Dataset):
    def __init__(self, chromosomes, forzoi_dir, cadd_dir, label_dir, input_type="forzoi"):
        assert input_type in {"forzoi", "cadd", "combo"}
        self.input_type = input_type
        self.chromosomes = chromosomes
        self.forzoi_dir = forzoi_dir
        self.cadd_dir = cadd_dir
        self.label_dir = label_dir

        self.file_meta = []
        self.cumulative_idx = [0]
        self.global_indices = []
        self.all_labels = []

        # Store pointers to data locations
        self._forzoi = {}
        self._cadd = {}
        self._labels = {}

        for chrom in chromosomes:
            try:
                forzoi_path = os.path.join(forzoi_dir, f"{chrom}_l2.npy")
                cadd_path = os.path.join(cadd_dir, f"CADD_scores_{chrom}.npy")
                label_path = os.path.join(label_dir, f"{chrom}_labels.csv")

                # Verify file existence
                if not os.path.exists(forzoi_path) or not os.path.exists(label_path):
                    print(f"Skipping {chrom} due to missing Forzoi or Label files.")
                    continue

                # Load Forzoi data and labels
                forzoi = np.load(forzoi_path, mmap_mode="r")
                labels = pd.read_csv(label_path)["variant_label"].to_numpy(dtype=np.float32)

                if input_type in {"cadd", "combo"}:
                    if not os.path.exists(cadd_path):
                        print(f"Skipping {chrom} due to missing CADD file.")
                        continue
                    cadd = np.load(cadd_path, mmap_mode="r")
                    assert forzoi.shape[0] == cadd.shape[0], f"Mismatch in {chrom} Forzoi vs CADD"

                # Ensure that lengths match
                assert forzoi.shape[0] == len(labels), f"Mismatch in {chrom} labels"

                # Store data pointers
                self._forzoi[chrom] = forzoi
                if input_type in {"cadd", "combo"}:
                    self._cadd[chrom] = cadd
                self._labels[chrom] = labels

                # Update cumulative indices
                start_idx = self.cumulative_idx[-1]
                end_idx = start_idx + forzoi.shape[0]
                self.cumulative_idx.append(end_idx)

                # Append labels and global indices
                self.all_labels.extend(labels)
                self.global_indices.extend([(chrom, idx) for idx in range(forzoi.shape[0])])

                # Update file_meta
                num_features = forzoi.shape[1] if input_type == "forzoi" else \
                               cadd.shape[1] if input_type == "cadd" else \
                               forzoi.shape[1] + 1  # 1 for CADD scalar score

                self.file_meta.append({
                    "chrom": chrom,
                    "forzoi_path": forzoi_path,
                    "cadd_path": cadd_path if input_type in {"cadd", "combo"} else None,
                    "label_path": label_path,
                    "length": forzoi.shape[0],
                    "num_features": num_features
                })

            except Exception as e:
                print(f"Error processing {chrom}: {e}")

        # Convert labels to numpy array for efficient indexing
        self.all_labels = np.array(self.all_labels, dtype=np.float32)

        # Check for empty file_meta
        if not self.file_meta:
            raise ValueError("No valid data found for the provided chromosomes.")

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, idx):
        chrom, local_idx = self.global_indices[idx]
        forzoi = self._forzoi[chrom]
        labels = self._labels[chrom]
        label = labels[local_idx]

        if self.input_type == "forzoi":
            feats = forzoi[local_idx]
        elif self.input_type == "cadd":
            cadd = self._cadd[chrom]
            feats = np.array([cadd[local_idx].item()])
        else:
            forzoi_feats = forzoi[local_idx]
            cadd_feats = np.array([self._cadd[chrom][local_idx].item()])
            feats = np.concatenate([forzoi_feats, cadd_feats])

        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def get_labels(self):
        """Return all labels as a single concatenated array."""
        return self.all_labels



######################################## Regression Model ##############################

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dims, output_dims)

    def forward(self, x):
        # y_pred = torch.sigmoid(self.linear(x))
        y_pred = self.linear(x)
        return y_pred 

######################################## DANN Model ##############################

class DANNModel(nn.Module):
    def __init__(self, input_dim):
        super(DANNModel, self).__init__()
        self.dropout_rate = 0.1
        self.hidden_dim = 1000

        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

########################################################################################


def load_labels_for_indices(chromosomes, label_dir, indices):
    """
    Load labels only for the specified indices.
    """
    labels = []
    offset = 0

    for chrom in chromosomes:
        label_path = os.path.join(label_dir, f"{chrom}_labels.csv")
        chrom_labels = pd.read_csv(label_path, usecols=["variant_label"])["variant_label"].to_numpy()
        num_samples = len(chrom_labels)

        # Get indices for the current chromosome
        chrom_indices = [idx - offset for idx in indices if offset <= idx < offset + num_samples]

        # Extract labels for these indices
        selected_labels = chrom_labels[chrom_indices]
        labels.append(selected_labels)

        offset += num_samples

    return np.concatenate(labels)


class BalancedSampler(Sampler):
    def __init__(self, labels):
        """
        Args:
            labels (array-like): A list or array of labels corresponding to the dataset.
        """
        self.labels = np.array(labels)
        self.indices_by_class = {label: np.where(self.labels == label)[0] for label in np.unique(self.labels)}
        self.min_samples = min(len(indices) for indices in self.indices_by_class.values())
        self.num_samples = self.min_samples * len(self.indices_by_class)

    def __iter__(self):
        indices = []
        for label, label_indices in self.indices_by_class.items():
            # Randomly sample without replacement
            sampled_indices = np.random.choice(label_indices, self.min_samples, replace=False)
            indices.extend(sampled_indices)

        # Shuffle the indices
        indices = np.random.permutation(indices)
        return iter(indices)
    
    def __len__(self):
        return self.num_samples



def train_and_evaluate_model(chromosomes, label_dir, dataset, name, epochs=1, batch_size=512, num_workers=1):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    # Access the global labels directly
    train_labels = dataset.get_labels()[train_ds.indices]

    train_sampler = BalancedSampler(train_labels)


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=num_workers, pin_memory=True) # set shuffle to False when using sampler

    sampled_labels = []
    for _, labels in train_loader:
        sampled_labels.extend(labels.cpu().numpy())
    
    print(f"Sampled Label Distribution in Training Set: {np.bincount(sampled_labels)}", flush=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    input_dim = dataset.file_meta[0]["num_features"]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = LogisticRegression(input_dims=input_dim, output_dims=1)
    model = DANNModel(input_dim=input_dim)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # loss_fn = torch.nn.BCEWithLogitsLoss() # use for logistic regression
    loss_fn = torch.nn.BCELoss()

    # training 
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            y_pred = model(X).view(-1)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{name}] Epoch {epoch+1}, Loss = {total_loss:.4f}")

    model.eval()
    all_preds = []
    all_labels = []

    # evaluation 
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(X).view(-1)
            preds = torch.sigmoid(logits).cpu().numpy()  # Apply sigmoid here
            labels = y.cpu().numpy()
            # preds = model(X).view(-1).cpu().numpy()
            # labels = y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    auroc = roc_auc_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    return fpr, tpr, auroc


# chromosomes = ['chr22', 'chrY', 'chrX']
chromosomes = ['chr21', 'chr22']
# chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 
#                 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 
#                 'chr20', 'chr21', 'chr22', 'chrX', 'chrY'] 

forzoi_dir = "/media/walt/asdalvi/results/predictions/npy_preds"
cadd_dir = '/media/walt/asdalvi/resources/CADD_v7_subsetted/'
label_dir = "/media/walt/asdalvi/resources/labels"

output_fig = "/home/ashdalvi/forzoi_test/figures/model_AUROC.png"



plt.figure(figsize=(8, 6))

# Forzoi only
start = time.time()
forzoi_dataset = PredictionsDataset(chromosomes, forzoi_dir, cadd_dir, label_dir, input_type="forzoi")
fpr_forzoi, tpr_forzoi, auroc_forzoi = train_and_evaluate_model(chromosomes, label_dir, forzoi_dataset, "Forzoi")
plt.plot(fpr_forzoi, tpr_forzoi, label=f"Forzoi AUROC = {auroc_forzoi:.4f}", color='g')

# CADD only
cadd_dataset = PredictionsDataset(chromosomes, forzoi_dir, cadd_dir, label_dir, input_type="cadd")
fpr_cadd, tpr_cadd, auroc_cadd = train_and_evaluate_model(chromosomes, label_dir, cadd_dataset, "CADD")
plt.plot(fpr_cadd, tpr_cadd, label=f"CADD AUROC = {auroc_cadd:.4f}", color='r')

# Forzoi + CADD
combo_dataset = PredictionsDataset(chromosomes, forzoi_dir, cadd_dir, label_dir, input_type="combo")
fpr_combo, tpr_combo, auroc_combo = train_and_evaluate_model(chromosomes, label_dir, combo_dataset, "Forzoi+CADD")
plt.plot(fpr_combo, tpr_combo, label=f"Combo AUROC = {auroc_combo:.4f}", color='b')
end = time.time()

total_time = (end-start)
print(f'took {total_time:.5f} sec')

# Final ROC plot
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Across Models")
plt.legend()
plt.grid(True)
plt.savefig(output_fig)
print(f"Saved ROC comparison plot to {output_fig}")



