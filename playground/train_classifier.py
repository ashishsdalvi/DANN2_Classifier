import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split 
import os 
import time
import h5py
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# from lance.dataset import LanceDataset




class PredictionsDataset(Dataset):
    """
    Prediction data are stored in .npy files, and labels are stored in separate .npy files.
    Each chromosome has its own feature and label file.
    """
    def __init__(self, chromosomes, npy_feature_dir, csv_label_dir):
        self.chromosomes = chromosomes
        self.npy_feature_dir = npy_feature_dir
        self.csv_label_dir = csv_label_dir

        self.file_meta = []  # metadata for each chromosome
        self.cumulative_idx = [0]  # global row index boundaries
        self._features = {}  # lazy-loaded mmap npy files
        self._labels = {}    # lazy-loaded mmap npy files

        for chrom in chromosomes:
            feature_path = os.path.join(npy_feature_dir, f"{chrom}_l2.npy")
            label_path = os.path.join(csv_label_dir, f"{chrom}_labels.csv")

            features = np.load(feature_path, mmap_mode="r")
            # labels = np.load(label_path, mmap_mode="r")
            labels = pd.read_csv(label_path)

            assert features.shape[0] == len(labels), f"Mismatch in {chrom}"

            self.file_meta.append({
                "chrom": chrom,
                "feature_path": feature_path,
                "label_path": label_path,
                "length": features.shape[0],
                "num_features": features.shape[1]
            })

            self.cumulative_idx.append(self.cumulative_idx[-1] + features.shape[0])

    def __len__(self):
        return self.cumulative_idx[-1]

    def _load_data(self, chrom):
        if chrom not in self._features:
            self._features[chrom] = np.load(os.path.join(self.npy_feature_dir, f"{chrom}_l2.npy"), mmap_mode="r")
        if chrom not in self._labels:
            df = pd.read_csv(os.path.join(self.csv_label_dir, f"{chrom}_labels.csv"))
            self._labels[chrom] = df["variant_label"].to_numpy(dtype=np.float32)
        return self._features[chrom], self._labels[chrom]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_idx, idx, side='right') - 1
        meta = self.file_meta[file_idx]
        local_idx = idx - self.cumulative_idx[file_idx]
        chrom = meta["chrom"]

        features, labels = self._load_data(chrom)
        feature_row = features[local_idx]
        label = labels[local_idx]

        return torch.tensor(feature_row, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dims, output_dims)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred 


chromosomes = ['chrY', 'chr22']
# chromosomes = ['chrY']

# h5_dir = "/media/walt/asdalvi/results/predictions/h5_preds"
lance_dir = "/media/walt/asdalvi/results/predictions/npy_preds"
csv_dir = "/media/walt/asdalvi/resources/labels"

full_dataset = PredictionsDataset(chromosomes, lance_dir, csv_dir)

# Split into train/test globally
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

######################### Model Training #########################

input_dim = full_dataset.file_meta[0]["num_features"]
model = LogisticRegression(input_dims=input_dim, output_dims=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCELoss()

for epoch in range(1):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(X).view(-1)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss = {total_loss:.4f}")

########################## Evaluate globally on test set #########################
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    start = time.time()
    for X, y in test_loader:
        preds = model(X).view(-1).numpy()
        labels = y.numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    elapsed = time.time() - start
    print(f"Training took {elapsed:.2f} s", flush=True)

########################## Compute and plot AUROC #########################
auroc = roc_auc_score(all_labels, all_preds)
fpr, tpr, _ = roc_curve(all_labels, all_preds)

plt.plot(fpr, tpr, label=f'Global AUROC = {auroc:.4f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Global ROC Across Chromosomes")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('/home/ashdalvi/forzoi_test/figures/model_performance.png')





# chromosomes = ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 
#                     'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 
#                     'chr20', 'chr21', 'chr22', 'chrX', 'chrY'] # no chr 1 





# class PredictionsDataset(Dataset):
#     """
#     Prediction data are stored in h5 and labels are stored in csv. 
#     """
#     def __init__(self, chromosomes, h5_dir, csv_labels_dir):
#         self.chromosomes = chromosomes
#         self.h5_dir = h5_dir
#         self.csv_labels_dir = csv_labels_dir

#         self.file_meta = [] # stores meta info for each chr 
#         self.cumulative_idx = [0] # a global idx for looking up values 
#         self._files = {}

#         for chromosome in chromosomes:
#             h5_path = os.path.join(h5_dir, f"{chromosome}_l2.h5") 
#             csv_label_path = os.path.join(csv_labels_dir, f"{chromosome}_labels.csv")

#             labels = pd.read_csv(csv_label_path)

#             with h5py.File(h5_path, "r") as f:
#                 columns = sorted(f.keys())
#                 length = f[columns[0]].shape[0]

#             assert length == len(labels), f"Mismatch in {chromosome}" # Ensure label dims and pred dims match
#             self.file_meta.append({
#                 "chrom": chromosome,
#                 "h5_path": h5_path,
#                 "label_df": labels,
#                 "columns": columns,
#                 "length": length
#             })
#             self.cumulative_idx.append(self.cumulative_idx[-1] + length)


#     def __len__(self):
#         return self.cumulative_idx[-1]

#     def _get_file(self, chromosome):
#         if chromosome not in self._files:
#             self._files[chromosome] = h5py.File(os.path.join(self.h5_dir, f"{chromosome}_l2.h5"), "r")
#         return self._files[chromosome]

#     def __getitem__(self, idx):
#         file_idx = np.searchsorted(self.cumulative_idx, idx, side='right') - 1 
#         meta_data = self.file_meta[file_idx]    # get chromosome's metadata
#         local_idx = idx - self.cumulative_idx[file_idx]
#         chrom = meta_data["chrom"]  # get chrom name 

#         h5f = self._get_file(chrom) # get chrom file 
#         features = np.array([h5f[col][local_idx] for col in meta_data["columns"]], dtype=np.float32)
#         # label = meta_data["label_df"].iloc[local_idx].values[0]
#         label = float(meta_data["label_df"].iloc[local_idx]["variant_label"])


#         return torch.tensor(features), torch.tensor(label, dtype=torch.float32)



# class PredictionsDataset(Dataset):
#     """
#     Prediction data are stored in Lance (.lance) files, and labels are stored in CSV.
#     """
#     def __init__(self, chromosomes, lance_dir, csv_labels_dir):
#         self.chromosomes = chromosomes
#         self.lance_dir = lance_dir
#         self.csv_labels_dir = csv_labels_dir

#         self.file_meta = []  # metadata for each chromosome
#         self.cumulative_idx = [0]  # global row index boundaries
#         self._tables = {}  # cached Arrow tables (to avoid repeated to_table())

#         for chrom in chromosomes:
#             lance_path = os.path.join(lance_dir, f"{chrom}_l2.lance")
#             label_path = os.path.join(csv_labels_dir, f"{chrom}_labels.csv")

#             # Read labels
#             labels = pd.read_csv(label_path)
#             label_array = labels["variant_label"].astype("float32").to_numpy()

#             # Read Lance table once
#             table = LanceDataset(lance_path).to_table()
#             columns = [col for col in table.schema.names if col != "variant"]
#             length = table.num_rows

#             assert length == len(labels), f"Mismatch in {chrom}"  # sanity check

#             self.file_meta.append({
#                 "chrom": chrom,
#                 "table": table,
#                 "columns": columns,
#                 "labels": label_array,
#                 "length": length
#             })

#             self.cumulative_idx.append(self.cumulative_idx[-1] + length)

#     def __len__(self):
#         return self.cumulative_idx[-1]

#     def __getitem__(self, idx):
#         file_idx = np.searchsorted(self.cumulative_idx, idx, side='right') - 1
#         meta = self.file_meta[file_idx]
#         local_idx = idx - self.cumulative_idx[file_idx]

#         row = meta["table"].slice(local_idx, 1).to_pydict()
#         features = np.array([row[col][0] for col in meta["columns"]], dtype=np.float32)
#         label = meta["labels"][local_idx]

#         return torch.tensor(features), torch.tensor(label, dtype=torch.float32)