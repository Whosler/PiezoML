import json
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import from_networkx, add_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, NNConv
import torch.nn.functional as F
from mendeleev import element
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
import copy
import pickle
import csv


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=data_list[0].x.shape[1], out_channels=256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.conv2 = GCNConv(in_channels=256, out_channels=128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.conv3 = GCNConv(in_channels=128, out_channels=64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.conv4 = GCNConv(in_channels=64, out_channels=32)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.lin = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        # Add self-loops and update edge_weight
        edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, fill_value=1.0)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x, edge_index, edge_weight=edge_weight)
        x = self.bn4(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return torch.sigmoid(x)

def train():
    n_high = 0.25
    n_low = 0.75
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).view(-1)
        pos_weight = torch.tensor([n_low / n_high]).to(device)  # weight for positive class
        loss = F.binary_cross_entropy(out, data.y.float(), weight=pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    print(total_loss / len(train_loader.dataset))
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_ids = []
    for data in loader:
        data = data.to(device)
        out = model(data).view(-1)
        all_probs.extend(out.cpu().detach().numpy())
        pred = (out > 0.5).long()
        all_labels.extend(data.y.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_ids.extend(data.compound_id)
    return roc_auc_score(all_labels, all_probs), classification_report(all_labels, all_preds), all_labels, all_probs, all_ids

with open('dij_data_list_wg_0.25.pkl', 'rb') as f:
    data_list = pickle.load(f)

# Split data into training and testing sets
train_data, test_data = train_test_split(data_list, test_size=0.2, stratify=[data.y.item() for data in data_list])
# Split data into majority and minority
high_data = [data for data in data_list if data.y.item() == 1]
low_data = [data for data in data_list if data.y.item() == 0]

# Upsample minority class
high_data_upsampled = resample(high_data, replace=True, n_samples=len(low_data), random_state=41)
balanced_data = low_data + high_data_upsampled

# Convert to PyTorch DataLoader
train_loader = DataLoader(balanced_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

best_roc_auc = 0
best_model_state = None
best_report = None
best_labels = None
best_probs = None
best_ids = None

for epoch in range(1, 501):
    loss = train()
    roc_auc, report, labels, probs, ids = test(test_loader)
    if(roc_auc > best_roc_auc):
        best_roc_auc = roc_auc
        best_report = report
        best_model_state = copy.deepcopy(model.state_dict())
        best_labels = labels
        best_probs = probs
        best_ids = ids
    print(f'Epoch: {epoch:03d}, roc_auc: {roc_auc}')
    print(report)

print(f'best roc auc: {best_roc_auc}')
print(best_report)

torch.save(best_model_state, 'best_wg_dij_model_0.25.pth')

# Save predictions to CSV
with open('dij_median_predictions_wg_0.25.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['compound_id', 'Actual Label', 'Probability'])
    for i in range(len(best_ids)):
        writer.writerow([best_ids[i], best_labels[i], best_probs[i]])

# Plot and save ROC curve
fpr, tpr, _ = roc_curve(best_labels, best_probs)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {best_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DIJ Weighted GCN Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('wg_dij_roc_auc_curve_0.25.png')
plt.show()
