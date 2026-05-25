import torch    
import torch_geometric.transforms as T
import mdtraj as md
import os
import torch 
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv, knn_graph, Linear
from torch.optim import SGD, Adam, Optimizer
import math
from torch.nn.init import kaiming_uniform_
from torch_geometric.transforms import ToDevice
import scipy
import sys
import random

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device='cuda'
positives = []
positives_PCO4 = []
negatives_PCO4 = []
# model2=torch.load("../Sim5/model10_FE2dis_inrWater42.pt")
# # model2 = model2.to('cuda')

# # 3,6,7??
for i in [2,3,5,6]:
    negatives_PCO4 += torch.load("/media/bw973/Seagate Hub1/Oxygenases/PCO_MUTANTS/ARABI/PCO4_WT_100_O2IF/Sim%d/neg_embedding_inWater_test.pt"%i)[::6]
    print("PCO- Sim%d"%i,len(negatives_PCO4), flush=True)

negatives=torch.load("neg_PHD2_O2IF_25.pt")[0:21572]
print("len", len(negatives), flush=True)
negatives += torch.load("neg_PHD2_O2IF_50.pt")[0:10764] #  144355 and 111732 PHD2 256087 [0:250000] 
print("len", len(negatives), flush=True)
negatives += torch.load("neg_PHD2_O2IF_100.pt")[::2][0:36364]
print("len", len(negatives), flush=True)

negatives += negatives_PCO4[0:30000]
print("len", len(negatives), flush=True)


positives += torch.load("pos_PHD2_O2IF.pt")[0:] # 47379-(133 three dup events Sim7 + 179 four dup events Sim5) = 47379+1384 = PCO4 and 63108-1384 =  PHD2 # rm duplicate Sim2/Event 37633-40546 #
print('len(positives)', len(positives), flush=True)
# positives += torch.load("pos_lowPeturb_mse2.pt")[50000:195600] # 145600 of 44178 PCO4 and 101422 PHD2
for i in [2,3,5,6,7]:
    positives_PCO4 += torch.load("/media/bw973/Seagate Hub1/Oxygenases/PCO_MUTANTS/ARABI/PCO4_WT_100_O2IF/Sim%d/pos_embedding_inWater_test.pt"%i)[::2]
    print("PCO++ Sim%d"%i,len(positives_PCO4), flush=True)
positives += positives_PCO4[0:30000]
print('len(positives)', len(positives), flush=True)



# for i in [1,5]:
#     pos_embedding_arr += torch.load("/media/bw973/Seagate Hub/Oxygenases/PCO_MUTANTS/ARABI/PCO4_WT_50_O2IF/Sim%d/pos_embedding_inWater_test.pt"%i)[::to_]
#     print("+ Sim%d"%i,len(pos_embedding_arr), flush=True)
# for i in [1,5]:
#     neg_embedding_arr += torch.load("/media/bw973/Seagate Hub/Oxygenases/PCO_MUTANTS/ARABI/PCO4_WT_50_O2IF/Sim%d/neg_embedding_inWater_test.pt"%i)[::to_]
#     print("- Sim%d"%i,len(neg_embedding_arr), flush=True)

# print(f"Len + {len(pos_embedding_arr)}; Len -  {len(neg_embedding_arr)}; ratio {len(neg_embedding_arr)/len(pos_embedding_arr)}", flush=True)
# # neg_use = int(len(neg_embedding_arr)/len(pos_embedding_arr))
# neg_use = ((len(neg_embedding_arr))/len(pos_embedding_arr))
# print("Negatives times", neg_use, flush=True)



# torch.save(pos_embedding_arr, "pos_embedding_inWater_stride%d_PCObal.pt"%to_) # 62581
# torch.save(neg_embedding_arr, "neg_embedding_inWater_stride%d_PCObal.pt"%to_) # ~ 67247


# mutants??
# for i in [4,5]:
#     pos_embedding_arr += torch.load("/media/bw973/Seagate Hub/Oxygenases/PCO_MUTANTS/ARABI/PCO4_T153A_100_O2IF/Sim%d/pos_embedding_inWater_test.pt"%i)[::10]
#     print("+ Sim%d"%i,len(pos_embedding_arr), flush=True) 
#     pos_embedding_arr += torch.load("/media/bw973/Seagate Hub/Oxygenases/PCO_MUTANTS/ARABI/PCO4_T153C_100_O2IF/Sim%d/pos_embedding_inWater_test.pt"%i)[::10]
#     print("+ Sim%d"%i,len(pos_embedding_arr), flush=True)
#     # pos_embedding_arr += torch.load("/media/bw973/Seagate Hub/Oxygenases/PCO_MUTANTS/ARABI/PCO4_K142R_100_O2IF/Sim%d/pos_embedding_inWater_test.pt"%i)[::4]
#     # print("+ Sim%d"%i,len(pos_embedding_arr), flush=True)

#     neg_embedding_arr += torch.load("/media/bw973/Seagate Hub/Oxygenases/PCO_MUTANTS/ARABI/PCO4_T153A_100_O2IF/Sim%d/neg_embedding_inWater_test.pt"%i)[::20]
#     print("- Sim%d"%i,len(neg_embedding_arr), flush=True) 
#     neg_embedding_arr += torch.load("/media/bw973/Seagate Hub/Oxygenases/PCO_MUTANTS/ARABI/PCO4_T153C_100_O2IF/Sim%d/neg_embedding_inWater_test.pt"%i)[::20]
#     print("- Sim%d"%i,len(neg_embedding_arr), flush=True)
#     # neg_embedding_arr += torch.load("/media/bw973/Seagate Hub/Oxygenases/PCO_MUTANTS/ARABI/PCO4_K142R_100_O2IF/Sim%d/neg_embedding_inWater_test.pt"%i)[::4]
#     # print("- Sim%d"%i,len(neg_embedding_arr), flush=True)
    
print(f"Len + {len(positives)}; Len -  {len(negatives)}; ratio {len(negatives)/len(positives)}", flush=True)
# neg_use = int(len(neg_embedding_arr)/len(pos_embedding_arr))


from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit

for pos in positives:
    pos.perturbed_from = pos.center
    pos.atoms = "atoms"
for neg in negatives:
    neg.atoms = "atoms"


import random
def shuffle_data_list_(data_list, seed=None):
    if seed is not None:
        rnd = random.Random(seed)  # independent RNG
        rnd.shuffle(data_list)     # shuffle in place
    else:
        random.shuffle(data_list)  # default RNG

class PointCloudDataset(Dataset):
    def __init__(self, positives, negatives):
        self.positives = positives
        self.negatives = negatives
        # self.n = min(len(positives), len(negatives))
        # self.total = 2 * self.n
        self.n_pos = len(positives)
        self.n_neg = 1 * self.n_pos  # we expect negatives to be at least 2× positives
        
        # dataset size = pos + neg
        self.total = self.n_pos + self.n_neg

    def __len__(self):
        return self.total

    def __getitem__(self, idx):

        if idx < self.n_pos:
            return self.positives[idx], 1.0, idx
        else:
            neg_idx = idx - self.n_pos
            return self.negatives[neg_idx], 0.0, idx
        
shuffle_data_list_(positives, seed=42)
shuffle_data_list_(negatives, seed=42)

# torch.save(pos_embedding_arr[0:150000], "pos_embedding_inWater_stride%d_PCO_small2.pt"%to_) # 62581
# torch.save(neg_embedding_arr[0:150000], "neg_embedding_inWater_stride%d_PCO_small2.pt"%to_) # ~ 67247

n = len(positives)
# n = 1000


# n_pos=150000
# n_pos=int(n_pos)
# n_pos=50133 # changed this from 25000 to 15000 and this made epoch 18... (19 w/ 0-index)
n_neg = n # 301368
n_pos = n # changed this from 25000 to 15000 and this made epoch 18... (19 w/ 0-index)
# n_neg = n_pos


# positives = pos_embedding_arr#[:n_pos]
# negatives = neg_embedding_arr#[:n_neg]

labels = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])
full_dataset = PointCloudDataset(positives[0:n], negatives[:n])
indices = torch.arange(n_pos + n_neg)

# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
sss1 = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.4,
    random_state=42
)
train_idx, temp_idx = next(sss1.split(indices, labels))
sss2 = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=42
)
val_rel_idx, test_rel_idx = next(
    sss2.split(temp_idx, labels[temp_idx])
)

val_idx = temp_idx[val_rel_idx]
test_idx = temp_idx[test_rel_idx]

train_idx = torch.from_numpy(train_idx)
val_idx   = torch.from_numpy(val_idx)
test_idx   = torch.from_numpy(test_idx)

for name, idx in [
    ("Train", train_idx),
    ("Val", val_idx),
    ("Test", test_idx),
]:
    split_labels = labels[idx]
    n_pos = (split_labels == 1).sum().item()
    n_neg = (split_labels == 0).sum().item()

    print(f"{name}: +{n_pos} / -{n_neg}", flush=True)


train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset  = Subset(full_dataset, test_idx)

# Materialize only the subset samples
test_subset_data = [test_dataset[i] for i in range(len(test_dataset))]
torch.save(test_subset_data, "test_dataset_sameReplicatesPHD2OnlyPCO4again.pt")

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# def perturb_node_attr(batch, radius=6.0, sigma=0.08):
#     coords = batch.node_attr[:, :3]   # [total_nodes, 3]

#     # batch.batch maps each node -> graph index
#     graph_ids = batch.batch           # [total_nodes]

#     num_graphs = batch.num_graphs

#     # random direction
#     shift = torch.randn(num_graphs, 3, device=coords.device)
#     shift = shift / shift.norm(dim=1, keepdim=True)

#     # random magnitude between 0 and 0.25 Å
#     mag = torch.randn(num_graphs, 1, device=coords.device) * sigma

#     # actual center shifts in Å
#     delta_c = shift * mag             # [12, 3]

#     # convert to normalized shift
#     delta_norm = delta_c / radius

#     # apply corresponding shift to each node
#     new_coords = coords - delta_norm[graph_ids]

#     return mag, new_coords

def perturb_node_attr2(
    batch,
    radius=6.0,
    sigma=0.08,
    perturb_prob=0.5,
):
    coords = batch.node_attr[:, :3]   # [total_nodes, 3]

    # node -> graph mapping
    graph_ids = batch.batch

    num_graphs = batch.num_graphs
    device = coords.device

    # random direction
    shift = torch.randn(num_graphs, 3, device=device)
    shift = shift / shift.norm(dim=1, keepdim=True)

    # signed magnitudes in Å
    mag = torch.randn(num_graphs, 1, device=device) * sigma

    # choose which graphs to perturb
    mask = (torch.rand(num_graphs, 1, device=device) < perturb_prob)

    # zero perturbation for unselected graphs
    mag = mag * mask

    # actual center shifts in Å
    delta_c = shift * mag

    # convert to normalized shift
    delta_norm = delta_c / radius

    # apply graph-wise perturbation
    new_coords = coords - delta_norm[graph_ids]

    return mag[mask], new_coords


# orig_model = torch.load("...pt")
from e3nn.io import CartesianTensor
from torch_geometric.loader import DataLoader
from copy import deepcopy
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Gate
from e3nn.o3 import Irreps
from e3nn.nn.models.v2106.gate_points_networks import SimpleNetwork, NetworkForAGraphWithAttributes

model = torch.load('/data/pompei/bw973/Oxygenases/PHD2/PHD2_50_O2IF/Bundle/Sim4/predict/models_10_Both/output_ep_4_bs_24_lr_0.0003_opt_adamw_inw_xavier_neigh_45_nodes_85_mul_30_lay_3_lmax_2.pt')

# irreps_node_attr = Irreps("17x0e + 1x1o")
irreps_node_attr = Irreps("1x0e + 1x1o")
# 2. Build new model with 7 scalar features
layers_=3
model2 = NetworkForAGraphWithAttributes(
    # irreps_node_input=Irreps("1x0e + 1x1o"),      # updated input
    irreps_node_input=Irreps("9x0e"),
    irreps_node_attr=irreps_node_attr,     # keep the same
    irreps_edge_attr=model.irreps_edge_attr,     # keep the same
    irreps_node_output=model.irreps_node_output,
    max_radius=5.0,
    num_neighbors=65,
    num_nodes=80,
    mul=50,
    layers=layers_,
    lmax=2,
    pool_nodes=True,
)
with torch.no_grad():
    for param in model2.mp.layers[layers_].alpha.parameters():
        param.uniform_(-1e-3, 1e-3)


# model2=torch.load("../Sim5/model10_FE2dis_inrWater42.pt")
model2 = model2.to('cuda')
current_train_idx = train_idx.clone()

mags = []
optimizer=torch.optim.Adam(model2.parameters(), lr=0.0005)
print(optimizer)
for epoch in range(0,30):
    print("../Sim5/model%d_FE2dis_inrWaterPHD2_5.pt"%epoch, flush=True)
    easy_positives = []
    hard_positives = []
    easy_negatives = []
    hard_negatives = []
    sampler = torch.utils.data.SubsetRandomSampler(current_train_idx)
    train_loader = DataLoader(full_dataset, batch_size=24, sampler=sampler)

    model2.train()
    total_train = 0
    correct_train = 0
    pos_total_train = 0
    pos_correct_train = 0
    neg_total_train = 0
    neg_correct_train = 0

    cum_pos_logit_sum = 0.0
    cum_neg_logit_sum = 0.0
    cum_pos_prob_sum = 0.0
    cum_neg_prob_sum = 0.0

    cum_pos_count = 0
    cum_neg_count = 0

    for batch_idx, (data_list, labels, idx) in enumerate(train_loader):
        # If batch_size > 1, data_list will be a list of Data objects
        # Batch them for PyG model input:
        # print(labels.sum())
        if isinstance(data_list, list):
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(data_list)
        else:
            batch = data_list  # if batch_size=1, it might already be a Data object

        batch = batch.to('cuda')
        labels = labels.float()
        labels = labels.to('cuda')

        num_pos = (labels == 1).sum()
        num_neg = (labels == 0).sum()

        # Avoid division by zero
        if num_pos > 0:
            pos_weight = torch.tensor([num_neg / num_pos], device=labels.device)
        else:
            pos_weight = torch.tensor([1.0], device=labels.device)  # neutral weight

        perturb_dists, batch.new_node_attr_with_perturbation =  perturb_node_attr2(batch, perturb_prob=0.8, sigma=0.1)
        mags.append(perturb_dists)

        distance = torch.norm(batch.new_node_attr_with_perturbation[:, :3], dim=1, keepdim=True)  # [N, 1]

        # 2. Convert Cartesian vectors to irreps vector
        x = CartesianTensor("i")
        vector_irrep = x.from_cartesian(batch.new_node_attr_with_perturbation[:, :3])  # [N, 3] This doesn't change any outcome

        # 3. Concatenate scalar + vector as node_attr tensor
        atom_type_onehot = batch.x[:, 0:9]
        frame_emb = batch.x[:, 9:] 
        # node_attr = torch.cat([distance, vector_irrep, frame_emb], dim=1)
        node_attr = torch.cat([
            distance,            # 1 scalar (0e)
            # atom_type_onehot,    # 6 scalars (0e)
            # frame_emb,           # k scalars (0e)
            vector_irrep,        # 3-vector (1o)
        ], dim=1)
        # node_input = torch.ones((batch.num_nodes, 1), device=batch.x.device)
        # node_input = torch.cat([
        #     # batch.distance.unsqueeze(-1), # 1 scalar (0e) distance of center of graph to iron
        #     batch.node_distances.unsqueeze(-1),    # 1 scalar (0e) distance to iron for each atom
        #     F.normalize(batch.node_directions, p=2, dim=1),        # 3-vector (1o) direction to iron for each atom
        # ], dim=1)

        data = {
            "batch": batch.batch,
            # "x": batch.x[:,0:6], # atom type
            # "frame_emb": frame,
            "x": atom_type_onehot,
            "node_attr": node_attr, 
            "edge_index": batch.edge_index,
            "edge_attr": batch.edge_attr,
            "pos": batch.pos,  # if needed in preprocess
        }

        optimizer.zero_grad()
        # outputs = model(node_input, node_attr, edge_index, edge_attr).squeeze()
        outputs = model2(data)
            
        logits = outputs.squeeze(-1)
        probs = torch.sigmoid(logits)
        labels_bool = labels.bool()
        pos_mask = labels_bool
        neg_mask = ~labels_bool

        # ---- accumulate ----
        cum_pos_logit_sum += logits[pos_mask].sum().item()
        cum_neg_logit_sum += logits[neg_mask].sum().item()

        cum_pos_prob_sum += probs[pos_mask].sum().item()
        cum_neg_prob_sum += probs[neg_mask].sum().item()

        cum_pos_count += pos_mask.sum().item()
        cum_neg_count += neg_mask.sum().item()
        

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(outputs.squeeze(-1), labels)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
        optimizer.step()

        # Predictions
        preds = (probs > 0.5)

        # Convert labels to bool
        labels_bool = labels.bool()
        easy_pos = (probs > 0.8) & labels_bool
        hard_pos = (probs <= 0.2) & labels_bool
        easy_neg = (probs < 0.2) & ~labels_bool
        hard_neg = (probs >= 0.8) & ~labels_bool
        easy_positives.extend(idx[easy_pos.cpu()].tolist())
        hard_positives.extend(idx[hard_pos.cpu()].tolist())
        easy_negatives.extend(idx[easy_neg.cpu()].tolist())
        hard_negatives.extend(idx[hard_neg.cpu()].tolist())

        # ---- metrics ----
        correct_train += (preds == labels_bool).sum().item()
        total_train += labels.size(0)

        # positive (label=1)
        pos_mask = labels_bool
        pos_total_train += pos_mask.sum().item()

        # pos_correct = ((preds & pos_mask)).sum().item()
        pos_correct_train += ((preds & pos_mask)).sum().item()
        if num_pos > 0:
            pos_acc = pos_correct_train / pos_total_train
        else:
            pos_acc = float("nan")

        # negative (label=0)
        neg_mask = ~labels_bool
        neg_total_train += neg_mask.sum().item()
        neg_correct_train += ((~preds & neg_mask)).sum().item()
        if neg_total_train > 0:
            neg_acc = neg_correct_train / neg_total_train
        else:
            neg_acc = float("nan")

        # print every 20 batches
        if (batch_idx + 1) % 20 == 0:
            train_acc = correct_train / total_train
            print(
                f"Epoch {epoch+1}, Batch {batch_idx+1} | "
                f"Acc: {train_acc:.4f} | "
                f"PosAcc: {pos_acc:.4f} | "
                f"NegAcc: {neg_acc:.4f}",
                flush=True
            )
            print("len(easy_positives):",len(easy_positives), ", len(easy_negatives):",len(easy_negatives))
            print("len(hard_positives):",len(hard_positives), ", len(hard_negatives):",len(hard_negatives))
            print()
   
             # cumulative means
            mean_pos_prob = cum_pos_prob_sum / cum_pos_count if cum_pos_count > 0 else float("nan")
            mean_neg_prob = cum_neg_prob_sum / cum_neg_count if cum_neg_count > 0 else float("nan")

            mean_pos_logit = cum_pos_logit_sum / cum_pos_count if cum_pos_count > 0 else float("nan")
            mean_neg_logit = cum_neg_logit_sum / cum_neg_count if cum_neg_count > 0 else float("nan")
            print(f"CUM Mean pos prob: {mean_pos_prob:.3f}, Mean neg prob: {mean_neg_prob:.3f}")
            print(f"CUM Mean pos logit: {mean_pos_logit:.3f}, Mean neg logit: {mean_neg_logit:.3f}")

            print()

        torch.cuda.empty_cache()
    train_accuracy = correct_train / total_train
    pos_acc_train = pos_correct_train / pos_total_train if pos_total_train > 0 else 0
    neg_acc_train = neg_correct_train / neg_total_train if neg_total_train > 0 else 0
    torch.save(easy_positives, "easy_positives.pt")
    torch.save(hard_positives, "hard_positives.pt")
    torch.save(easy_negatives, "easy_negatives.pt")
    torch.save(hard_negatives, "hard_negatives.pt")

    print('\ttrain_accuracy:', train_accuracy, flush=True)
    print('\tpos train_accuracy:', pos_acc_train, flush=True)
    print('\tneg train_accuracy:', neg_acc_train, flush=True)

    cum_pos_logit_sum = 0.0
    cum_neg_logit_sum = 0.0
    cum_pos_prob_sum = 0.0
    cum_neg_prob_sum = 0.0

    cum_pos_count = 0
    cum_neg_count = 0

    with torch.no_grad():
        easy_positives = []
        hard_positives = []
        easy_negatives = []
        hard_negatives = []

        model2.eval()
        total_train = 0
        correct_train = 0
        pos_total_train = 0
        pos_correct_train = 0

        neg_total_train = 0
        neg_correct_train = 0

        
        for batch_idx, (data_list, labels, idx) in enumerate(val_loader):
            # print(labels.sum())
            # If batch_size > 1, data_list will be a list of Data objects
            # Batch them for PyG model input:
            if isinstance(data_list, list):
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(data_list)
            else:
                batch = data_list  # if batch_size=1, it might already be a Data object

            batch = batch.to('cuda')
            labels = labels.to('cuda')

            num_pos = (labels == 1).sum()
            num_neg = (labels == 0).sum()
               
            distance = torch.norm(batch.node_attr[:, :3], dim=1, keepdim=True)  # [N, 1]

            # 2. Convert Cartesian vectors to irreps vector
            x = CartesianTensor("i")
            vector_irrep = x.from_cartesian(batch.node_attr[:, :3])  # [N, 3]

            # 3. Concatenate scalar + vector as node_attr tensor
            atom_type_onehot = batch.x[:, 0:9]
            frame_emb = batch.x[:, 9:] 
            # node_attr = torch.cat([distance, vector_irrep, frame_emb], dim=1)
            node_attr = torch.cat([
                distance,            # 1 scalar (0e)
                # atom_type_onehot,    # 6 scalars (0e)
                # frame_emb,           # k scalars (0e)
                vector_irrep,        # 3-vector (1o)
            ], dim=1)
            # node_input = torch.ones((batch.num_nodes, 1), device=batch.x.device)
            # node_input = torch.cat([
            #     # batch.distance.unsqueeze(-1), # 1 scalar (0e) distance of center of graph to iron
            #     batch.node_distances.unsqueeze(-1),    # 1 scalar (0e) distance to iron for each atom
            #     F.normalize(batch.node_directions, p=2, dim=1),        # 3-vector (1o) direction to iron for each atom
            # ], dim=1)

            data = {
                "batch": batch.batch,
                # "x": batch.x[:,0:6], # atom type
                # "frame_emb": frame,
                "x": atom_type_onehot,
                "node_attr": node_attr, 
                "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "pos": batch.pos,  # if needed in preprocess
            }
           
            # outputs = model(node_input, node_attr, edge_index, edge_attr).squeeze()
            outputs = model2(data)
            outs = torch.sigmoid(outputs.squeeze(-1))
            preds = (outs > 0.5)

            # Convert labels to bool
            labels_bool = labels.bool()
            easy_pos = (outs > 0.8) & labels_bool
            hard_pos = (outs <= 0.2) & labels_bool
            easy_neg = (outs < 0.2) & ~labels_bool
            hard_neg = (outs >= 0.8) & ~labels_bool
            easy_positives.extend(idx[easy_pos.cpu()].tolist())
            hard_positives.extend(idx[hard_pos.cpu()].tolist())
            easy_negatives.extend(idx[easy_neg.cpu()].tolist())
            hard_negatives.extend(idx[hard_neg.cpu()].tolist())

            # ---- metrics ----
            correct_train += (preds == labels_bool).sum().item()
            total_train += labels.size(0)

            logits = outputs.squeeze(-1)
            probs = torch.sigmoid(logits)

            labels_bool = labels.bool()
            pos_mask = labels_bool
            neg_mask = ~labels_bool

            # ---- accumulate ----
            cum_pos_logit_sum += logits[pos_mask].sum().item()
            cum_neg_logit_sum += logits[neg_mask].sum().item()

            cum_pos_prob_sum += probs[pos_mask].sum().item()
            cum_neg_prob_sum += probs[neg_mask].sum().item()

            cum_pos_count += pos_mask.sum().item()
            cum_neg_count += neg_mask.sum().item()

            
            pos_mask = labels_bool
            pos_total_train += pos_mask.sum().item()
            # pos_correct = ((preds & pos_mask)).sum().item()
            pos_correct_train += ((preds & pos_mask)).sum().item()
            if num_pos > 0:
                pos_acc = pos_correct_train / pos_total_train
            else:
                pos_acc = float("nan")

            # negative (label=0)
            neg_mask = ~labels_bool
            neg_total_train += neg_mask.sum().item()
            neg_correct_train += ((~preds & neg_mask)).sum().item()
            if neg_total_train > 0:
                neg_acc = neg_correct_train / neg_total_train
            else:
                neg_acc = float("nan")

            # print every 20 batches
            if (batch_idx + 1) % 20 == 0:
                train_acc = correct_train / total_train
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx+1} | "
                    f"Val Acc: {train_acc:.4f} | "
                    f"Val PosAcc: {pos_acc:.4f} | "
                    f"Val NegAcc: {neg_acc:.4f}",
                    flush=True
                )
                print("len(easy_positives):",len(easy_positives), ", len(easy_negatives):",len(easy_negatives))
                print("len(hard_positives):",len(hard_positives), ", len(hard_negatives):",len(hard_negatives))

                 # cumulative means
                mean_pos_prob = cum_pos_prob_sum / cum_pos_count if cum_pos_count > 0 else float("nan")
                mean_neg_prob = cum_neg_prob_sum / cum_neg_count if cum_neg_count > 0 else float("nan")

                mean_pos_logit = cum_pos_logit_sum / cum_pos_count if cum_pos_count > 0 else float("nan")
                mean_neg_logit = cum_neg_logit_sum / cum_neg_count if cum_neg_count > 0 else float("nan")

                print(f"CUM Mean pos prob: {mean_pos_prob:.3f}, Mean neg prob: {mean_neg_prob:.3f}")
                print(f"CUM Mean pos logit: {mean_pos_logit:.3f}, Mean neg logit: {mean_neg_logit:.3f}")
                print()

            torch.cuda.empty_cache()
        train_accuracy = correct_train / total_train
        pos_acc_val = pos_correct_train / pos_total_train if pos_total_train > 0 else 0
        neg_acc_val = neg_correct_train / neg_total_train if neg_total_train > 0 else 0
        torch.save(mags,"mags%d.pt" % epoch)
        torch.save(model2, "../Sim5/model%d_FE2dis_inrWaterPHD2_5.pt"%epoch)
        print('\tval_accuracy:', train_accuracy, flush=True)
        print('\tpos val_accuracy:', pos_acc_val, flush=True)
        print('\tneg val_accuracy:', neg_acc_val, flush=True)
        # if pos_acc_train > 0.8 and (pos_acc_train - pos_acc_val) > 0.2:
        #     print("Overfit positives so adding noise to positives and negatives")
        #     train_loader = noisy_train_loader
        # elif neg_acc_train > 0.9 and (neg_acc_train - neg_acc_val) > 0.1:
        #     print("Overfit negatives so adding noise to positives and negatives")
        #     train_loader = noisy_train_loader
    
