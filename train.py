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
from torch.optim import SGD, Adam, Optimizer, NAdam, AdamW
import math
from torch.nn.init import kaiming_uniform_
from torch_geometric.transforms import ToDevice
import scipy
import sys
import random

# self.mp(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device='cuda'
print(device, flush=True)
positives = []
positives_PCO4 = []
negatives_PCO4 = []

# model2 = torch.load("model9_FE2dis_inrWaterPHD2Bath17.2.pt")

negatives=torch.load("neg_PHD2_O2IF_50_query.pt", weights_only=False)[0:16344]
print("len PHD2 O2 -", len(negatives), flush=True)
negatives += torch.load("neg_PHD2_O2IF_100_query.pt", weights_only=False)[::2][0:32688] #  
print("len PHD2 O2 -", len(negatives), flush=True)
negatives += torch.load("/data/pompei/bw973/Oxygenases/PHD2/PHD2_CO2/Bundle/Sim5/neg_PHD2_CO2_query.pt", weights_only=False)[0:49032]
print("len PHD2 O2/CO2 - ", len(negatives), flush=True)

positives=torch.load("pos_PHD2_O2IF_50_query.pt", weights_only=False)[0:16344]
print("len PHD2 O2 +", len(positives), flush=True)
positives += torch.load("pos_PHD2_O2IF_100_query.pt", weights_only=False)[0:32688] #  
print("len PHD2 O2 +", len(positives), flush=True)
positives += torch.load("/data/pompei/bw973/Oxygenases/PHD2/PHD2_CO2/Bundle/Sim5/pos_PHD2_CO2_query.pt", weights_only=False)
print("len PHD2 O2/CO2 + ", len(positives), flush=True)


from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit



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
        
n = len(positives)
n_neg = n 
n_pos = n 

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
    test_size=0.25,
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


train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
#train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


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
from torch.optim.lr_scheduler import StepLR

# model = torch.load('/data/pompei/bw973/Oxygenases/PHD2/PHD2_50_O2IF/Bundle/Sim4/predict/models_10_Both/output_ep_4_bs_24_lr_0.0003_opt_adamw_inw_xavier_neigh_45_nodes_85_mul_30_lay_3_lmax_2.pt')

# irreps_node_attr = Irreps("17x0e + 1x1o")
irreps_node_attr = Irreps("1x0e + 1x1o")
# 2. Build new model with 7 scalar features
layers_=3
model2 = NetworkForAGraphWithAttributes(
    # irreps_node_input=Irreps("1x0e + 1x1o"),      # updated input
    irreps_node_input=Irreps("2x0e"),
    irreps_node_attr=irreps_node_attr,     # keep the same
    irreps_edge_attr="1x0e",     # keep the same
    irreps_node_output="1x0e",
    max_radius=6.0,
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



model2 = model2.to(device)
#current_train_idx = train_idx.clone()

#mags = []
optimizer=torch.optim.AdamW(model2.parameters(), lr=0.0005, weight_decay=5e-4)
#scheduler = StepLR(optimizer, step_size=1, gamma=0.75)
criterion = torch.nn.BCEWithLogitsLoss()

print(optimizer)

train_losses_global = []
val_losses_global = []
train_probs_global = []
val_probs_global = []
train_labels_global = []
val_labels_global = []
model_suffix=17.2

#val_probs_global=torch.load("val_probs_global%d.pt" % model_suffix)
#val_losses_global=torch.load("val_losses_global%d.pt" % model_suffix)
#val_labels_global=torch.load("val_labels_global%d.pt" % model_suffix)
#train_probs_global=torch.load("train_probs_global%d.pt" % model_suffix)
#train_losses_global=torch.load("train_losses_global%d.pt" % model_suffix)
#train_labels_global=torch.load("train_labels_global%d.pt" % model_suffix)

for epoch in range(0,30):
    print("model%d_inWaterPHD2_CO2_O2_%.1f.pt"% (epoch, model_suffix), flush=True)
    easy_positives = []
    hard_positives = []
    easy_negatives = []
    hard_negatives = []
    #sampler = torch.utils.data.SubsetRandomSampler(current_train_idx)
    #train_loader = DataLoader(full_dataset, batch_size=24, sampler=sampler)

    epoch_losses_val = []
    epoch_probs_val = []
    epoch_labels_val = []
    epoch_losses_train = []
    epoch_probs_train = []
    epoch_labels_train = []
    
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

        batch = batch.to(device)
        epoch_labels_train.append(labels)
        labels = labels.float()
        labels = labels.to(device)

        num_pos = (labels == 1).sum()
        num_neg = (labels == 0).sum()

        # Avoid division by zero
        if num_pos > 0:
            pos_weight = torch.tensor([num_neg / num_pos], device=labels.device)
        else:
            pos_weight = torch.tensor([1.0], device=labels.device)  # neutral weight

        #perturb_dists, batch.new_node_attr_with_perturbation =  perturb_node_attr2(batch, perturb_prob=0.1, sigma=0.08)
        #mags.append(perturb_dists)

        distance = torch.norm(batch.node_attr[:, :3], dim=1, keepdim=True)  # [N, 1]

        # 2. Convert Cartesian vectors to irreps vector
        x = CartesianTensor("i")
        vector_irrep = x.from_cartesian(batch.node_attr[:, :3])  # [N, 3] This doesn't change any outcome

        # 3. Concatenate scalar + vector as node_attr tensor
        atom_type_onehot = batch.x[:, 0:9]
        mol_atom_type = batch.x2
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
            "x": mol_atom_type,
            "node_attr": node_attr, 
            "edge_index": batch.edge_index,
            "edge_attr": batch.edge_attr,
            "pos": batch.pos,  # if needed in preprocess
        }
        # print("SHAPES:", node_attr.shape, mol_atom_type.shape)

        optimizer.zero_grad()
        # outputs = model(node_input, node_attr, edge_index, edge_attr).squeeze()
        outputs = model2(data)
            
        logits = outputs.squeeze(-1)
        probs = torch.sigmoid(logits)
        epoch_probs_train.append(probs.detach())
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

        epsilon = 0.02
        labels2 = labels.float() * (1 - epsilon) + epsilon / 2
        #labels2 = labels * 0.9 + 0.05
        loss = criterion(logits, labels2)
       
        epoch_losses_train.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
        optimizer.step()

        # Predictions
        preds = (probs > 0.5)

        # Convert labels to bool
        #labels_bool = labels.bool()
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
        neg_total_train += neg_mask.sum().item()
        neg_correct_train += ((~preds & neg_mask)).sum().item()
        if neg_total_train > 0:
            neg_acc = neg_correct_train / neg_total_train
        else:
            neg_acc = float("nan")

        # print every 20 batches
        if (batch_idx + 1) % 50 == 0:
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

    epoch_probs_train = torch.cat(epoch_probs_train).cpu().numpy()
    epoch_labels_train = torch.cat(epoch_labels_train).cpu().numpy()
    epoch_losses_train = np.array(epoch_losses_train)
    train_probs_global.append(epoch_probs_train)
    train_losses_global.append(epoch_losses_train)
    train_labels_global.append(epoch_labels_train)
    

    train_accuracy = correct_train / total_train
    pos_acc_train = pos_correct_train / pos_total_train if pos_total_train > 0 else 0
    neg_acc_train = neg_correct_train / neg_total_train if neg_total_train > 0 else 0
    torch.save(easy_positives, "easy_positives%.1f.pt" % model_suffix)
    torch.save(hard_positives, "hard_positives%.1f.pt" % model_suffix)
    torch.save(easy_negatives, "easy_negatives%.1f.pt" % model_suffix)
    torch.save(hard_negatives, "hard_negatives%.1f.pt" % model_suffix)

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
        total_val = 0
        correct_val = 0
        pos_total_val = 0
        pos_correct_val = 0

        neg_total_val = 0
        neg_correct_val = 0

        
        for batch_idx, (data_list, labels, idx) in enumerate(val_loader):
            # print(labels.sum())
            # If batch_size > 1, data_list will be a list of Data objects
            # Batch them for PyG model input:
            if isinstance(data_list, list):
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(data_list)
            else:
                batch = data_list  # if batch_size=1, it might already be a Data object

            batch = batch.to(device)
            labels = labels.to(device)
            epoch_labels_val.append(labels)
            num_pos = (labels == 1).sum()
            num_neg = (labels == 0).sum()
               
            distance = torch.norm(batch.node_attr[:, :3], dim=1, keepdim=True)  # [N, 1]

            # 2. Convert Cartesian vectors to irreps vector
            x = CartesianTensor("i")
            vector_irrep = x.from_cartesian(batch.node_attr[:, :3])  # [N, 3]

            # 3. Concatenate scalar + vector as node_attr tensor
            atom_type_onehot = batch.x[:, 0:9]
            mol_atom_type = batch.x2
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
                "x": mol_atom_type,
                "node_attr": node_attr, 
                "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "pos": batch.pos,  # if needed in preprocess
            }
           
            # outputs = model(node_input, node_attr, edge_index, edge_attr).squeeze()
            outputs = model2(data)
            logits = outputs.squeeze(-1)
            probs = torch.sigmoid(logits)
            epoch_probs_val.append(probs.detach())
            
            labels2 = labels.float() * (1 - epsilon) + epsilon / 2
            #labels2 = labels * 0.9 + 0.05
            loss = criterion(logits, labels2)
            epoch_losses_val.append(loss.item())

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
            correct_val += (preds == labels_bool).sum().item()
            total_val += labels.size(0)

            
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
            pos_total_val += pos_mask.sum().item()
            # pos_correct = ((preds & pos_mask)).sum().item()
            pos_correct_val += ((preds & pos_mask)).sum().item()
            if num_pos > 0:
                pos_acc = pos_correct_val / pos_total_val
            else:
                pos_acc = float("nan")

            # negative (label=0)
            neg_mask = ~labels_bool
            neg_total_val += neg_mask.sum().item()
            neg_correct_val += ((~preds & neg_mask)).sum().item()
            if neg_total_val > 0:
                neg_acc = neg_correct_val / neg_total_val
            else:
                neg_acc = float("nan")

            # print every 20 batches
            if (batch_idx + 1) % 20 == 0:
                val_acc = correct_val / total_val
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx+1} | "
                    f"Val Acc: {val_acc:.4f} | "
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

        epoch_probs_val = torch.cat(epoch_probs_val).cpu().numpy()
        epoch_labels_val = torch.cat(epoch_labels_val).cpu().numpy()
        epoch_losses_val = np.array(epoch_losses_val)
        val_probs_global.append(epoch_probs_val)
        val_losses_global.append(epoch_losses_val)
        val_labels_global.append(epoch_labels_val)

        val_accuracy = correct_val / total_val
        pos_acc_val = pos_correct_val / pos_total_val if pos_total_val > 0 else 0
        neg_acc_val = neg_correct_val / neg_total_val if neg_total_val > 0 else 0
        #torch.save(mags,"mags_epoch%d_%d.pt" % (epoch,model_suffix))
        torch.save(model2, "model%d_inWaterPHD2_CO2_O2_%.1f.pt" % (epoch,model_suffix))
        print('\tval_accuracy:', val_accuracy, flush=True)
        print('\tpos val_accuracy:', pos_acc_val, flush=True)
        print('\tneg val_accuracy:', neg_acc_val, flush=True)
        
        #scheduler.step()
        torch.save(val_probs_global, "val_probs_global%.1f.pt" % model_suffix)
        torch.save(val_losses_global, "val_losses_global%.1f.pt" % model_suffix)
        torch.save(val_labels_global, "val_labels_global%.1f.pt" % model_suffix)
        torch.save(train_probs_global, "train_probs_global%.1f.pt" % model_suffix)
        torch.save(train_losses_global, "train_losses_global%.1f.pt" % model_suffix)
        torch.save(train_labels_global, "train_labels_global%.1f.pt" % model_suffix)
    
# train_allNewSolventPHD2_CO2_O2_21.4.log 