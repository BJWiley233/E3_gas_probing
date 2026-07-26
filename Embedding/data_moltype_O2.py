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

top ='/data/pompei/bw973/Oxygenases/PHD2/PHD2_50_O2IF/Bundle/Sim5/equil5.gro'
t = "/data/pompei/bw973/Oxygenases/PHD2/PHD2_50_O2IF/Bundle/Sim9/an1_water.pdb"
traj = md.load_frame(t, index=0,top=top)
gas2 = traj.topology.select('resname O2IF')
protein_atoms = traj.topology.select('protein or resname AKG Fe2p')
solvent = traj.topology.select('water or resname SOD CLA POT')
n_atoms = traj.topology.n_atoms
atom_labels = torch.tensor(np.full(n_atoms, -1, dtype=np.int8))  # optional default
atom_labels[solvent] = 0
atom_labels[protein_atoms] = 1
atom_labels[gas2] = 2 # for O2
ele2AtomicNumber = torch.tensor([6, 1, 8, 7, 16, 26, 12, 11, 17])


device='cuda'
pos_embedding_arr = []
pos_lowPerturb_arr = []
neg_embedding_arr_25 = []
neg_embedding_arr_50 = []
neg_embedding_arr_100 = []


for i in [6, 7, 8, 10]: # 6K for Sim7, 33K for Sim8, 44k for Sim6? already stride 2, 47 for Sim10
    neg_embedding_50 = torch.load("/data/pompei/bw973/Oxygenases/PHD2/PHD2_50_O2IF/Bundle/Sim%d/neg_embedding_inWater_test.pt"%i)[::1]
    print("- Sim%d"%i,len(neg_embedding_50), flush=True)
    for neg in neg_embedding_50:
        neg.Sim = i 
    neg_embedding_arr_50 += neg_embedding_50

print("- Total",len(neg_embedding_arr_50), flush=True)

for neg in neg_embedding_arr_50:
    neg.atoms = "atoms"
    # CO2 had an extra query for for .x and .pos, for O2 going to add
    neg.x = torch.vstack([neg.x, torch.zeros((1, neg.x.shape[1]), dtype=neg.x.dtype, device=neg.x.device)])
    moltype = atom_labels[neg.inside_indices]
    atomic_number = ele2AtomicNumber[torch.where(neg.x[:,0:9])[1]]
    neg.x2 = torch.stack([moltype, atomic_number], dim=1)
    ## append it, ele2AtomicNumber[torch.where(neg.x[:,0:9])[1]] won't return anything for all zeros row for query node
    neg.x2 = torch.vstack([neg.x2, torch.tensor([[2,0]])]) # query node 3 for CO2 (0 solvent, 1 solute, 2 O2, 3 CO2, 4 Xe?, 5 H?, ...)
    ## the node_attr of center positions needs to be added not Data.pos
    ## as I don't use it for the graph that is premade.  Although the last
    ## Data.pos entry should include [0,0,0] for query node
    neg.node_attr = torch.vstack([neg.node_attr, torch.tensor([[0.,0.,0.]])])
    # CO2 had an extra query for for .x and .pos, for O2 going to add
    # pos is node_attr * 6 or unnormalized
    neg.pos = torch.vstack([neg.pos, torch.tensor([[0.,0.,0.]])])



# 30K for Sim6, 2K for Sim7,  7K for Sim8, 12K Sim10: total 51K
for i in [6,7,8,10]:  # 2K for Sim7, 32K for Sim6, 9k for Sim8, 19K Sim10?
    pos_embedding = torch.load("/data/pompei/bw973/Oxygenases/PHD2/PHD2_50_O2IF/Bundle/Sim%d/pos_embedding_inWater_test.pt"%i)[::1]
    print("++ Sim%d"%i,len(pos_embedding), flush=True) #  * 0.25 torch.load
    for pos in pos_embedding:
        pos.Sim = i
    pos_embedding_arr += pos_embedding
print("++ Total",len(pos_embedding_arr), flush=True)

for pos in pos_embedding_arr:
    pos.perturbed_from = pos.center
    pos.atoms = "atoms"
    # CO2 had an extra query for for .x and .pos, for O2 going to add
    pos.x = torch.vstack([pos.x, torch.zeros((1, pos.x.shape[1]), dtype=pos.x.dtype, device=pos.x.device)])
    moltype = atom_labels[pos.inside_indices]
    atomic_number = ele2AtomicNumber[torch.where(pos.x[:,0:9])[1]]
    ## append it, ele2AtomicNumber[torch.where(neg.x[:,0:9])[1]] won't return anything for all zeros row for query node
    pos.x2 = torch.stack([moltype, atomic_number], dim=1)
    pos.x2 = torch.vstack([pos.x2, torch.tensor([[2,0]])]) # query node 3 for CO2 (0 solvent, 1 solute, 2 O2, 3 CO2, 4 Xe?, 5 H?, ...)
    ## the node_attr of center positions needs to be added not Data.pos
    ## as I don't use it for the graph that is premade.  Although the last
    ## Data.pos entry should include [0,0,0] for query node
    pos.node_attr = torch.vstack([pos.node_attr, torch.tensor([[0.,0.,0.]])])
    # CO2 had an extra query for for .x and .pos, for O2 going to add
    # pos is node_attr * 6 or unnormalized
    pos.pos = torch.vstack([pos.pos, torch.tensor([[0.,0.,0.]])])


torch.save(pos_embedding_arr, "pos_PHD2_O2IF_50_query.pt") # 
# torch.save(pos_embedding_arr, "3Ppos_PHD2_O2IF_50_2.pt") # 
# torch.save(neg_embedding_arr_25, "neg_PHD2_O2IF_25.pt")
# torch.save(neg_embedding_arr_50, "3Pneg_PHD2_O2IF_50_2.pt")
torch.save(neg_embedding_arr_50, "neg_PHD2_O2IF_50_query.pt") # 
# torch.save(neg_embedding_arr_100, "3Pneg_PHD2_O2IF_100.pt")
exit()

