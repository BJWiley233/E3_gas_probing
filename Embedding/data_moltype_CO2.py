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

top ='equil5.gro'
t = "../Sim9/an1_water.pdb"
traj = md.load_frame(t, index=0,top=top)
gas2 = traj.topology.select('resname CO2')
protein_atoms = traj.topology.select('protein or resname AKG Fe2p')
solvent = traj.topology.select('water or resname SOD CLA POT')
n_atoms = traj.topology.n_atoms
atom_labels = torch.tensor(np.full(n_atoms, -1, dtype=np.int8))  # optional default
atom_labels[solvent] = 0
atom_labels[protein_atoms] = 1
atom_labels[gas2] = 3 # for CO2
ele2AtomicNumber = torch.tensor([6, 1, 8, 7, 16, 26, 12, 11, 17])


device='cuda'
pos_embedding_arr = []
pos_lowPerturb_arr = []
neg_embedding_arr_25 = []
neg_embedding_arr_50 = []
neg_embedding_arr_100 = []

n = min([len(pos_embedding_arr), len(pos_lowPerturb_arr), len(neg_embedding_arr_100)])
print("min length:", n, flush=True)
# model2=torch.load("../Sim5/model4_FE2dis_inrWaterReg.pt")

# # 3,6,7??
to_ = 2



for i in [3,5,7]: # 6K for Sim7, 33K for Sim8, 44k for Sim6? already stride 2, 47 for Sim10
    neg_embedding_50 = torch.load("../Sim%d/neg_embedding_inWater_query.pt"%i)[::1]
    print("- Sim%d"%i,len(neg_embedding_50), flush=True)
    for neg in neg_embedding_50:
        neg.Sim = i 
    neg_embedding_arr_50 += neg_embedding_50
print("- Total",len(neg_embedding_arr_50), flush=True)

for neg in neg_embedding_arr_50:
    neg.atoms = "atoms"
    moltype = atom_labels[neg.inside_indices]
    ## one hot to [moltype, atomic_number]
    atomic_number = ele2AtomicNumber[torch.where(neg.x[:,0:9])[1]]
    neg.x2 = torch.stack([moltype, atomic_number], dim=1)
    ## for O2 embedding I need to append it
    neg.x2 = torch.vstack([neg.x2, torch.tensor([[3,0]])]) # query node 3 for CO2 (0 solvent, 1 solute, 2 O2, 3 CO2, 4 Xe?, 5 H?, ...)
    ## the node_attr of center positions needs to be added not Data.pos
    ## as I don't use it for the graph that is premade.  Although the last
    ## Data.pos entry should include [0,0,0] for query node
    neg.node_attr = torch.vstack([neg.node_attr, torch.tensor([[0.,0.,0.]])])
    neg.pos[-1] = torch.tensor([0.,0.,0.])






for i in [3,5,7]: #
    pos_embedding = torch.load("../Sim%d/pos_embedding_inWater_query.pt"%i)[::1]
    print("++ Sim%d"%i,len(pos_embedding), flush=True) #  * 0.25
    for pos in pos_embedding:
        pos.Sim = i
    pos_embedding_arr += pos_embedding
print("++ Total",len(pos_embedding_arr), flush=True)

for pos in pos_embedding_arr:
    pos.perturbed_from = pos.center
    pos.atoms = "atoms"
    moltype = atom_labels[pos.inside_indices]
    atomic_number = ele2AtomicNumber[torch.where(pos.x[:,0:9])[1]]
    pos.x2 = torch.stack([moltype, atomic_number], dim=1)
    pos.x2 = torch.vstack([pos.x2, torch.tensor([[3,0]])]) # query node
    pos.node_attr = torch.vstack([pos.node_attr, torch.tensor([[0.,0.,0.]])])
    pos.pos[-1] = torch.tensor([0.,0.,0.])



torch.save(pos_embedding_arr, "pos_PHD2_CO2_query.pt") # 
torch.save(neg_embedding_arr_50, "neg_PHD2_CO2_query.pt")
exit()