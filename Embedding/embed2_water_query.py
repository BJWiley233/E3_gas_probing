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
from e3nn.io import CartesianTensor
import math
import pandas as pd 
import MDAnalysis as mda
import gc

device='cuda'
top ='equil5.gro'
t = "fit2.dcd"
# traj = md.load(t, top=top)
u = mda.Universe(top, t)
print("traj loaded", flush=True)
from rdkit import Chem
mol = Chem.MolFromPDBFile("../Sim9/an1_water.pdb", removeHs=False)
Chem.SanitizeMol(mol)

def build_complete_edge_index(N, device):
    idx = torch.arange(N, device=device)
    i, j = torch.meshgrid(idx, idx, indexing="ij")

    mask = i != j  # no self-loops
    edge_index = torch.stack([i[mask], j[mask]], dim=0)
    return edge_index


from torch_geometric.data import Data
def extract_point_cloud(atom_matrix, positions, center):
    """
    Formats the already-cropped atom features and positions into a Data object.
    """
    if positions.shape[0] == 0:
        return None  # Skip empty cubes

    # Center coordinates relative to cube center
    # add a query node which will have element number 0 and gas number [2 for O2, 3 for CO2, 4 for XXX, ...]
    centered_positions = positions - center
    atom_matrix = torch.vstack([atom_matrix, torch.zeros((1, atom_matrix.shape[1]), dtype=atom_matrix.dtype, device=atom_matrix.device)])
    centered_positions = torch.vstack([centered_positions, torch.zeros((1, centered_positions.shape[1]), dtype=centered_positions.dtype, device=centered_positions.device)])

    return Data(x=atom_matrix, pos=centered_positions)

traj = md.load("cat_fit.dcd", top='an1.gro')
xyz = torch.tensor(traj.xyz) * 10

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "Fe": 5, "Mg":6, "Na":7, "Cl":8}
t = "../Sim9/an1_water.pdb"
traj = md.load_frame(t, index=0,top=top)
gas='CO2'
metal='FE'
gas2 = traj.topology.select('resname %s' % gas)
residue_ref = np.array([traj.topology.atom(ind).residue.resSeq for ind in gas2])
FE = traj.topology.select('resname Fe2p')
residue_sel_un = np.unique(residue_ref) # gas
residue_sel_un

print('LEN FE', FE)
residue_sel_un = np.unique(residue_ref) # gas
residue_sel_un
nogas = np.setdiff1d(range(0,traj.xyz.shape[1]),gas2)
rnames = np.array([traj.topology.atom(ind).residue.name for ind in range(len(u.atoms))])
rindex = np.array([traj.topology.atom(ind).residue.resSeq for ind in range(len(u.atoms))])
anames = np.array([traj.topology.atom(ind).element.symbol for ind in range(len(u.atoms))])
anames2=np.array([traj.topology.atom(i).name for i in range(len(u.atoms))])
anums = [ele2num[a] if a != 'VS' else ele2num[a][metal] for a in anames]
rnames2 = np.array([traj.topology.atom(ind).residue for ind in range(len(u.atoms))])

# rnames = np.array([traj.topology.atom(ind).residue.name for ind in nogas])
# rindex = np.array([traj.topology.atom(ind).residue.resSeq for ind in nogas])
# anames = np.array([traj.topology.atom(ind).element.symbol for ind in nogas])


cat = np.where(anames=='Fe')[0]
print(cat)

device = torch.device("cuda")
# device = torch.device("cpu")
# protein_coords_traj = xyz[:,nogas,:]
rs = int(len(residue_ref)/len(residue_sel_un))
gas_atoms=gas2.reshape(len(residue_sel_un),rs)

dioxygen_coords_ave = xyz[:,gas2,:].reshape(-1,len(residue_sel_un),rs,3).mean(axis=2)
d=torch.cdist(dioxygen_coords_ave, xyz[:,cat,:])
# diox = np.where(d < 6.0)[1]
# frames = np.where(d < 6.0)[0]

types_array_atom = torch.zeros((len(nogas)+len(gas2), (len(ele2num))))
for i, t in enumerate(anums):
    types_array_atom[i,t] = 1.0
# types_array_atom.shape
# types_array_atom[gas2,ele2num['O']] = 1
types_array_atom = types_array_atom.to(device)

path_dict = {
    'P1':'P1', 
    'P_main (PmR)':'PmR', 
    'P_main (mid)':'mid', 
    'P_reverse':'P_reverse', 
    'P_main (PmL)':'PmL',
       'P3':'P3'
}

# DO THIS ONCE
global_bonds = []  # (a1, a2, order)

for bond in mol.GetBonds():
    a1 = bond.GetBeginAtomIdx()
    a2 = bond.GetEndAtomIdx()

    if a1 in gas_atoms or a2 in gas_atoms:
        continue

    bt = bond.GetBondType()
    order = (
        1 if bt == Chem.rdchem.BondType.SINGLE else
        2 if bt == Chem.rdchem.BondType.DOUBLE else
        3 if bt == Chem.rdchem.BondType.AROMATIC else
        1
    )

    global_bonds.append((a1, a2, order))
    global_bonds.append((a2, a1, order))

# gas bonds
for a0, a1, a2 in gas_atoms:
    global_bonds.append((a0, a1, 2))
    global_bonds.append((a1, a0, 2))
    global_bonds.append((a0, a2, 2))
    global_bonds.append((a2, a0, 2))




predicted_paths = pd.read_csv("/home/bw973/Documents/PHD2/other_gases2.csv")
df = predicted_paths[(predicted_paths.FF=='CO2') & (predicted_paths.Sim==7) & (predicted_paths.prop=='counts') & (predicted_paths.molecules==100)]
# df.iloc[:,[0,1,2,3,4,5,6,7,8,254]]

time=0
for i, row in df.iloc[0:,].iterrows():
    gas_resid=row['Gas_Resid']
    gas_idx = np.where(residue_sel_un==gas_resid)[0][0]
    protein_cofactors = np.setdiff1d(np.arange(0,xyz.shape[1]), gas_atoms[gas_idx])
    print(protein_cofactors.shape)
    
    start = int(row['range'].split('-')[0])
    end = int(row['range'].split('-')[1])
    time += (end-start+1)

time


path_data = {
    'P1':[], 
    'P_main (PmR)':[], 
    'P_main (mid)':[], 
    'P_reverse':[], 
    'P_main (PmL)':[],
       'P3':[]
}

# Map from global RDKit atom idx → local 0..N-1
import itertools




def build_edges_and_attrs_fast(inside_indices, edge_index_cache):
    device = inside_indices.device
    N = inside_indices.shape[0]

    # global → local
    atom_to_local = {int(a): i for i, a in enumerate(inside_indices.tolist())}

    # edge_index is reused
    edge_index = edge_index_cache

    # initialize all nonbonded = 0
    edge_attr = torch.zeros(
        (edge_index.shape[1], 1),
        dtype=torch.long,
        device=device
    )

    # overwrite bonded edges only
    # map (i,j) → edge position once
    edge_pos = {
        (int(i), int(j)): k
        for k, (i, j) in enumerate(edge_index.t().tolist())
    }

    for a1, a2, order in global_bonds:
        if a1 in atom_to_local and a2 in atom_to_local:
            i = atom_to_local[a1]
            j = atom_to_local[a2]
            edge_attr[edge_pos[(i, j)], 0] = order

    return edge_index, edge_attr


def sinusoidal_embedding(frame_idx: torch.Tensor, dim: int = 16):
    """
    frame_idx: tensor of shape [N] with normalized frame values in [0,1]
    dim: embedding dimension (should be even)
    Returns: tensor of shape [N, dim]
    """
    device = frame_idx.device
    N = frame_idx.size(0)
    pe = torch.zeros(N, dim, device=device)

    position = frame_idx.unsqueeze(1)  # [N, 1]
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))  # [dim/2]

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.to(device)  # [N, dim]


def embed_event_pos(path, gas_resid, gas_idx, in_row, start, end, protein_cofactors,global_to_local,direction,coords_all):
    pos_embedding = []
    device='cuda'
    end2=end-start
    # print(start, end)
    frames = torch.arange(0, end2, dtype=torch.float32)  # [0,1,...,N-1]
    frame_norm = frames / (end2 - 0)    
    frame_emb = sinusoidal_embedding(frame_norm.to(device), dim=16)
    print(frame_emb)
    print("frame_emb.shape", frame_emb.shape, flush=True)
    
    # protein_coords_traj = xyz[:,protein_cofactors,:]

    if direction == 'reverse':
        frame_indices = reversed(range(end-start))
    else:
        frame_indices = range(end-start)

    for idx in frame_indices:
        c=start+idx
        print(idx, c)
        coords = coords_all[idx]
        mean_gas = coords[gas2].reshape(len(residue_sel_un),rs,3).mean(axis=1).to(device)

        atom_coords = coords[protein_cofactors].to(device)
        # atom_coords = protein_coords_traj[c].to(device)
        name="frame_%d_gas_%d_sequence_%d_path_%s_in_%d_pos" % (c, gas_resid, idx, path, in_row)
        gas_indices = gas_atoms[gas_idx]
        gas_coords = coords[gas_indices]

        for cen, center in enumerate([mean_gas[gas_idx]]):
        # center = mean_gas[gas_idx]
            print("Center %d:" % (cen+1), center)
            dist_FE = torch.norm(center - atom_coords[cat[0],:])
        

            radius = 6.0

            # Compute distances of all atoms to perturbed point
            dists = torch.norm(atom_coords - center, dim=1)

            # Indices of atoms inside sphere
            # shape is -2 of all
            inside_indices = protein_cofactors[torch.where(dists <= radius)[0]]
            local_inside_indices = torch.where(dists <= radius)[0]
            # Ensure inside_indices is always a 1D tensor
            if torch.is_tensor(inside_indices) and inside_indices.ndim == 0:
                inside_indices = inside_indices.unsqueeze(0)

            if isinstance(inside_indices, (int, np.integer)):
                inside_indices = torch.tensor([inside_indices], device=device)

            if inside_indices.numel() == 0:
                dddd = torch.cdist(center.unsqueeze(0), atom_coords)
                print('too far start or end', dddd.min(), flush=True)
                continue

            # local_inside_indices = np.array([global_to_local[int(g)] for g in inside_indices]) # might not even need this if its the same as local_inside_indices = torch.where(dists <= radius)[0] which it is!
            # print(local_inside_indices)
            # print(inside_indices)
            
            print("\tpos sequence %d of %d; frame %d" % (idx,(end-start), c), "len indices:", len(local_inside_indices), flush=True)

            atom_matrix=types_array_atom[inside_indices] # this should be global/inside_indices since we get the indices from protein_cofactors which already skips the current O2 and atom_matrix is all atoms
            # i.e. protein_cofactors[3824:3828] gives [3824, 3827, 3828, 3829]
            positions=atom_coords[local_inside_indices]
            node_directions = positions - atom_coords[cat[0],:]
            node_distances = torch.norm(node_directions, dim=1)
            local_pos_normalized = (positions - center)/6  # shape [N, 3]
            
            # atom_matrix_arr.append(atom_matrix)
            # positions_arr.append(positions)
            test = extract_point_cloud(atom_matrix, positions, center)
            test.name = name
            # test.x=torch.column_stack([test.x, torch.tensor(frame_emb[idx]).repeat(len(test.x))])
            frame_vector = frame_emb[idx]          # [16]
            frame_vector = frame_vector.unsqueeze(0)           # [1, 16]
            frame_vector = frame_vector.expand(len(test.x), -1)  # [N, 16]
            test.x = torch.cat([test.x, frame_vector], dim=1)    # [N, 6 + 16]
            
            N=len(inside_indices)
            edge_index_cache = build_complete_edge_index(N, 'cuda')

            test.edge_index, test.edge_attr = build_edges_and_attrs_fast(inside_indices, edge_index_cache)
            test.center = center
            test.node_attr = local_pos_normalized
            test.node_directions = node_directions
            test.node_distances = node_distances
            test.inside_indices=inside_indices
            test.local_inside_indices=local_inside_indices
            test.distance = dist_FE.expand(len(inside_indices))
            # print("SHAPE", test.x.shape, "CENTER", test.center)
            pos_embedding.append(test.cpu())
    
    return pos_embedding

def embed_event_neg(path, gas_resid, gas_idx, in_row, start, end, protein_cofactors, global_to_local, direction,coords_all, n_candidates=512):
    neg_embedding = []
    device='cuda'
    end2=end-start
    frames = torch.arange(0, end2, dtype=torch.float32)  # [0,1,...,N-1]
    frame_norm = frames / (end2 - 0)    
    frame_emb = sinusoidal_embedding(frame_norm.to(device), dim=16)
    print(frame_emb)

    if direction == 'reverse':
        frame_indices = reversed(range(end-start))
    else:
        frame_indices = range(end-start)

 
    for idx in frame_indices:
        c = start+idx
        coords = coords_all[idx]
        # coords = torch.tensor(u.atoms.positions.copy())
        mean_gas = coords[gas2].reshape(len(residue_sel_un),rs,3).mean(axis=1).to(device)
        atom_coords = coords[protein_cofactors].to(device)  # should this be coords[nogas]???
        # mean_gas=xyz[c][gas2].reshape(len(residue_sel_un),rs,3).mean(axis=1).to(device)
        # atom_coords = protein_coords_traj[c].to(device)
        name="frame_%d_gas_%d_sequence_%d_path_%s_in_%d_neg" % (c, gas_resid, idx, path, in_row)
        
        center = mean_gas[gas_idx]
        gas_indices = gas_atoms[gas_idx]
        gas_coords = coords[gas_indices]

        # min_dist = 2.75
        min_dists_list = [2.25, 2.5,  2.65, 2.75, 3.0]
        candidate_schedule = [5000, 5000, 5000, 10000, 30000]
        max_dist = 4.

        for w, (min_dist, n_candidates) in enumerate(zip(min_dists_list, candidate_schedule)):

            # directions = torch.randn(n_candidates, 3, device=device)
            # directions /= directions.norm(dim=1, keepdim=True)

            # steps = torch.rand(n_candidates, 1, device=device) * (8.0 - 1.0) + 1.0
            # candidates = center + directions * steps
            candidates = center + torch.randn(n_candidates, 3, device=device)
            translations = candidates - center
            cand_center = candidates
            cand_C1 = gas_coords[0].unsqueeze(0) + translations
            cand_O1 = gas_coords[1].unsqueeze(0) + translations
            cand_O2 = gas_coords[2].unsqueeze(0) + translations

            

            # --- enforce radial constraint (1–7 Å from center) ---
            radial_dist = torch.norm(candidates - center, dim=1)
            radial_mask = (radial_dist >= 1.0) & (radial_dist <= 7.0)

            # dists = torch.cdist(candidates, atom_coords)
            # this is more being at least >= min distance from any system atoms
            min_center = torch.cdist(cand_center, atom_coords).min(dim=1).values # also  needs to be at least 1 ang from any pos O2 atom 
            min_C1 = torch.cdist(cand_C1, atom_coords).min(dim=1).values # also  needs to be at least 1 ang from any pos CO2 atom
            min_O1 = torch.cdist(cand_O1, atom_coords).min(dim=1).values # also  needs to be at least 1 ang from any pos CO2 atom
            min_O2 = torch.cdist(cand_O2, atom_coords).min(dim=1).values # also  needs to be at least 1 ang from any pos CO2 atom

            # this is more being at least 1-7 A distance from any gas atoms
            min_center2 = torch.cdist(cand_center, gas_coords).min(dim=1).values
            min_C1_2 = torch.cdist(cand_C1, gas_coords).min(dim=1).values
            min_O1_2 = torch.cdist(cand_O1, gas_coords).min(dim=1).values
            min_O2_2 = torch.cdist(cand_O2, gas_coords).min(dim=1).values

            mins = torch.stack([min_center, min_C1, min_O1, min_O2], dim=1) # not within min dist systems atoms
            mins2 = torch.stack([min_center2, min_C1_2, min_O1_2, min_O2_2], dim=1) # at least 1 angstrom from every pos gas atom
            mask = radial_mask & (mins.min(dim=1).values >= min_dist) & (mins2.min(dim=1).values >= 1.0)

            # min_d = dists.min(dim=1).values
            # near_any = (dists <= max_dist).any(dim=1)
            # atom_mask = (min_d >= min_dist)

            # mask = (min_d >= min_dist) & near_any
            # mask = radial_mask & atom_mask
            all_candidates = candidates[mask]
            

            if len(all_candidates) == 0:
                print(
                    f"No candidate for min_dist={min_dist:.2f} frame={c}",
                    flush=True
                )
                continue   # ← ONLY skip this tier
            else:
                perturbed = all_candidates[0]

            translation = perturbed - center
            perturbed_gas_coords = gas_coords + translation
            
            for cen, perturb, in zip([center], 
                                       [perturbed]):
             
                pdist = torch.norm(perturb-cen)
                print("pdist, cen, perturb:", pdist.item(), cen, perturb)
        
            
                # if good_2_cont:
                dist_FE = torch.norm(perturb - atom_coords[cat[0],:])
                radius = 6.0

                # Compute distances of all atoms to perturbed point
                dists = torch.norm(atom_coords - perturb, dim=1)
                
                # Indices of atoms inside sphere
                inside_indices = protein_cofactors[torch.where(dists <= radius)[0]]
                local_inside_indices = torch.where(dists <= radius)[0]

                # Ensure inside_indices is always a 1D tensor
                if torch.is_tensor(inside_indices) and inside_indices.ndim == 0:
                    inside_indices = inside_indices.unsqueeze(0)

                if isinstance(inside_indices, (int, np.integer)):
                    inside_indices = torch.tensor([inside_indices], device=device)

                if inside_indices.numel() == 0:
                    dddd = torch.cdist(center.unsqueeze(0), atom_coords)
                    print('too far start or end', dddd.min(), flush=True)
                    continue
                
                # local_inside_indices = np.array([global_to_local[int(g)] for g in inside_indices])

                print("\tneg sequence %d (w=%d) of %d; frame %d; dist %f" % (idx, w,(end-start), c, pdist.item()), "len indices:", len(local_inside_indices), flush=True)
            
                atom_matrix=types_array_atom[inside_indices] # this should be global/inside_indices since we get the indices from protein_cofactors which already skips the current O2 and atom_matrix is all atoms
                # i.e. protein_cofactors[3824:3828] gives [3824, 3827, 3828, 3829]
                positions=atom_coords[local_inside_indices]
                node_directions = positions - atom_coords[cat[0],:]
                node_distances = torch.norm(node_directions, dim=1)
                local_pos_normalized = (positions - perturb)/6  # shape [N, 3]
                
                # atom_matrix_arr.append(atom_matrix)
                # positions_arr.append(positions)
                test = extract_point_cloud(atom_matrix, positions, perturbed)
                test.name = name
                # test.x=torch.column_stack([test.x, torch.tensor(frame_emb[idx]).repeat(len(test.x))])
                frame_vector = frame_emb[idx]          # [16]
                frame_vector = frame_vector.unsqueeze(0)           # [1, 16]
                frame_vector = frame_vector.expand(len(test.x), -1)  # [N, 16]
                test.x = torch.cat([test.x, frame_vector], dim=1)    # [N, 6 + 16]
                
                N=len(inside_indices)
                edge_index_cache = build_complete_edge_index(N, 'cuda')
                test.edge_index, test.edge_attr = build_edges_and_attrs_fast(inside_indices, edge_index_cache)
                test.center = perturb
                test.perturbed_from = cen
                test.node_attr = local_pos_normalized
                test.node_directions = node_directions
                test.node_distances = node_distances
                test.inside_indices=inside_indices
                test.local_inside_indices=local_inside_indices
                test.distance = dist_FE.expand(len(inside_indices))
                neg_embedding.append(test.cpu())
        # del coords, atom_coords, dists, candidates, directions, steps
        gc.collect()
        # else:
            
    
    return neg_embedding


pos_embedding_arr = []
neg_embedding_arr = []
pos_embedding_arr_r = []
neg_embedding_arr_r = []
count=0
for j, (i, row) in enumerate(df.iloc[0:,].iterrows()):
    print("ROW:", j, flush=True)
    gas_resid=row['Gas_Resid']
    gas_idx = np.where(residue_sel_un==gas_resid)[0][0]
    start = int(row['range'].split('-')[0])
    end = int(row['range'].split('-')[1])
    path = row['PATH']
    dists = d[start:end+1,gas_idx,0]
    first = torch.where(dists < 6)[0][0]
    # instead of going to just first one < 6 we could get more positives if we go to last one < 6
    last = torch.where(dists < 6)[0][-1]
    middle = int((first+last)/2)
    protein_cofactors = torch.tensor(np.setdiff1d(np.arange(0,len(u.atoms)), gas_atoms[gas_idx])).to(device)
    global_to_local = {int(g): i for i, g in enumerate(protein_cofactors)}
    print(len(protein_cofactors), gas_resid, row['range'])
    coords_all = torch.stack([
        torch.from_numpy(u.trajectory[i].positions)
        for i in range(start-2, start+middle+1)
    ]).float().to(device)
    # print(global_to_local)

    if row['in_row']:
        pos_embedding = embed_event_pos(path, gas_resid, gas_idx, row['in_row'], start-2,  start+middle+1, protein_cofactors, global_to_local, 'forward',coords_all)
        # pos_embedding_arr += [p.cpu() for p in pos_embedding]
        # pos_embedding = [p.cpu() for p in pos_embedding]
        torch.save(pos_embedding, "tmp/pos_%d.pt"%count)
        
        neg_embedding = embed_event_neg(path, gas_resid, gas_idx, row['in_row'], start-2,  start+middle+1, protein_cofactors, global_to_local, 'forward',coords_all,n_candidates=2000)
        # neg_embedding = [p.cpu() for p in neg_embedding]
        torch.save(neg_embedding, "tmp/neg_%d.pt"%count)
        
        count+=1
        # exit()

    # else: 
    #     pos_embedding = embed_event_pos(path, gas_resid, gas_idx, row['in_row'], start+first+1, end+1, protein_cofactors, global_to_local, 'reverse')
    #     pos_embedding_arr_r += [p.cpu() for p in pos_embedding]
    #     neg_embedding = embed_event_neg(path, gas_resid, gas_idx, row['in_row'], start+first+1, end+1, protein_cofactors, global_to_local, 'reverse')
    #     neg_embedding_arr_r += [p.cpu() for p in neg_embedding]

# torch.save(pos_embedding_arr, "pos_embedding_inWaterTest.pt")
# torch.save(neg_embedding_arr, "neg_embedding_inWater.pt")
# torch.save(pos_embedding_arr_r, "pos_embedding_out3.pt")
# torch.save(neg_embedding_arr_r, "neg_embedding_out3.pt")