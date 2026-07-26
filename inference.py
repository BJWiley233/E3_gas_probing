from rdkit import Chem
import torch 
import torch_geometric.transforms as T
import os
import torch 
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
import math
import sys
from e3nn.io import CartesianTensor
import gc
import time

import MDAnalysis as mda
import copy
import warnings

warnings.filterwarnings(
    "ignore",
    message="__array_wrap__ must accept context and return_scalar arguments"
)
warnings.filterwarnings(
    "ignore",
    message=r".*Found no information for attr: *",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Found missing chainIDs. *",
    category=UserWarning,
)
                     # Sim device start end     dir       embedded      no_zip    one_hot     model


# for i in 10; do python Sim${i}_predict_water_HEX.py 1 cuda 600 7000 2 1 1 0 model18_FE2dis_inrWaterPHD2Bath17.9_cpu.pt > Sim${i}_witWaterModel2.log 2>&1 & -- 3007840
# python Sim1_predict_water_HEX.py 1 cuda 600 7000 2 1 0 1 model4_FE2dis_inrWaterPHD2Bath18_cpu.pt > Sim1_witWaterModel3.log 2>&1 & -- 3670767


# python predict_water_HEX.py 4 cuda 600 7000 2 1 1 1 model4_FE2dis_inrWaterPHD2Bath18_cpu.pt > Sim4_witWaterModel2.log 2>&1 & -- 3008061
# python Sim4_predict_water_HEX.py 4 cuda 600 7000 2 1 0 1 model4_FE2dis_inrWaterPHD2Bath18_cpu.pt > Sim4_witWaterModel3.log 2>&1 & -- 3672541


# python predict_water_HEX.py 6 cuda 600 7000 2 1 1 1 model4_FE2dis_inrWaterPHD2Bath18_cpu.pt > Sim6_witWaterModel2.log 2>&1 & -- 3018568
# python Sim6_predict_water_HEX.py 6 cuda 600 7000 3 1 1 0 model22_FE2dis_inrWaterPHD2Bath17.2_cpu.pt > Sim6_witWaterModel3.log 2>&1 & -- 3674950


# python predict_water_HEX.py 9 cuda 600 7000 2 1 1 1 model4_FE2dis_inrWaterPHD2Bath18_cpu.pt > Sim9_witWaterModel2.log 2>&1 & -- 3027710
# python Sim9_predict_water_HEX.py 9 cuda 600 7000 2 1 0 1 model4_FE2dis_inrWaterPHD2Bath18_cpu.pt > Sim9_witWaterModel3.log 2>&1 & -- 3677086


# python predict_water_HEX.py 10 cuda 600 7000 2 1 1 1 model4_FE2dis_inrWaterPHD2Bath18_cpu.pt > Sim10_witWaterModel2.log 2>&1 & -- 3029836
# python Sim10_predict_water_HEX.py 10 cuda 600 7000 2 1 0 1 model4_FE2dis_inrWaterPHD2Bath18_cpu.pt > Sim10_witWaterModel3.log 2>&1 & -- 3679346




data_time = time.time()
device=sys.argv[2]
torch.cuda.is_available()


model_path ='/homes/bw973/simulations/Oxygenases/train/%s'%sys.argv[9]
best=0.8
start_thresh=0.6
print(model_path, flush=True)

from rdkit import Chem
mount='0'
mol = Chem.MolFromPDBFile("/mnt/faster%s/bw973/Oxy/PHD2/Sim4/an1_water.pdb"%mount, removeHs=False)
Chem.SanitizeMol(mol)

top = '/mnt/faster%s/bw973/Oxy/PHD2/Sim4/equil5.gro' % mount
Sim=int(sys.argv[1]) # not 2,3,7
# t = "/mnt/faster%s/bw973/Oxy/PHD2/Sim%d/fit2_small.dcd"%(mount,Sim)
# u = mda.Universe(top, t)


ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "Fe": 5, "Mg":6, "Na":7, "Cl":8}
t = "/mnt/faster%s/bw973/Oxy/PHD2/Sim4/an1_water.pdb" % mount
u = mda.Universe(t)
gas='O2IF'
metal='FE'
gas2 = u.select_atoms(f'resname {gas}')
water = u.select_atoms('resname TIP3')
ignore_starts = u.select_atoms(
    'protein and ((resid 268-284) or (resid 214-232))'
)
print("max ignore_starts", np.max(ignore_starts.resids) if len(ignore_starts) else None)
ignore_starts = ignore_starts.indices
print("water length", len(water))

residue_ref = residue_ref = np.array([a.resid for a in gas2])
FE = u.select_atoms('resname Fe2p')
print('LEN FE', len(FE))
residue_sel_un = np.unique(residue_ref) # gas

gas2 = gas2.indices
water = water.indices
nogas = np.setdiff1d(np.arange(u.atoms.n_atoms), gas2)
nogasnowater = np.setdiff1d(nogas, water)


rnames = np.array([a.resname for a in u.atoms])
rindex = np.array([a.resid for a in u.atoms])
anames = np.array([a.type for a in u.atoms])
anames2 = np.array([a.name for a in u.atoms])

anums = [
    ele2num[a] if a != 'VS' else ele2num[a][metal]
    for a in anames
]
rnames2 = np.array([a.residue for a in u.atoms])

cat = np.where(anames=='Fe')[0]
print(cat)
device = torch.device("cpu")
rs = int(len(residue_ref)/len(residue_sel_un))
gas_atoms=gas2.reshape(len(residue_sel_un),rs)


types_array_atom = torch.zeros((len(nogas)+len(gas2), (len(ele2num))))
for i, t in enumerate(anums):
    types_array_atom[i,t] = 1.0

types_array_atom[gas2,ele2num['O']] = 2
types_array_atom = types_array_atom.to(device)

gas_idx = np.where(residue_sel_un==10000)[0]
protein_cofactors = torch.tensor(np.setdiff1d(np.arange(0,len(u.atoms)), gas_atoms[gas_idx])).to(device)
# protein_cofactors = torch.tensor(np.setdiff1d(np.setdiff1d(np.arange(0,len(u.atoms)), gas_atoms[gas_idx]),water)).to(device)
print("Len protein_cofactors", len(protein_cofactors))
print("Len nogasnowater", len(nogasnowater))
global_to_local = {int(g): i for i, g in enumerate(protein_cofactors)}

# del traj

# t = "../../Sim4/fit2_small.dcd"
# traj = md.load_frame(t, index=0,top="../../Sim4/equil5.gro")
# rnames2 = np.array([traj.topology.atom(ind).residue for ind in protein_cofactors]) # so we skip any O2 but we don't
# anames2=np.array([traj.topology.atom(i).name for i in protein_cofactors])
anames_nogas = anames[nogas]
t = "/mnt/faster%s/bw973/Oxy/PHD2/Sim%d/fit2_small.dcd"%(mount,Sim)
u = mda.Universe(top, t)


protein_atoms = u.select_atoms('protein or resname AKG FE2P')
protein_atoms = protein_atoms.indices
solvent = u.select_atoms('resname TIP3 SOD CLA POT')
solvent = solvent.indices
n_atoms = u.atoms.n_atoms

atom_labels = torch.tensor(np.full(n_atoms, -1, dtype=np.int8))  # optional default
atom_labels[solvent] = 0
atom_labels[protein_atoms] = 1
atom_labels[gas2] = 2
ele2AtomicNumber = torch.tensor([6, 1, 8, 7, 16, 26, 12, 11, 17])


def run_step2(frame_pos, possible_paths, start, step, subpath, model, batch_size=24, one_hot=False): 
    if not isinstance(frame_pos, torch.Tensor):
        frame_pos = torch.tensor(frame_pos)
    distance=torch.norm(frame_pos)
    if distance > 24:
        print("Subpath %s too far away.  Terminating at step %d, frame %d" % (subpath, step, start))
        # possible_paths[0][subpath]["still_valid"] = False
        possible_paths[subpath]["still_valid"] = False
        pos_embedding = None
        results = None
        return pos_embedding, results

    print("Distance",distance)
    torch.cuda.empty_cache()
    color=colors[start % 5]
    print("START:", start)
    print("frame_pos:", frame_pos.numpy())
    print("SUBPATH:", subpath)
    
    u.trajectory[start]
    coords = torch.tensor(u.atoms.positions.copy())
    # print(coords, flush=True)
    protein_coords_traj_use = coords[nogas] # without gas
    # protein_coords_traj_use = coords[protein_cofactors] # all atoms minus gas of interest for xyz2 so we can use those points
    translated = protein_coords_traj_use - protein_coords_traj_use.cpu()[cat].numpy()
    
    shifted = (coords - coords.cpu().numpy()[cat])
    # apply to universe
    u.atoms.positions = shifted
    # write single-frame PDB
    with mda.Writer(f"/mnt/faster{mount}/bw973/Oxy/PHD2/pdbsSim{Sim}/translated_frame{start}_Sim{Sim}.pdb", n_atoms=u.atoms.n_atoms) as W:
        W.write(u.atoms)

    translated2 = coords - coords.cpu()[cat].numpy()

    if distance > 20:
        # radius=min(max(6., next_radius_compare), 8.5)
        # radius=radius
        space=0.65
        xyz2=get_points_step(start, translated, frame_pos, space=space, bottom_threshold=lower, top_threshold=6.5, max_radius=radius, distance_to=distance)
        # xyz2=point_centers(xyz2)
        print("SPACE:",space)
    elif distance > 17:
        # radius=min(max(6., next_radius_compare), 8.5)
        # radius=radius
        space=0.65
        xyz2=get_points_step(start, translated, frame_pos, space=space, bottom_threshold=lower, top_threshold=6.5, max_radius=radius, distance_to=distance)
        # xyz2=point_centers(xyz2)
        print("SPACE:",space)
    elif distance > 15:
        # radius=min(max(6., next_radius_compare), 8.5)
        # radius=radius
        space=0.65
        xyz2=get_points_step(start, translated, frame_pos, space=space, bottom_threshold=lower, top_threshold=5.0, max_radius=radius, distance_to=distance)
        # xyz2=point_centers(xyz2)
        print("SPACE:",space)
    elif distance > 12: # we don't want to cross through betasheets that harbour the metals between 9 and 14 angstroms on either outside of each sheet.
        # radius=min(max(6., next_radius_compare), 8.5) 
        # radius=radius
        space=0.65
        xyz2=get_points_step(start, translated, frame_pos, space=space, bottom_threshold=lower, top_threshold=4.5, max_radius=radius, distance_to=distance)
        # xyz2=point_centers(xyz2)
        print("SPACE:",space)
   
    else:  # we don't want to cross through betasheets that harbour the metals between 9 and 14 angstroms on either outside of each sheet.
        space=0.4
            # radius=min(max(6., next_radius_compare), 8.5)
        # radius=radius
        threshold=lower
        print("SPACE:",space)
        xyz2=get_points_step(start, translated, frame_pos, space=space, bottom_threshold=threshold, top_threshold=4.5, max_radius=radius, distance_to=distance)
        # xyz2=point_centers(xyz2)
        
    print(xyz2.shape)
    if len(xyz2) == 0:
        print("Subpath %s has no possible points.  Terminating at step %d, frame %d" % (subpath, step, start))
        # possible_paths[0][subpath]["still_valid"] = False
        possible_paths[subpath]["still_valid"] = False
        pos_embedding = None
        results = None
        return pos_embedding, results
 
    # def embed(xyz2, step, protein_coords_list,protein_cofactors,global_to_local=global_to_local, radius=6, frame_emb=None):
    # xyz2, pos_embedding = embed_event_pos(xyz2, step=step, gas_indices=gas_indices, coords=translated2, protein_cofactors=protein_cofactors, radius=6, frame_emb=frame_emb)
    xyz2, pos_embedding = embed(xyz2, step=step, protein_coords_list=translated,protein_cofactors=protein_cofactors,global_to_local=global_to_local, radius=6, frame_emb=frame_emb)

    assert(all([p.x[:,0:9].sum().item() == p.x.shape[0] for p in pos_embedding]))
    
    index, frame_pos__, results = predict(pos_embedding, model=model, bs=batch_size,frame_emb_bool=False, node_input_bool=False, atom_as_node_attr=False, waterModel=True, clip=False, threeclass=False, one_hot=one_hot)
    print("SUBPATH:", subpath)
    print("results:", results.mean(), results.max(), (results<0.5).sum(),(results>=0.5).sum(), flush=True)
    
    if len(xyz2[results>=mpk]) > 0:
        bias=get_fe_bias(xyz2[results>=mpk], frame_pos, alpha=50, beta=2, denom=4, range_=bias_r, point=[0,0,0])
    else:
        print("Subpath %s has no positives values.  Terminating at step %d, frame %d" % (subpath, step, start))
        # possible_paths[0][subpath]["still_valid"] = False
        possible_paths[subpath]["still_valid"] = False
        write_pdb2(torch.arange(0,len(pos_embedding)),results,xyz2[:,0],xyz2[:,1],xyz2[:,2], file="/mnt/faster%s/bw973/Oxy/PHD2/pdbsSim%d/"%(mount,Sim)+str(start)+ '_Sim%d_frame'%Sim + str(step) + '_of_' + str((end)) + '_subpath-%s.pdb'%subpath)
        return pos_embedding, results

    
    results_biased = torch.tensor(results[results>=mpk]) * (1+bias)
    
    print("Max biased =", results_biased.max())
    total_biased_positives = (results_biased>=(0.5*(1+bias_r))).sum() # saying min_thresh/(1+bias) 1.5/6 = 0.25
    if (len(pos_embedding) < 500 and (results>=0.5).sum() < 100) or total_biased_positives < 20:
        if (distance < 10 and total_biased_positives < 5) or len(pos_embedding)==1:
            print("k=1")
            scores, points, have_positives, density, keep = topk_with_radius(results_biased, xyz2[results>=mpk], bias_r=bias_r, k=1, radius=1, threshold=min_thresh, initial_results=results[results>=mpk], times=1)
        elif distance < 12 and total_biased_positives < 20:
            print("k=2")
            scores, points, have_positives, density, keep = topk_with_radius(results_biased, xyz2[results>=mpk], bias_r=bias_r, k=2, radius=1, threshold=min_thresh, initial_results=results[results>=mpk], times=1)
            if len(scores) == 0:
                print("k=1")
                scores, points, have_positives, density, keep = topk_with_radius(results_biased, xyz2[results>=mpk], bias_r=bias_r, k=1, radius=1, threshold=min_thresh, initial_results=results[results>=mpk])
        else:
            if len(results[results>=mpk]) < 2:
                print("k=1")
                scores, points, have_positives, density, keep = topk_with_radius(results_biased, xyz2[results>=mpk], bias_r=bias_r, k=1, radius=2, threshold=min_thresh, initial_results=results[results>=mpk],times=1)
            elif len(results[results>=mpk]) < 3:
                print("k=2")
                scores, points, have_positives, density, keep = topk_with_radius(results_biased, xyz2[results>=mpk], bias_r=bias_r, k=2, radius=2, threshold=min_thresh, initial_results=results[results>=mpk],times=1)
            else:
                print("k=3")
                scores, points, have_positives, density, keep = topk_with_radius(results_biased, xyz2[results>=mpk], bias_r=bias_r, k=3, radius=2, threshold=min_thresh, initial_results=results[results>=mpk],times=1)
            
    else:
        print("k=3, len(pos_embedding)", len(pos_embedding))
        scores, points, have_positives, density, keep = topk_with_radius(results_biased, xyz2[results>=mpk], k=3, radius=2, threshold=min_thresh, initial_results=results[results>=mpk],times=1)

    if len(scores) == 0:
        print("PROBABLY going to kill path here since no + points > len K here...")
    kept_positions = []
    res = results[results>=mpk][keep]
    resb = results_biased.numpy()[keep]
    if not isinstance(res,np.ndarray):
        res=np.array([res])
    if not isinstance(resb,np.ndarray):
        resb=np.array([resb])

    possible_paths[subpath]["still_valid"] = False
    top10 = sorted(scores.items(), key=lambda x: x[1]['group_density'], reverse=True)[:]
    parent_score = possible_paths[subpath]["cum_score"]
    child_idx = 0

    for key, vals in top10:
        if vals['group_density'] < 0:
            continue
        if len(kept_positions) == 3:
            continue
        frame_pos=points[vals['nn_idx']].mean(axis=0).numpy()

        if kept_positions:
            # print("kept")
            dists = np.linalg.norm(
                    np.array(kept_positions) - frame_pos[None, :],
                    axis=1
                )
            if np.any(dists < min_dist_apart):
                # print("continue")
                continue

            if not point_exists(global_loc_frame_dict[start], frame_pos):
                global_loc_frame_dict[start].append(frame_pos)
            else:
                print(f"Duplicate path at frame: {start} pos: {frame_pos}")
                ## need to end this subpath??
                dup = find_duplicate(global_loc_frame_dict[start], frame_pos)
                if dup is not None:
                    print("Found:", dup)
                else:
                    print("No match")
                continue
            
            kept_positions.append(frame_pos)
        else:
            if not point_exists(global_loc_frame_dict[start], frame_pos):
                global_loc_frame_dict[start].append(frame_pos)
            else:
                print(f"Duplicate path at frame: {start} pos: {frame_pos}")
                ## need to end this subpath??
                dup = find_duplicate(global_loc_frame_dict[start], frame_pos)
                if dup is not None:
                    print("Found:", dup)
                else:
                    print("No match")
                continue
            kept_positions.append(frame_pos)

        d2=torch.norm(translated2-frame_pos, dim=1)
        iron_dis=np.linalg.norm(frame_pos)
        # d3 = torch.norm(translated2[gas_indices].mean(dim=0)-frame_pos)

        
        child_key = f"{subpath}.{child_idx}"
        child_idx += 1
        local_score = res[vals['nn_idx']].mean()
        biased_score = resb[vals['nn_idx']].mean()


        # this is where if we are on 0 then it is 0.0, ... 0.N
        # then if we are on 0.2.1 then it is 0.2.1.0, ... 0.2.1.N
        possible_paths[child_key] = {
            "parent": subpath,
            "still_valid": True,
            "score": local_score,
            "cum_score": np.log(parent_score * biased_score),
            "path": (
                possible_paths[subpath]["path"]
                + [{
                    "step": step,
                    "frame": start,
                    "frame_pos": frame_pos,
                    "fp": f"draw sphere {{{frame_pos[0]} {frame_pos[1]} {frame_pos[2]}}} radius 0.5",
                    "score": local_score,
                    "biased_score": biased_score,
                    "k": len(vals['nn_idx']),
                    "closest_atom": "%s, %s: %f" % (rnames2[d2.argmin()], anames2[d2.argmin()], d2.min()), "iron_dist":iron_dis
            
                }]
            )
        }
        # parent has now been expanded
        possible_paths[subpath]["still_valid"] = False


        print("draw color %s" % color)
        print("draw sphere {",frame_pos[0],frame_pos[1],frame_pos[2],"} radius 0.5", flush=True)
        print("\tBiased Score:", biased_score)
        print("\tScore:", local_score)
        print("\tDistance to iron:", iron_dis)

        try:
            print("\tClosest atom this frame %s, %s: %f" % (rnames2[d2.argmin()], anames2[d2.argmin()], d2.min()))
        except:
            print("Closest or another oxygen")
        # print("\tActual distance to oxy of interest:", d3.item())


    write_pdb2(torch.arange(0,len(pos_embedding)),results,xyz2[:,0],xyz2[:,1],xyz2[:,2], file="/mnt/faster%s/bw973/Oxy/PHD2/pdbsSim%d/"%(mount,Sim)+str(start)+ '_Sim%d_frame'%Sim + str(step) + '_of_' + str((end)) + '_subpath-%s.pdb'%subpath)

        
        # if len_active == max_paths:
        #     print(f"Reached {max_paths} active paths!")
        #     break
    return pos_embedding, results        
      
        

def build_complete_edge_index(N, device):
    idx = torch.arange(N, device=device)
    i, j = torch.meshgrid(idx, idx, indexing="ij")

    mask = i != j  # no self-loops
    edge_index = torch.stack([i[mask], j[mask]], dim=0)
    return edge_index

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
for a1, a2 in gas_atoms:
    global_bonds.append((a1, a2, 1))
    global_bonds.append((a2, a1, 1))


from torch_geometric.data import Data

def extract_point_cloud(atom_matrix, positions, center):
    """
    Formats the already-cropped atom features and positions into a Data object.
    """
    if positions.shape[0] == 0:
        return None  # Skip empty cubes

    # Center coordinates relative to cube center
    centered_positions = positions - center

    return Data(x=atom_matrix, pos=centered_positions)



def write_pdb2(inds, what, xs, ys, zs, chain='C', file="ml_out.pdb"):
    import numpy as np
    
    print(f"{file}")
    
    fpdb = open(f"{file}", 'wt')
    
    # normalize and compute -log10 for all inds at once
    norm = np.max(what[inds])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        yvals = -np.log10(what[inds] / norm)      # vectorized
        if not isinstance(yvals,np.ndarray):
            yvals=np.array([yvals])
        yvals[np.isneginf(yvals)] = 0             # replace -inf with 0
    
    i_atom = 1
    i_resid = 1
    
    for i, dind in enumerate(inds):
        y = yvals[i]  # scalar for this atom
        
        fpdb.write('{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n'.format(
            'ATOM', i_atom,
            'GG', '', 'GG',
            chain, i_resid, '',
            xs[dind], ys[dind], zs[dind],
            y, what[dind],
            'K', ''
        ))
        
        i_atom += 1
        if i_atom > 999:
            i_atom = 1
            i_resid += 1
    
    fpdb.write('TER\n')
    fpdb.close()



threshold_map = {
    'C': 2.60,
    'H': 1.90,
    'O': 2.5,
    'N': 2.70,
    'S': 2.85,
}

def get_points_init(
    start,
    translated,          # [N_atoms, 3]
    frame_pos,           # [3]
    space=0.65,
    bottom_threshold=2.5,
    top_threshold=3.5,
    min_radius=0.,
    max_radius=30.,
    distance_to=20.0,
    chunk_size=2048,
):
    """
    Generate grid points, exclude points too close to atoms,
    and keep points near frame_pos.

    Memory safe: no full distance matrix.
    """
    
    device = translated.device

    # ---------------------------------------------------------
    # 1. Grid extent selection
    # ---------------------------------------------------------
    if distance_to <= 7.5:
        extent = 12.5
    elif distance_to <= 9:
        extent = 15
    elif distance_to <= 12:
        extent = 18
    else:
        extent = 25
    print("extent:", extent)
    edges = np.arange(-extent, extent + space, space)

    # ---------------------------------------------------------
    # 2. Generate spherical shell grid
    # ---------------------------------------------------------
    pts = []
    rmin2 = 3.5 ** 2
    rmax2 = extent ** 2

    for x in edges:
        x2 = x * x
        for y in edges:
            xy2 = x2 + y * y
            if xy2 > rmax2:
                continue
            for z in edges:
                r2 = xy2 + z * z
                if rmin2 <= r2 <= rmax2:
                    pts.append((x, y, z))

    points_3d = torch.tensor(pts, dtype=torch.float32, device=device)

    # ---------------------------------------------------------
    # 3. Cull atoms that can never affect kept points
    # ---------------------------------------------------------
    # Worst-case geometry bound
    print("max_radius:", max_radius)
    print(torch.norm(frame_pos), max(max_radius, distance_to), top_threshold)
    max_relevant = torch.norm(frame_pos) + max(max_radius, distance_to) + top_threshold

    atom_dist = torch.norm(translated, dim=1)
    atom_mask = atom_dist <= max_relevant
    exclude_points = translated[atom_mask]

    assert(len(anames_nogas) == translated.shape[0])
    exclude_atom_types = anames_nogas[atom_mask]

    # ---------------------------------------------------------
    # 4. Distance-based exclusion (chunked)
    # ---------------------------------------------------------
    unique = sorted(set(exclude_atom_types))
    lut = {u: threshold_map.get(u, bottom_threshold) for u in unique}
    atom_thresholds = torch.tensor([lut[a] for a in exclude_atom_types], device=device)

    P = points_3d.shape[0]

    valid_far   = torch.ones(P, dtype=torch.bool, device=device)   # outside all thresholds
    valid_close = torch.zeros(P, dtype=torch.bool, device=device) 


    for i in range(0, exclude_points.shape[0], chunk_size):
        ep = exclude_points[i:i + chunk_size]
        th = atom_thresholds[i:i + chunk_size]
        d = torch.cdist(points_3d, ep)  # [P, chunk]
        
        # 1) violate any per-atom threshold → reject
        valid_far &= ~(d < th).any(dim=1)

        # 2) within top_threshold of at least one atom → accept shell
        valid_close |= (d < top_threshold).any(dim=1)

    valid = valid_far & valid_close
    filtered_points = points_3d[valid]

    # ---------------------------------------------------------
    # 5. Final restriction around frame_pos
    # ---------------------------------------------------------
    dist_to_frame = torch.norm(filtered_points - frame_pos, dim=1)

    
    filtered_points = filtered_points[(dist_to_frame <= max_radius) & (dist_to_frame >= min_radius)]

    return filtered_points



def get_fe_bias(
    xyz2,
    frame_pos,
    point=None,
    alpha=10.0,   # origin weight
    beta=1.0,     # line weight
    denom=4,
    range_=0.75,
):

    if point is None:
        point = torch.zeros(3, device=xyz2.device, dtype=xyz2.dtype)
    else:
        point = torch.as_tensor(point, device=xyz2.device, dtype=xyz2.dtype)

    xyz_rel = xyz2 - point

    # Distance to origin
    dist_point = torch.norm(xyz_rel, dim=1)

    # Smooth decay instead of 1/d^n
    origin_score = 1.0 / (1.0 + dist_point**denom)

    # Line distance
    line_dir = frame_pos / torch.norm(frame_pos)
    t = (xyz_rel * line_dir).sum(dim=1, keepdim=True)
    d_perp = torch.norm(xyz_rel - t * line_dir, dim=1)

    line_score = 1.0 / (1.0 + d_perp)

    origin_score = origin_score / origin_score.max()
    line_score = line_score / line_score.max()

    combined = alpha * origin_score + beta * line_score
    score = range_ * combined / combined.max()

    return score



def best_k_clique(adj, scores, k):
    """
    adj    : [N,N] bool adjacency
    scores : [N]
    k      : clique size
    """

    N = adj.shape[0]

    if N < k:
        return None, -float("inf")

    adj = adj.clone()
    adj.fill_diagonal_(False)

    best_score = -float("inf")
    best_nodes = None

    def expand(clique, candidates, current_score):

        nonlocal best_score, best_nodes

        if len(clique) == k:
            if current_score > best_score:
                best_score = current_score
                best_nodes = torch.tensor(clique, device=scores.device)
            return

        remaining = k - len(clique)

        if len(candidates) < remaining:
            return

        # upper bound pruning
        if len(candidates) > 0:
            topk = torch.topk(scores[candidates], remaining).values.sum()
            if current_score + topk <= best_score:
                return

        cand_list = candidates.tolist()

        for idx, v in enumerate(cand_list):

            # enforce clique constraint
            next_candidates = candidates[idx + 1:]
            next_candidates = next_candidates[adj[v, next_candidates]]

            expand(
                clique + [v],
                next_candidates,
                current_score + scores[v].item()
            )

    expand([], torch.arange(N, device=scores.device), 0.0)

    return best_nodes, best_score


def topk_with_radius(
    density,
    xyz2,
    k=2,
    radius=2.0,
    threshold=0.5,
    times=2,
    bias_r=2.5,
    initial_results=None,
    min_thresh=1e-3
):

    have_pos = True

    if not isinstance(density, torch.Tensor):
        density = torch.tensor(density, dtype=torch.float32)

    points = xyz2

    keep = torch.where(density > threshold)[0]

    above_p1 = (
        (density > 0.25 * (1 + bias_r)).any().item()
        or (initial_results is not None and (initial_results > 0.25).any().item())
    )

    points = points[keep]
    density = density[keep]

    print("Keeping:", density.shape, ", threshold >=", threshold, ", k=", k)

    N = points.shape[0]

    dists = torch.cdist(points, points)
    adj = (dists <= radius)
    adj.fill_diagonal_(False)

    max_k_possible = int(adj.sum(dim=1).max().item()) + 1
    if max_k_possible < k:
        k = max_k_possible
        print(f"max K within radius {radius} is {k}")

    scores = {}

    # -------------------------
    # k == 1 case
    # -------------------------
    if k == 1:
        for i in range(N):
            scores[i] = {
                "group_density": density[i].item(),
                "mean_density": density[i].item(),
                "nn_idx": torch.tensor([i], device=density.device)
            }

        return scores, points, have_pos, density, keep

    # -------------------------
    # k >= 2 case
    # -------------------------
    seen = set()

    for i in range(N):

        # MUST include i (anchor constraint)
        neighbor_idx = torch.where(adj[i])[0]

        if len(neighbor_idx) < (k - 1):
            scores[i] = {
                "group_density": float("-inf"),
                "mean_density": float("-inf"),
                "nn_idx": None
            }
            continue

        sub_adj = adj[neighbor_idx][:, neighbor_idx]
        sub_scores = density[neighbor_idx]

        best_local, best_score = best_k_clique(
            sub_adj,
            sub_scores,
            k=k - 1
        )

        if best_local is None:
            scores[i] = {
                "group_density": float("-inf"),
                "mean_density": float("-inf"),
                "nn_idx": None
            }
            continue

        best_global = neighbor_idx[best_local]

        # enforce anchor inclusion (TRUE constraint)
        clique = torch.cat([torch.tensor([i], device=best_global.device), best_global])
        clique = torch.sort(clique).values

        clique_key = tuple(clique.tolist())

        if clique_key in seen:
            continue
        seen.add(clique_key)

        total_score = density[best_global].sum() + density[i]

        scores[i] = {
            "group_density": total_score.item(),
            "mean_density": (total_score / k).item(),
            "nn_idx": clique
        }

    return scores, points, have_pos, density, keep



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



def embed(xyz2, step, protein_coords_list,protein_cofactors,global_to_local=global_to_local, radius=6, frame_emb=None):
    """
    This is the embed for the start prediction and for inference in which we remove gas so we can predict to go to any available gas site.  The one above is wrong in the large cell of functions()
    For mimicing an event of one gas, we still include all N-1 gases in the embedding with embed_event_pos(...)
    """
    print("STEP:", step)
    # xyz2:  grid point coordinates for the frame we are trying to predictt, i.e. the next frame, 
    # frame_pos: the given O2 in frame 0
    # first frame frame_pos for O2 is given, each one after is prediction from one before
    # 0 is first step which is given and we start to predict from 1

    pos_embedding = []
    xyz = []
    atom_coords = protein_coords_list # without gas
    print("atom_coords.shape",atom_coords.shape)
    
    # print(dist)
    min_dist = 2.5
    
    for i, point in enumerate(xyz2[0:]):
        center = point
        dist = torch.norm(center - atom_coords[cat[0],:])
      
      
        radius = 6.0

        # Compute distances of all atoms to perturbed point
        dists = torch.norm(atom_coords - center, dim=1)

        # Indices of atoms inside sphere
        # Indices of atoms inside sphere
        # See below this includes GAS now so don't use IT and it is not global
        # EDIT: now okay since protein_cofactors is all so same as nogas[torch.where(dists <= radius)[0]]
        inside_indices = protein_cofactors[nogas][torch.where(dists <= radius)[0]] # this is wrong since protein_cofactors is all atoms now but len dists is length of nogas because atom_coords is gaseless
        # when we added the gas atom_types to the end of types_array_atom, those were never sampled because len(dists) always = total atoms - gas atoms so this was always same as local_inside_indices but had indices of gas atoms, broken + broken = "fixed"
        # WE USE THIS ONLY TO GET THE positions BELOW BECAUSE atom_coords HAS NO GAS!!!!!!!!!!!! and we get global index back with indices: nogas[local_inside_indices]
        local_inside_indices = torch.where(dists <= radius)[0] # indices of atoms not including gas or the translated file without gas atoms

        # Ensure inside_indices is always a 1D tensor
        if torch.is_tensor(inside_indices) and inside_indices.ndim == 0:
            inside_indices = inside_indices.unsqueeze(0)

        if isinstance(inside_indices, (int, np.integer)):
            inside_indices = torch.tensor([inside_indices], device=device)

        if inside_indices.numel() == 0:
            dddd = torch.cdist(center.unsqueeze(0), atom_coords)
            print('too far start or end', dddd.min(), flush=True)
            continue
        xyz.append(i)
        # local_inside_indices = np.array([global_to_local[int(g)] for g in inside_indices])

        # nogas[local_inside_indices] will be any index skipping over gas
        # THIS GETS YOU THE INDEX OF ANY ATOM AND SKIPS OVER ANY GAS INDEX, FOR INSTANCE THIS WILL NEVER BE INDEX 3791-3990, EVER!!!!!
        no_gas_indices = nogas[local_inside_indices] # global
        atom_matrix=types_array_atom[nogas[local_inside_indices]] # now get atomtypes from nogas indices to be sure, i.e. the Global indices for global atoms and should never had a 2, no column should have a 2!!!
        positions=atom_coords[local_inside_indices] # since atom_coords doesn't contain gas
        node_directions = positions - atom_coords[cat[0],:]
        node_distances = torch.norm(node_directions, dim=1)
        local_pos_normalized = (positions - center)/radius  # shape [N, 3]
        
        # atom_matrix_arr.append(atom_matrix)
        # positions_arr.append(positions)
        test = extract_point_cloud(atom_matrix, positions, center)
        # test.x=torch.column_stack([test.x, torch.tensor(frame_emb[idx]).repeat(len(test.x))])
        if frame_emb is not None:
            frame_vector = frame_emb[step]          # [16]
            frame_vector = frame_vector.unsqueeze(0)           # [1, 16]
            frame_vector = frame_vector.expand(len(test.x), -1)  # [N, 16]
            test.distance = dist.expand(len(test.x))
            test.x = torch.cat([test.x, frame_vector], dim=1)    # [N, 6 + 16]

        # print("bulding edges for point:", i+1)
        N=len(inside_indices)
        edge_index_cache = build_complete_edge_index(N, 'cpu')
        # test.edge_index, test.edge_attr = build_edges_and_attrs(inside_indices)
        test.edge_index, test.edge_attr = build_edges_and_attrs_fast(inside_indices, edge_index_cache)
        # print("\tdone bulding edges for point:", i+1)
        test.inside_indices = inside_indices
        test.local_inside_indices = local_inside_indices
        test.center = center
        test.node_attr = local_pos_normalized
        test.node_directions = node_directions
        test.node_distances = node_distances
        test.atoms = np.array(["%s-%s" % (rnames[i], anames[i]) for i in  no_gas_indices])
        pos_embedding.append(test)
    
    xyz2 = xyz2[xyz]
    for pos in pos_embedding:
        # need to add QUERY
        pos.perturbed_from = pos.center
        pos.atoms = "atoms"
        # pos.x = torch.vstack([pos.x, torch.zeros((1, pos.x.shape[1]), dtype=pos.x.dtype, device=pos.x.device)])
        moltype = atom_labels[pos.inside_indices]
        atomic_number = ele2AtomicNumber[torch.where(pos.x[:,0:9])[1]]
        pos.x2 = torch.stack([moltype, atomic_number], dim=1)
        # pos.x2 = torch.vstack([pos.x2, torch.tensor([[2,0]])]) # query 2,0 for O2 or 3,0 for CO2
        # pos.node_attr = torch.vstack([pos.node_attr, torch.tensor([[0.,0.,0.]])])

    return xyz2, pos_embedding


import math

def predict(pos_embedding, model, bs=5, frame_emb_bool=False, node_input_bool=False, atom_as_node_attr=False, waterModel=True, clip=False, threeclass=False, one_hot=False):
  
    device='cuda'
    model = model.to(device)
    

    print("Batch size:", bs)
    test_loader = DataLoader(pos_embedding, batch_size=bs, shuffle=False) 


    values = []
    with torch.no_grad():
        for batch_idx, (data_list) in enumerate(test_loader):
            torch.cuda.empty_cache()
            if (batch_idx + 1) % 50 == 0:
                print("Batch",batch_idx+1, flush=True)
            if isinstance(data_list, list):
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(data_list)
            else:
                batch = data_list  # if batch_size=1, it might already be a Data object

            batch = batch.to('cuda')
            

            
            distance = torch.norm(batch.node_attr[:, :3], dim=1, keepdim=True)  # [N, 1]

            # 2. Convert Cartesian vectors to irreps vector
            x = CartesianTensor("i")
            vector_irrep = x.from_cartesian(batch.node_attr[:, :3])  # [N, 3]

            if waterModel:
                atom_type_onehot = batch.x[:, 0:9]
                mol_atom_type = batch.x2
            else:
                atom_type_onehot = batch.x[:, 0:6]
            
            if one_hot:
                mol_atom_type = atom_type_onehot
            
            frame_emb = batch.x[:, 9:] 
            if frame_emb_bool:
                if atom_as_node_attr:
                    node_attr = torch.cat([
                        distance,            # 1 scalar (0e)
                        # atom_type_onehot,    # 
                        mol_atom_type,
                        frame_emb,           # k scalars (0e)
                        vector_irrep,        # 3-vector (1o)
                    ], dim=1)
                else:
                    node_attr = torch.cat([
                        distance,            # 1 scalar (0e)
                        # atom_type_onehot,    # 
                        frame_emb,           # k scalars (0e)
                        vector_irrep,        # 3-vector (1o)
                    ], dim=1)

            else:
                # node_attr = torch.cat([distance, vector_irrep, frame_emb], dim=1)
                if atom_as_node_attr:
                    node_attr = torch.cat([
                        distance,            # 1 scalar (0e)
                        # atom_type_onehot,    # 6
                        mol_atom_type,
                        # frame_emb,           # k scalars (0e)
                        vector_irrep,        # 3-vector (1o)
                    ], dim=1)
                else:
                    node_attr = torch.cat([
                        distance,            # 1 scalar (0e)
                        # atom_type_onehot,    # 6 scalars (0e)
                        # frame_emb,           # k scalars (0e)
                        vector_irrep,        # 3-vector (1o)
                    ], dim=1)

            
            # node_input = torch.ones((batch.num_nodes, 1), device=batch.x.device)
            if node_input_bool:
                node_input = torch.cat([
                    # batch.distance.unsqueeze(-1), # 1 scalar (0e) distance of center of graph to iron
                    batch.node_distances.unsqueeze(-1),    # 1 scalar (0e) distance to iron for each atom
                    F.normalize(batch.node_directions, p=2, dim=1),        # 3-vector (1o) direction to iron for each atom
                ], dim=1)
            else:
                node_input = torch.ones((batch.num_nodes, 1), device=batch.x.device)

            if not atom_as_node_attr:
                data = {
                "batch": batch.batch,
                # "x": batch.x[:,0:6], # atom type
                # "frame_emb": frame,
                # "x": atom_type_onehot,
                "x": mol_atom_type,
                "node_attr": node_attr, 
                "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "pos": batch.pos,  # if needed in preprocess
            }
            else:
                data = {
                    "batch": batch.batch,
                    # "x": batch.x[:,0:6], # atom type
                    # "frame_emb": frame,
                    "x": node_input,
                    "node_attr": node_attr, 
                    "edge_index": batch.edge_index,
                    "edge_attr": batch.edge_attr,
                    "pos": batch.pos,  # if needed in preprocess
                }

            outputs = model(data)
            if threeclass:
                probs = torch.softmax(outputs, dim=1)   # [B, 3]
                vals = torch.tensor([0.0, 1.2, 2.0], device=probs.device)
                preds = (probs * vals).sum(dim=1)
            else:
                if clip:
                    preds = torch.clamp(outputs.squeeze(-1), 0.0, 2.0)
                else:
                    preds = torch.sigmoid(outputs.squeeze(-1))
                    
            values.extend(preds.detach().cpu().tolist())
            torch.cuda.empty_cache()

        results = np.array(values)
        best = results.argmax() # 2552
        best_pos = best#xyz2[best]
        return best, best_pos, results


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
    return pe  # [N, dim]


def get_points_step(
    start,
    translated,          # [N_atoms, 3]
    frame_pos,           # [3]
    space=0.65,
    bottom_threshold=2.5,
    top_threshold=3.5,
    min_radius=0.,
    max_radius=30.,
    distance_to=20.0,
    chunk_size=2048,
):
    """
    Generate grid points, exclude points too close to atoms,
    and keep points near frame_pos.

    Memory safe: no full distance matrix.
    """
    
    device = translated.device
    protein = translated[protein_atoms]

    # ---------------------------------------------------------
    # 1. Grid extent selection
    # ---------------------------------------------------------
    center = frame_pos

    edges_x = torch.arange(
        center[0] - max_radius,
        center[0] + max_radius + space,
        space,
        device=device
    )

    edges_y = torch.arange(
        center[1] - max_radius,
        center[1] + max_radius + space,
        space,
        device=device
    )

    edges_z = torch.arange(
        center[2] - max_radius,
        center[2] + max_radius + space,
        space,
        device=device
    )

    # square
    X, Y, Z = torch.meshgrid(
        edges_x,
        edges_y,
        edges_z,
        indexing="ij"
    )

    pts = torch.stack([
        X.reshape(-1),
        Y.reshape(-1),
        Z.reshape(-1)
    ], dim=1)

    # ---------------------------------------------------------
    # 2. Generate spherical shell grid
    # ---------------------------------------------------------
    r2 = torch.sum((pts - center)**2, dim=1)

    mask = (
        (r2 >= min_radius**2) &
        (r2 <= max_radius**2)
    )

    points_3d = pts[mask]

    dist_to_protein = torch.cdist(points_3d, protein)
    prot_mask_close = (dist_to_protein < top_threshold).any(dim=1)
    points_3d = points_3d[prot_mask_close]
    # ---------------------------------------------------------
    # 3. Cull atoms that can never affect kept points
    # ---------------------------------------------------------
    # Worst-case geometry bound
    print("max_radius:", max_radius)
    # print(torch.norm(frame_pos), max(max_radius, distance_to), top_threshold)
    # max_relevant = torch.norm(frame_pos) + max(max_radius, distance_to) + top_threshold

    atom_dist_to_frame_pos = torch.norm(translated-frame_pos, dim=1)
    atom_mask = atom_dist_to_frame_pos <= (max_radius+3)
    exclude_points = translated[atom_mask]

    assert(len(anames_nogas) == translated.shape[0])
    exclude_atom_types = anames_nogas[atom_mask]

    # ---------------------------------------------------------
    # 4. Distance-based exclusion (chunked)
    # ---------------------------------------------------------
    unique = sorted(set(exclude_atom_types))
    lut = {u: threshold_map.get(u, bottom_threshold) for u in unique}
    atom_thresholds = torch.tensor([lut[a] for a in exclude_atom_types], device=device)

    P = points_3d.shape[0]

    valid_far   = torch.ones(P, dtype=torch.bool, device=device)   # outside all thresholds
    valid_close = torch.zeros(P, dtype=torch.bool, device=device) 

    
    for i in range(0, exclude_points.shape[0], chunk_size):
        ep = exclude_points[i:i + chunk_size]
        th = atom_thresholds[i:i + chunk_size]
        d = torch.cdist(points_3d, ep)  # [P, chunk]
        print('d.shape', d.shape)
        # 1) violate any per-atom threshold → reject
        valid_far &= ~(d < th).any(dim=1)

        # 2) within top_threshold of at least one atom → accept shell
        valid_close |= (d < top_threshold).any(dim=1)

    valid = valid_far & valid_close
    filtered_points = points_3d[valid]
  

    return filtered_points
    


end=100
start=0
frames = torch.arange(0, end, dtype=torch.float32)  # [0,1,...,N-1]
frame_norm = frames / (end - start)    
frame_emb = sinusoidal_embedding(frame_norm, dim=16)

model_start_path = model_path
model_start = torch.load(model_start_path, weights_only=False).to('cuda').eval()

def get_frame_start(start, s=1.25, min_dist = 4.0, min_res=0.3, k=4, radius=4, ran=False):
    frame_pos=torch.tensor([0.,0.,0.])
    step=2
    u.trajectory[start]
    coords = torch.tensor(u.atoms.positions.copy())
    # protein_coords_traj_use = coords[nogas] # without gas
    protein_coords_traj_use = coords[nogas] # without gas and with water
    translated = protein_coords_traj_use - protein_coords_traj_use.cpu()[cat].numpy()
    #translated = protein_coords_traj_use - protein_coords_traj_use.cpu()[cat-len(gas2)].numpy() # better way to hard code with different length number of gases but I know that Fe is after 200 gas atoms and this is no gas
    print(translated)
    print('translated.shape', translated.shape)

    ignore_starts_coords = translated[ignore_starts]

    shifted = (coords - coords.cpu().numpy()[cat])
    # apply to universe
    u.atoms.positions = shifted
    # write single-frame PDB
    with mda.Writer(f"/mnt/faster{mount}/bw973/Oxy/PHD2/pdbsSim{Sim}/translated_frame{start}_Sim{Sim}.pdb", n_atoms=u.atoms.n_atoms) as W:
        W.write(u.atoms)


    if not ran:
        xyz2=get_points_init(start, translated, frame_pos, space=s, bottom_threshold=2, top_threshold=4.5, min_radius=14., max_radius=23.)
        print(xyz2.shape)
        # return

        xyz2, pos_embedding = embed(xyz2, step=step, protein_coords_list=translated,protein_cofactors=protein_cofactors,global_to_local=global_to_local, frame_emb=frame_emb)
        print(xyz2.shape)
        assert(all([p.x[:,0:9].sum().item() == p.x.shape[0] for p in pos_embedding]))
        
        for i, pos in enumerate(pos_embedding):
            if pos.edge_attr.shape[0] == 0:
                pos.edge_index = torch.tensor([[0],[0]])
                pos.edge_attr = torch.tensor([[0]])
        
        torch.save(pos_embedding,"/mnt/faster%s/bw973/Oxy/PHD2/Sim%d/pos_embedding_Sim%d_f%d_1s_2.5-4.5d_Water.pt"%(mount,Sim,Sim,start))
        torch.save(xyz2,"/mnt/faster%s/bw973/Oxy/PHD2/Sim%d/xyz2_Sim%d_f%d_1s_2.5-4.5d_Water.pt"%(mount,Sim,Sim,start))
    else:
        pos_embedding=torch.load("/mnt/faster%s/bw973/Oxy/PHD2/Sim%d/pos_embedding_Sim%d_f%d_1s_2.5-4.5d_Water.pt"%(mount,Sim,Sim,start))
        xyz2=torch.load("/mnt/faster%s/bw973/Oxy/PHD2/Sim%d/xyz2_Sim%d_f%d_1s_2.5-4.5d_Water.pt"%(mount,Sim,Sim,start))
    
    print(xyz2.shape)
    torch.cuda.empty_cache()
    
    print(model_start_path, flush=True)
    index, frame_pos__, results3 = predict(pos_embedding, model=model_start, bs=6, frame_emb_bool=False, node_input_bool=False, atom_as_node_attr=False, waterModel=True, threeclass=False, one_hot=int(sys.argv[8]))
    print('results3.shape', results3.shape)
    print("Results mean:",results3.mean())
    print("Results < 0.1:", (results3 < 0.1).sum())
    torch.cuda.empty_cache()
    file="/mnt/faster%s/bw973/Oxy/PHD2/pdbsSim%d/"%(mount,Sim) + str(start)+ '_frame' + str(step) + '_of_' + str((end)) + 'far2_Sim%d.pdb'%Sim
    write_pdb2(torch.arange(0,len(xyz2)),results3,xyz2[:,0],xyz2[:,1],xyz2[:,2], file=file)



    kept_positions = []
    group_densities = []

    scores, points, have_positives, density, keep = topk_with_radius(results3, xyz2, k=k, radius=radius, threshold=min_res, initial_results=results3)
    top100 = np.array(sorted(scores.items(), key=lambda x: x[1]['group_density'], reverse=True))[0:200]

    for key, vals in top100:
        if vals['group_density'] < 0:
            continue
        frame_pos = points[vals['nn_idx']].mean(axis=0).numpy()
        # Check distance to all previously kept points
        if kept_positions:
            dists = np.linalg.norm(
                np.array(kept_positions) - frame_pos[None, :],
                axis=1
            )
            if np.any(dists < min_dist):
                continue  # too close → skip
        d = torch.norm(torch.tensor(frame_pos)-ignore_starts_coords, dim=1)
        if all(d > 6):
            kept_positions.append(frame_pos)
            group_densities.append(vals['group_density']/len(vals['nn_idx']))
            print("#density", vals['group_density']/len(vals['nn_idx']))
            print(
                "draw sphere {",
                frame_pos[0], frame_pos[1], frame_pos[2],
                "} radius 1"
            )


    if max(results3) >= best:
        scores, points, have_positives, density, keep = topk_with_radius(results3, xyz2, k=1, radius=1, threshold=best, times=1)
        top100 = np.array(sorted(scores.items(), key=lambda x: x[1]['group_density'], reverse=True))[0:200]

        for key, vals in top100:
            if vals['group_density'] < 0:
                continue
            frame_pos = points[vals['nn_idx']].mean(axis=0).numpy()

            # Check distance to all previously kept points
            if kept_positions:
                dists = np.linalg.norm(
                    np.array(kept_positions) - frame_pos[None, :],
                    axis=1
                )
                if np.any(dists < min_dist):
                    continue  # too close → skip
            d = torch.norm(torch.tensor(frame_pos)-ignore_starts_coords, dim=1)
            if all(d > 6):
                kept_positions.append(frame_pos)
                group_densities.append(vals['group_density'])
                print("#density", vals['group_density'])
                print(
                    "draw sphere {",
                    frame_pos[0], frame_pos[1], frame_pos[2],
                    "} radius 1"
                )
    
    return kept_positions, results3, group_densities

model_name = model_path.split("/")[-1].split(".")[0]
frames = np.arange(int(sys.argv[3]), int(sys.argv[4]), 200)
# frames2 = np.array([32000, 32100, 34500, 50700, 61700,  63500, 64800, 65600, 85000, 90700, 91200, 91800, 96600])
#frames = np.concatenate([frames, frames2])
import pickle 
print(frames)
starting_points = []
starting_densities = []
start = int(sys.argv[7])
if start:
    for f in frames:
        print(f, flush=True)
        if f == 0:
            kept_positions, results3, group_densities = get_frame_start(f, s=0.65, min_dist=3., min_res=start_thresh, ran=False)
        else:
            kept_positions, results3, group_densities = get_frame_start(f, s=0.65, min_dist=3., min_res=start_thresh, k=2, radius=2.5, ran=int(sys.argv[6]))
        starting_points.append(kept_positions)
        starting_densities.append(group_densities)

    my_zip = zip(frames, starting_points, starting_densities)
    np.save(f"Sim{Sim}_{frames[0]}_{frames[-1]}_{model_name}.npy", my_zip)
else:
    my_zip = np.load(f"Sim{Sim}_{frames[0]}_{frames[-1]}_{model_name}.npy", allow_pickle=True).item()


# GLOBALS
colors = ['blue','green','black', 'orange', 'yellow']
#min_thresh = 1.5 # because bias is 2.5 so 0.5*(1+2.5) = 1.75
# def point_exists(lst, p):
#     return any(np.array_equal(x, p) for x in lst)
def point_exists(lst, p, rtol=0, atol=1e-4):
    return any(np.allclose(x, p, rtol=rtol, atol=atol) for x in lst)

def find_duplicate(lst, p, atol=1e-4):
    for x in lst:
        if np.allclose(x, p, rtol=0, atol=atol):
            return x
    return None

global_loc_frame_dict = {}

bias_r=5.0
min_thresh=0.5 # test now that we only keep predictions above min + keep (mpk)
lower=1.9
mpk=min_thresh
min_dist_apart = 2.
radius = 6
predict_events = sys.argv[5]
total_events = 0




for predicted_frame_start, sps, sdens in list(my_zip)[0:]:
    print("******** Frame start", predicted_frame_start, flush=True)
    for sp, dens in zip(sps[1:], sdens[1:]):
        print("Predicting from", sp, ", with density", dens, flush=True)
        frame_pos=torch.tensor(sp)



        print("draw color magenta", flush=True)
        print("draw sphere {",frame_pos.numpy()[0],frame_pos.numpy()[1],frame_pos.numpy()[2],"} radius 1", flush=True)

        possible_paths = {
            "0": {
                "parent": None,
                "score": 1.0,
                "cum_score": 1.0,
                "still_valid": True,
                "path": [{
                    "step": 0,
                    "frame": predicted_frame_start,
                    "frame_pos": frame_pos.numpy(),
                    "fp": f"draw sphere {{{frame_pos[0]} {frame_pos[1]} {frame_pos[2]}}} radius 1.0",
                    "score": 1.0,
                    "model": model_path
                }]
            }
        }

        start_frame = predicted_frame_start+1
        current_frame = start_frame
        max_frame = current_frame + end - 1
        step = 1
        eval_window = 3
        beam_width = 3
        fallback_queue = []
        last_eval_step = 0
        less_than_6 = False
        event_node_id = None

        # stores last committed eval checkpoint
        checkpoint = {
            "step": 1,
            "frame": current_frame,
            "possible_paths": None,
            "global_loc_frame_dict": None,
            "node_id": None
        }

        while current_frame < max_frame and not less_than_6:

            if current_frame not in global_loc_frame_dict:
                global_loc_frame_dict[current_frame] = []

            active_nodes = [k for k, v in possible_paths.items() if v["still_valid"]]

            print(
                f"\nSTEP={step} FRAME={current_frame} "
                f"ACTIVE={len(active_nodes)} "
                f"FALLBACKS={len(fallback_queue)}"
            )

            # ---------------------------
            # FAILURE → RESTART FROM LAST CHECKPOINT
            # ---------------------------
            if not active_nodes:

                if not fallback_queue:
                    print("No paths remain")
                    break

                cp = fallback_queue.pop(0)
                print("\n=== RESTART ===")
                print(f"Using fallback: {cp['node_id']}")

                # restore LAST VALID EVAL CHECKPOINT (NOT failure state)
                possible_paths = copy.deepcopy(checkpoint["possible_paths"])
                global_loc_frame_dict = copy.deepcopy(checkpoint["global_loc_frame_dict"])

                step = checkpoint["step"]
                current_frame = start_frame + step - 1

                possible_paths[cp["node_id"]]["still_valid"] = True
                active_after_restart = [
                    k for k, v in possible_paths.items()
                    if v["still_valid"]
                ]

                print(f"RESTART FROM CHECKPOINT -> {cp['node_id']} @ step {step}")

                print(
                    f"ACTIVE AFTER RESTART "
                    f"({len(active_after_restart)}):"
                )
                continue

            # ---------------------------
            # EXPAND ACTIVE NODES
            # ---------------------------
            for node_id in active_nodes:
                
                last = possible_paths[node_id]["path"][-1]
                if 'iron_dist' in last and last['iron_dist'] < 6:
                    event_node_id = node_id
                    less_than_6 = True
                    print("I hope it breaks...")
                    break

                if step >=50 and 'iron_dist' in last and last['iron_dist'] > 15:
                    possible_paths[node_id]["still_valid"] = False
                    print("Print too far after step 50...")
                    break

                if not less_than_6:
                    frame_pos = last["frame_pos"] if isinstance(last, dict) else last
                
                    pos_embedding, results = run_step2(
                        frame_pos=frame_pos,
                        possible_paths=possible_paths,
                        start=current_frame,
                        step=step,
                        subpath=node_id,
                        model=model_start,
                        batch_size=6,
                        one_hot=int(sys.argv[8])
                    )

            # ---------------------------
            # EVAL WINDOW (COMMIT CHECKPOINT)
            # ---------------------------
            if step % eval_window == 0:

                last_eval_step = step
                depth = step + 1

                ranked = sorted(
                    [k for k, v in possible_paths.items()
                    if v["still_valid"] and len(k.split(".")) == depth],
                    key=lambda k: possible_paths[k]["cum_score"],
                    reverse=True
                )

                if ranked:

                    for v in possible_paths.values():
                        v["still_valid"] = False

                    possible_paths[ranked[0]]["still_valid"] = True

                    # SAVE CHECKPOINT (THIS IS THE ONLY RESTART SOURCE)
                    checkpoint = {
                        "step": step + 1,   # restart AFTER this eval boundary
                        "frame": current_frame,
                        "possible_paths": copy.deepcopy(possible_paths),
                        "global_loc_frame_dict": copy.deepcopy(global_loc_frame_dict),
                        "node_id": ranked[0]
                    }

                    # beam pruning
                    fallback_queue = [{
                        "node_id": nid,
                        "frame": current_frame,
                        "step": step,
                        "possible_paths": None,
                        "global_loc_frame_dict": None
                    } for nid in ranked[1:beam_width]]

                    

                    print(
                        f"COMMIT Step {step}: {ranked[0]} "
                        f"score={possible_paths[ranked[0]]['cum_score']:.4f}"
                    )
                
                else:
                    print("No ranked:", len(ranked))


            # ---------------------------
            # STEP ADVANCE
            # ---------------------------
            step += 1
            current_frame += 1
            if less_than_6:
                print("I hope it breaks...2")

        if event_node_id:
            total_events +=1
            print("Event:", total_events)
            start_frame = possible_paths[event_node_id]['path'][0]['frame'] 
            end_frame = possible_paths[event_node_id]['path'][-1]['frame'] 
            np.save("Sim%s_predict_events%s/start_%d_end_%d_event%d.npy" % (Sim, predict_events, start_frame, end_frame,total_events), possible_paths[node_id])
        else:
            print("TRY NEXT PREDICTION")


        
print("Total time {}".format(time.time()-data_time), flush=True)
