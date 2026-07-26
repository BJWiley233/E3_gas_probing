# E3_gas_probing


## Step 1 Embed using Embedding/README.md
Events can be extracted from smaller, non-solvated files, but the embedding is done with solvent so MDAnalysis is preferred to not need to load entire dcd/xtc into memory.


## Step 2  Train
using `train.py` which can load in any positives and negatives from any gas as long they have a different query node at the end of data.x, data.pos, data.node_attr, and data.x2 which has the [moltype, atomic_number] where moltype = 0 for solvent, 1 for solute, 2 for O2 gas, 3 for CO2 gas, 4 for whatever gas, ..., N for H gas.


## Step 3 Inference on short MD simulations
using `inference.py` which needs to be updated for multiple gases in the possible location updates, needs to update embedding function for 'query nodes/COM of gases'
