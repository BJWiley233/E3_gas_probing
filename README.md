# E3_gas_probing


## Step 1 Embed using Embedding/README.md
Events can be extracted from smaller, non-solvated files, but the embedding is done with solvent so MDAnalysis is preferred to not need to load entire dcd/xtc into memory.


## Step 2  Train
using `train.py` which can load in any positives and negatives from any gas as long
