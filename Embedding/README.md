how to embed for https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/v2106/gate_points_networks.py for 6 Ang radius graph

## First off I create csv files for the long MD we already ran that includes all events < 6 Ang (6 seems to be my magic number)
1.  First run contacts.py to get numpy file of contacts for events that reach < 6 A of Fe2+
2.  Then run contacts_df2.py to put that into a dataframe/.csv file that could be used potentially in tree based classifiers like PathInHydro
3.  That .csv file will be used for reading in events to embed, full with solvent, graphs using MDAanalysis and torch. see `embed2_water_query.py`


Then update in e3nn: (actually don't need to do this)

```python
        # edge_attr = data["edge_attr"]
        # Generate dummy edge_attr if not provided
        if "edge_attr" in data:
            edge_attr = data["edge_attr"]
        else:
            E = edge_src.shape[0]
            edge_attr = torch.ones(E, 1, device=data["pos"].device)
```


atoms are given both a [moltype, atomic_number] plus a query node at center of graph for the gas. see `data_moltype_CO2.py` after `embed2_water_query.py`

```
atom_labels[solvent] = 0
atom_labels[protein_atoms] = 1
atom_labels[gas2] = 2 # for O2

atom_labels[solvent] = 0
atom_labels[protein_atoms] = 1
atom_labels[gas2] = 3 # for CO2
```

For example:

Query O₂, neighbouring O₂ (O₂ only system)
```
[2, 0]    # COM of query O2
[2, 8]    # O atom
[2, 8]    # O atom
```

Query CO₂, neighbouring O₂ (Mixed CO₂/O₂ system)
```
[3, 0]    # COM of query CO2
[2, 8]    # O atom
[2, 8]    # O atom
```

Query CO₂, neighbouring CO₂ (Mixed CO₂/O₂ or only CO₂ systems)
```
[0, 3]    # COM of query CO2
[3, 8]    # O atom
[3, 6]    # C atom
[3, 8]    # O atom
```

Query O₂, neighbouring O₂ and CO₂ (Mixed CO₂/O₂ system)
```
[2, 0]    # COM of query O2
[2, 8]    # O2 O atom
[2, 8]    # O2 O atom
[3, 8]    # CO2 O atom
[3, 6]    # CO2 C atom
[3, 8]    # CO2 O atom
```
