how to embed for https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/v2106/gate_points_networks.py for 6 Ang radius graph

Then update in e3nn:

```python
        # edge_attr = data["edge_attr"]
        # Generate dummy edge_attr if not provided
        if "edge_attr" in data:
            edge_attr = data["edge_attr"]
        else:
            E = edge_src.shape[0]
            edge_attr = torch.ones(E, 1, device=data["pos"].device)
```

atom_labels[solvent] = 0
atom_labels[protein_atoms] = 1
atom_labels[gas2] = 2 # for O2

atom_labels[solvent] = 0
atom_labels[protein_atoms] = 1
atom_labels[gas2] = 3 # for CO2

For example:

Query O₂, neighbouring O₂
[0, 2]    # COM of query O2
[8, 2]    # O atom
[8, 2]    # O atom
Query CO₂, neighbouring O₂
[0, 3]    # COM of query CO2
[8, 2]    # O atom
[8, 2]    # O atom
Query CO₂, neighbouring CO₂
[0, 3]    # COM of query CO2
[8, 3]    # O atom
[6, 3]    # C atom
[8, 3]    # O atom
