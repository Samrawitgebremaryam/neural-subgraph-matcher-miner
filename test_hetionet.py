import sys
sys.path.insert(0, '.')
import networkx as nx
import torch
from common.label_encoder import UniversalLabelEncoder

print('============================================================')
print('  REAL DATASET TEST: Hetionet Biomedical Knowledge Graph')
print('============================================================')

# --- Load nodes ---
print('\n[1] Loading nodes...')
nodes = {}
node_types = set()
with open('hetionet-nodes.tsv') as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split('\t')
        node_id, name, kind = parts[0], parts[1], parts[2]
        nodes[node_id] = {'name': name, 'kind': kind}
        node_types.add(kind)

print('  Total nodes:', len(nodes))
print('  Node types found:', node_types)

# --- Load edges ---
print('\n[2] Loading edges...')
edges = []
edge_types = set()
with open('hetionet-edges.sif') as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split('\t')
        source, metaedge, target = parts[0], parts[1], parts[2]
        edges.append((source, metaedge, target))
        edge_types.add(metaedge)

print('  Total edges:', len(edges))
print('  Edge types found:', len(edge_types), 'types')
print('  Sample edge types:', list(edge_types)[:5])

# --- Build a small subgraph for testing ---
# Take only Gene and Disease nodes with their edges
print('\n[3] Building Gene-Disease subgraph...')
G = nx.Graph()

# Add nodes
for node_id, data in nodes.items():
    if data['kind'] in ['Gene', 'Disease', 'Compound', 'Anatomy']:
        G.add_node(node_id, label=data['kind'], name=data['name'])

# Add edges between those nodes only
edge_type_list = sorted(edge_types)
edge_type_to_id = {e: float(i) for i, e in enumerate(edge_type_list)}

added = 0
for source, metaedge, target in edges:
    if source in G.nodes() and target in G.nodes():
        G.add_edge(source, target, type=edge_type_to_id[metaedge])
        added += 1
    if added >= 5000:  # limit size for demo
        break

print('  Nodes in subgraph:', G.number_of_nodes())
print('  Edges in subgraph:', G.number_of_edges())
print('  Node types:', set(d['label'] for _,d in G.nodes(data=True)))

# --- Apply label encoder ---
print('\n[4] Encoding node labels with UniversalLabelEncoder...')
encoder = UniversalLabelEncoder()
for node, data in G.nodes(data=True):
    data['node_feature_emb'] = encoder.encode(data['label'])

sample_node = list(G.nodes())[0]
print('  Sample node:', sample_node)
print('  Node type:', G.nodes[sample_node]['label'])
print('  Node name:', G.nodes[sample_node]['name'])
print('  Encoded vector shape:', tuple(G.nodes[sample_node]['node_feature_emb'].shape))

# --- Verify same type = same vector ---
print('\n[5] Verifying same node type = same vector...')
nodes_by_type = {}
for n, d in G.nodes(data=True):
    nodes_by_type.setdefault(d['label'], []).append(n)

for label, node_list in nodes_by_type.items():
    if len(node_list) >= 2:
        v1 = G.nodes[node_list[0]]['node_feature_emb']
        v2 = G.nodes[node_list[1]]['node_feature_emb']
        print('  Two', label, 'nodes same vector:', torch.allclose(v1, v2))

print('\n============================================================')
print('  Hetionet test completed successfully')
print('============================================================')
