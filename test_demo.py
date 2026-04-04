import sys
sys.path.insert(0, '.')
import networkx as nx
import numpy as np
import torch
from common.label_encoder import UniversalLabelEncoder
from common.heterogenous_syn import get_heterogeneous_dataset

print('============================================================')
print('  TEST 1: Real dataset - Les Miserables character network')
print('============================================================')

G_real = nx.les_miserables_graph()
print('  Nodes:', G_real.number_of_nodes())
print('  Edges:', G_real.number_of_edges())

NODE_TYPES = ['major_character', 'minor_character', 'background_character']
EDGE_TYPES = ['strong_connection', 'weak_connection']

for node in G_real.nodes():
    degree = G_real.degree(node)
    if degree > 10:
        G_real.nodes[node]['label'] = 'major_character'
    elif degree > 4:
        G_real.nodes[node]['label'] = 'minor_character'
    else:
        G_real.nodes[node]['label'] = 'background_character'

for u, v in G_real.edges():
    weight = G_real.edges[u,v].get('weight', 1)
    G_real.edges[u,v]['type'] = float(EDGE_TYPES.index('strong_connection' if weight > 5 else 'weak_connection'))

print('  Node types:', set(d['label'] for _,d in G_real.nodes(data=True)))
print('  Edge types:', set(EDGE_TYPES[int(d['type'])] for _,_,d in G_real.edges(data=True)))

encoder = UniversalLabelEncoder()
for node, data in G_real.nodes(data=True):
    data['node_feature_emb'] = encoder.encode(data['label'])

print('  Vector shape:', tuple(G_real.nodes[list(G_real.nodes())[0]]['node_feature_emb'].shape))

nodes_by_type = {}
for n,d in G_real.nodes(data=True):
    nodes_by_type.setdefault(d['label'], []).append(n)

print('  Same label = same vector:')
for label, nodes in nodes_by_type.items():
    if len(nodes) >= 2:
        v1 = G_real.nodes[nodes[0]]['node_feature_emb']
        v2 = G_real.nodes[nodes[1]]['node_feature_emb']
        print('    ', label, ':', torch.allclose(v1, v2))

print('')
print('============================================================')
print('  TEST 2: Real dataset - Karate Club social network')
print('============================================================')

G_karate = nx.karate_club_graph()
print('  Nodes:', G_karate.number_of_nodes())
print('  Edges:', G_karate.number_of_edges())

for node in G_karate.nodes():
    club = G_karate.nodes[node]['club']
    G_karate.nodes[node]['label'] = 'officer' if club == 'Officer' else 'member'

for u, v in G_karate.edges():
    G_karate.edges[u,v]['type'] = float(0)

encoder2 = UniversalLabelEncoder()
for node, data in G_karate.nodes(data=True):
    data['node_feature_emb'] = encoder2.encode(data['label'])

print('  Node types:', set(d['label'] for _,d in G_karate.nodes(data=True)))
print('  Vector shape:', tuple(G_karate.nodes[0]['node_feature_emb'].shape))
print('  Two officer nodes same vector:', torch.allclose(
    G_karate.nodes[0]['node_feature_emb'],
    G_karate.nodes[1]['node_feature_emb']
))

print('')
print('============================================================')
print('  TEST 3: Generic domains - same code different vocabulary')
print('============================================================')

for domain, ntypes, etypes in [
    ('Biomedical', ['Gene','Protein','Disease','Drug'],  ['regulates','binds','treats','encodes']),
    ('Finance',    ['Account','Bank','Transaction'],     ['transfers','owns','audits']),
    ('Social',     ['User','Post','Group'],              ['follows','likes','joins']),
    ('Knowledge',  ['Person','Organization','Location'], ['worksAt','locatedIn','knows']),
]:
    ds = get_heterogeneous_dataset('graph', dataset_len=1, sizes=np.arange(6,10), node_types=ntypes, edge_types=etypes)
    gd = ds[0].G
    print('  ' + domain + ':')
    print('    node types:', set(d['label'] for _,d in gd.nodes(data=True)))
    print('    edge types:', set(etypes[int(d['type'])] for _,_,d in gd.edges(data=True)))

print('')
print('============================================================')
print('  All tests completed successfully')
print('============================================================')
