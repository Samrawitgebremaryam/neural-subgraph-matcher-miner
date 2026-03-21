import numpy as np
import deepsnap.dataset as dataset
from common.combined_syn import ERGenerator, WSGenerator, BAGenerator, PowerLawClusterGenerator


def assign_node_types(graph, node_types):
    for node in graph.nodes():
        label = np.random.choice(node_types)
        graph.nodes[node]["label"] = label
        graph.nodes[node]["node_feature"] = float(node_types.index(label))
    return graph


def assign_edge_types(graph, edge_types, valid_edges=None):
    for u, v in graph.edges():
        if valid_edges:
            u_label = graph.nodes[u].get("label", "")
            v_label = graph.nodes[v].get("label", "")
            allowed = valid_edges.get((u_label, v_label)) or valid_edges.get((v_label, u_label)) or edge_types
        else:
            allowed = edge_types
        chosen = np.random.choice(allowed)
        graph.edges[u, v]["type"] = float(edge_types.index(chosen))
    return graph


def make_heterogeneous(graph, node_types=None, edge_types=None, valid_edges=None):
    if node_types:
        graph = assign_node_types(graph, node_types)
    if edge_types:
        graph = assign_edge_types(graph, edge_types, valid_edges)
    return graph


class HeterogeneousERGenerator(ERGenerator):
    def __init__(self, sizes, node_types=None, edge_types=None, valid_edges=None, p_alpha=1.3, **kwargs):
        super().__init__(sizes, p_alpha=p_alpha, **kwargs)
        self.node_types = node_types
        self.edge_types = edge_types
        self.valid_edges = valid_edges
    def generate(self, size=None):
        return make_heterogeneous(super().generate(size), self.node_types, self.edge_types, self.valid_edges)


class HeterogeneousWSGenerator(WSGenerator):
    def __init__(self, sizes, node_types=None, edge_types=None, valid_edges=None, **kwargs):
        super().__init__(sizes, **kwargs)
        self.node_types = node_types
        self.edge_types = edge_types
        self.valid_edges = valid_edges
    def generate(self, size=None):
        return make_heterogeneous(super().generate(size), self.node_types, self.edge_types, self.valid_edges)


class HeterogeneousBAGenerator(BAGenerator):
    def __init__(self, sizes, node_types=None, edge_types=None, valid_edges=None, **kwargs):
        super().__init__(sizes, **kwargs)
        self.node_types = node_types
        self.edge_types = edge_types
        self.valid_edges = valid_edges
    def generate(self, size=None):
        return make_heterogeneous(super().generate(size), self.node_types, self.edge_types, self.valid_edges)


class HeterogeneousPowerLawGenerator(PowerLawClusterGenerator):
    def __init__(self, sizes, node_types=None, edge_types=None, valid_edges=None, **kwargs):
        super().__init__(sizes, **kwargs)
        self.node_types = node_types
        self.edge_types = edge_types
        self.valid_edges = valid_edges
    def generate(self, size=None):
        return make_heterogeneous(super().generate(size), self.node_types, self.edge_types, self.valid_edges)


def get_heterogeneous_generator(sizes, size_prob=None, dataset_len=None, node_types=None, edge_types=None, valid_edges=None):
    shared = dict(node_types=node_types, edge_types=edge_types, valid_edges=valid_edges, size_prob=size_prob)
    return dataset.EnsembleGenerator([
        HeterogeneousERGenerator(sizes, **shared),
        HeterogeneousWSGenerator(sizes, **shared),
        HeterogeneousBAGenerator(sizes, **shared),
        HeterogeneousPowerLawGenerator(sizes, **shared),
    ], dataset_len=dataset_len)


def get_heterogeneous_dataset(task, dataset_len, sizes, size_prob=None, node_types=None, edge_types=None, valid_edges=None, **kwargs):
    generator = get_heterogeneous_generator(sizes, size_prob=size_prob, dataset_len=dataset_len, node_types=node_types, edge_types=edge_types, valid_edges=valid_edges)
    return dataset.GraphDataset(None, task=task, generator=generator, **kwargs)


if __name__ == '__main__':
    sizes = np.arange(6, 15)

    NODE_TYPES = ['Gene', 'Protein', 'Disease', 'Drug']
    EDGE_TYPES = ['regulates', 'binds', 'treats', 'encodes']

    print('=== Test 1: Structure-only (no labels) ===')
    ds = get_heterogeneous_dataset('graph', dataset_len=2, sizes=sizes)
    g = ds[0].G
    print('  Nodes:', g.number_of_nodes(), 'Edges:', g.number_of_edges())
    print('  PASS')

    print('')
    print('=== Test 2: Biomedical domain with labels ===')
    ds2 = get_heterogeneous_dataset(
        'graph', dataset_len=2, sizes=sizes,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        valid_edges={
            ('Gene', 'Gene'): ['regulates'],
            ('Gene', 'Protein'): ['encodes'],
            ('Protein', 'Protein'): ['binds'],
            ('Drug', 'Disease'): ['treats'],
        }
    )
    g2 = ds2[0].G
    print('  Nodes:', g2.number_of_nodes(), 'Edges:', g2.number_of_edges())
    print('  Node types:', set(d['label'] for _, d in g2.nodes(data=True)))
    print('  Edge type ids:', set(int(d['type']) for _, _, d in g2.edges(data=True)))
    print('  Edge type names:', set(EDGE_TYPES[int(d['type'])] for _, _, d in g2.edges(data=True)))
    print('  Sample nodes:')
    for n, d in list(g2.nodes(data=True))[:3]:
        print('    node', n, '-> label:', d['label'], '| node_feature:', d['node_feature'])
    print('  Sample edges:')
    for u, v, d in list(g2.edges(data=True))[:3]:
        print('   ', u, '--[' + EDGE_TYPES[int(d['type'])] + ']-->', v)
    print('')
    print('Done.')
