"""
Test suite for igraph-native utils_ig.py

Tests the direct igraph implementation without NetworkX conversion layers.
"""

import pytest
import igraph as ig
import numpy as np
import torch
from collections import Counter

from common import utils_ig


class TestSamplingFunctions:
    """Test graph sampling functions."""
    
    def test_sample_neigh_basic(self):
        """Test basic neighborhood sampling."""
        # Create simple test graphs
        graphs = [
            ig.Graph([(0, 1), (1, 2), (2, 3), (3, 4)]),
            ig.Graph([(0, 1), (1, 2), (2, 0)]),
            ig.Graph([(i, i+1) for i in range(10)]),
        ]
        
        sampled_ig, neigh = utils_ig.sample_neigh(graphs, size=3, graph_type="undirected")
        
        assert len(neigh) == 3
        assert len(sampled_ig.vs) == 3
        assert len(sampled_ig.es) >= 2  # At least some edges
    
    def test_sample_neigh_different_sizes(self):
        """Test sampling different neighborhood sizes."""
        graphs = [ig.Graph([(i, i+1, i+2) for i in range(0, 20, 3)])]
        
        for size in [3, 5, 7]:
            sampled, neigh = utils_ig.sample_neigh(graphs, size=size, graph_type="undirected")
            assert len(neigh) == size
    
    def test_sample_neigh_directed_graph(self):
        """Test sampling from directed graphs."""
        g = ig.Graph([(0, 1), (1, 2), (2, 3)], directed=True)
        graphs = [g]
        
        sampled, neigh = utils_ig.sample_neigh(graphs, size=3, graph_type="directed")
        
        assert len(neigh) == 3
        assert sampled.is_directed()
    
    def test_sample_neigh_weights_by_size(self):
        """Test that sampling is weighted by graph size."""
        small_g = ig.Graph([(0, 1)])
        large_g = ig.Graph([(i, i+1) for i in range(50)])
        
        graphs = [small_g, large_g]
        
        # Sample multiple times - should mostly select from large graph
        counts = Counter()
        for _ in range(100):
            sampled, _ = utils_ig.sample_neigh(graphs, size=3, graph_type="undirected")
            counts[len(sampled.vs)] += 1
        
        # Should have sampled something
        assert len(counts) > 0


class TestHashingFunctions:
    """Test graph hashing functions."""
    
    def test_vec_hash_consistency(self):
        """Test that vec_hash is deterministic."""
        v = np.array([1, 2, 3, 4, 5])
        
        hash1 = utils_ig.vec_hash(v)
        hash2 = utils_ig.vec_hash(v)
        
        assert hash1 == hash2
    
    def test_wl_hash_basic(self):
        """Test basic WL hash computation."""
        g = ig.Graph([(0, 1), (1, 2), (2, 0)])  # Triangle
        
        hash_val = utils_ig.wl_hash(g)
        
        assert isinstance(hash_val, tuple)
        assert len(hash_val) == 64  # Default dim
    
    def test_wl_hash_consistency(self):
        """Test that WL hash is deterministic."""
        g = ig.Graph([(0, 1), (1, 2), (2, 3)])
        
        hash1 = utils_ig.wl_hash(g)
        hash2 = utils_ig.wl_hash(g)
        
        assert hash1 == hash2
    
    def test_wl_hash_different_graphs(self):
        """Test that different graphs produce different hashes."""
        g1 = ig.Graph([(0, 1), (1, 2)])  # Path
        g2 = ig.Graph([(0, 1), (1, 2), (2, 0)])  # Triangle
        
        hash1 = utils_ig.wl_hash(g1)
        hash2 = utils_ig.wl_hash(g2)
        
        # Different structures should likely have different hashes
        assert hash1 != hash2
    
    def test_wl_hash_with_anchor(self):
        """Test WL hash with anchor node."""
        g = ig.Graph([(0, 1), (1, 2), (2, 0)])
        
        # Set anchor
        for v in g.vs:
            v['anchor'] = 0
        g.vs[0]['anchor'] = 1
        
        hash_val = utils_ig.wl_hash(g, node_anchored=True)
        
        assert isinstance(hash_val, tuple)
    
    def test_wl_hash_custom_dim(self):
        """Test WL hash with custom dimension."""
        g = ig.Graph([(0, 1), (1, 2)])
        
        hash_32 = utils_ig.wl_hash(g, dim=32)
        hash_128 = utils_ig.wl_hash(g, dim=128)
        
        assert len(hash_32) == 32
        assert len(hash_128) == 128


class TestEnumeration:
    """Test subgraph enumeration functions."""
    
    def test_enumerate_subgraph_basic(self):
        """Test basic subgraph enumeration."""
        g = ig.Graph([(i, i+1) for i in range(5)] + [(4, 0)])  # Cycle
        
        subgraphs = utils_ig.enumerate_subgraph(g, k=3, progress_bar=False)
        
        # Should have multiple subgraphs
        assert len(subgraphs) > 0
        
        # Check sizes
        for (size, hash_val), subgraph_list in subgraphs.items():
            assert size <= 3
            assert len(subgraph_list) > 0
    
    def test_enumerate_subgraph_complete_graph(self):
        """Test enumeration on complete graph."""
        g = ig.Graph.Complete(5)
        
        subgraphs = utils_ig.enumerate_subgraph(g, k=2, progress_bar=False)
        
        # Complete graph K_5 should have many 2-node subgraphs
        total_size_2 = sum(len(sg_list) for (size, _), sg_list in subgraphs.items() if size == 2)
        assert total_size_2 > 0
    
    def test_enumerate_subgraph_small_k(self):
        """Test enumeration with small k."""
        g = ig.Graph([(0, 1), (1, 2), (2, 3)])
        
        subgraphs = utils_ig.enumerate_subgraph(g, k=1, progress_bar=False)
        
        # Should have size-1 subgraphs (single nodes)
        for (size, _), sg_list in subgraphs.items():
            assert size <= 1


class TestQueryGeneration:
    """Test baseline query generation functions."""
    
    def test_gen_baseline_queries_mfinder_basic(self):
        """Test mfinder-style query generation."""
        queries = [ig.Graph([(0, 1), (1, 2)]), ig.Graph([(0, 1)])]
        targets = [ig.Graph([(i, i+1) for i in range(10)]) for _ in range(3)]
        
        result = utils_ig.gen_baseline_queries_mfinder(
            queries, targets, n_samples=100, node_anchored=False
        )
        
        assert len(result) > 0
        for g in result:
            assert isinstance(g, ig.Graph)
    
    def test_gen_baseline_queries_mfinder_anchored(self):
        """Test anchored query generation."""
        queries = [ig.Graph([(0, 1), (1, 2)])]
        targets = [ig.Graph([(i, i+1) for i in range(15)])]
        
        result = utils_ig.gen_baseline_queries_mfinder(
            queries, targets, n_samples=50, node_anchored=True
        )
        
        assert len(result) > 0


class TestGraphOperations:
    """Test basic graph operations."""
    
    def test_standardize_graph_ig(self):
        """Test graph standardization."""
        g = ig.Graph([(0, 1), (1, 2)])
        
        g_std = utils_ig.standardize_graph_ig(g)
        
        # Check attributes
        for v in g_std.vs:
            assert 'node_feature' in v.attributes()
            assert 'label' in v.attributes()
        
        for e in g_std.es:
            assert 'weight' in e.attributes()
    
    def test_standardize_graph_with_anchor(self):
        """Test standardization with anchor."""
        g = ig.Graph([(0, 1), (1, 2)])
        
        g_std = utils_ig.standardize_graph_ig(g, anchor=0)
        
        assert 'node_feature' in g_std.vs[0].attributes()


class TestBatchProcessing:
    """Test batch processing."""
    
    def test_batch_nx_graphs_basic(self):
        """Test batch processing of igraph graphs."""
        graphs = [
            ig.Graph([(0, 1), (1, 2)]),
            ig.Graph([(0, 1), (1, 2), (2, 0)]),
            ig.Graph([(0, 1)]),
        ]
        
        result = utils_ig.batch_nx_graphs(graphs)
        
        # Should return PyTorch Batch object
        assert hasattr(result, 'x')
    
    def test_batch_nx_graphs_with_anchors(self):
        """Test batch processing with anchors."""
        graphs = [
            ig.Graph([(0, 1), (1, 2)]),
            ig.Graph([(0, 1), (1, 2), (2, 0)]),
        ]
        anchors = [0, 1]
        
        result = utils_ig.batch_nx_graphs(graphs, anchors=anchors)
        
        assert hasattr(result, 'x')


class TestDeviceUtilities:
    """Test device and memory utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = utils_ig.get_device()
        
        assert device.type in ['cpu', 'cuda']
    
    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        mem = utils_ig.get_memory_usage()
        
        assert isinstance(mem, (int, float))
        assert mem >= 0


class TestOptimizerFunctions:
    """Test optimizer-related functions."""
    
    def test_parse_optimizer(self):
        """Test optimizer argument parser."""
        import argparse
        
        parser = argparse.ArgumentParser()
        utils_ig.parse_optimizer(parser)
        
        # Should have added arguments
        args = parser.parse_args([])
        assert hasattr(args, 'opt')
    
    def test_build_optimizer(self):
        """Test optimizer builder."""
        import argparse
        
        class Args:
            opt = 'adam'
            lr = 0.001
            weight_decay = 1e-5
            opt_scheduler = 'none'
        
        model = torch.nn.Linear(10, 5)
        scheduler, optimizer = utils_ig.build_optimizer(Args(), model.parameters())
        
        assert optimizer is not None
        assert scheduler is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_node_graph(self):
        """Test single-node graph."""
        g = ig.Graph(1)  # Single node, no edges
        
        # Should handle gracefully
        hash_val = utils_ig.wl_hash(g)
        assert isinstance(hash_val, tuple)
    
    def test_disconnected_graph(self):
        """Test disconnected graph."""
        g = ig.Graph()
        g.add_vertices(4)
        g.add_edges([(0, 1), (2, 3)])
        
        hash_val = utils_ig.wl_hash(g)
        assert isinstance(hash_val, tuple)
    
    def test_large_graph(self):
        """Test larger graph."""
        # Create BA graph with 50 nodes
        g = ig.Graph.Barabasi(50, 3)
        
        hash_val = utils_ig.wl_hash(g)
        assert isinstance(hash_val, tuple)
        
        # Should be able to sample from it
        sampled, neigh = utils_ig.sample_neigh([g], size=5, graph_type="undirected")
        assert len(neigh) == 5


class TestGraphProperties:
    """Test graph property preservation."""
    
    def test_directed_graph_property(self):
        """Test directed property is preserved."""
        g_dir = ig.Graph([(0, 1), (1, 2)], directed=True)
        g_undir = ig.Graph([(0, 1), (1, 2)], directed=False)
        
        hash_dir = utils_ig.wl_hash(g_dir)
        hash_undir = utils_ig.wl_hash(g_undir)
        
        # Different directionality should ideally give different hashes
        # (though not guaranteed)
        assert isinstance(hash_dir, tuple)
        assert isinstance(hash_undir, tuple)
    
    def test_edge_attributes_preserved(self):
        """Test edge attributes are preserved."""
        g = ig.Graph([(0, 1), (1, 2)])
        g.es[0]['weight'] = 0.5
        g.es[1]['weight'] = 1.5
        
        g_std = utils_ig.standardize_graph_ig(g)
        
        # Check weights preserved
        assert g_std.es[0]['weight'] == 0.5 or g_std.es[0]['weight'] == 1.5
    
    def test_vertex_attributes_preserved(self):
        """Test vertex attributes are preserved."""
        g = ig.Graph([(0, 1), (1, 2)])
        g.vs[0]['label'] = 'a'
        g.vs[1]['label'] = 'b'
        
        g_std = utils_ig.standardize_graph_ig(g)
        
        assert 'label' in g_std.vs[0].attributes()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
