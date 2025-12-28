"""
Convert LiveJournal edge list to NetworkX pickle format.
Fair benchmark preparation script.
"""

import networkx as nx
import pickle
import time

def main():
    print("=" * 70)
    print("CONVERTING LIVEJOURNAL TO PICKLE FORMAT")
    print("=" * 70)
    print()
    
    input_file = 'soc-LiveJournal1.txt'
    output_file = 'livejournal.pkl'
    
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()
    
    start_time = time.time()
    
    # Create directed graph
    G = nx.DiGraph()
    
    print("Reading edges...")
    line_count = 0
    edge_count = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            line_count += 1
            
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Parse edge
            try:
                u, v = line.strip().split()
                G.add_edge(int(u), int(v))
                edge_count += 1
            except:
                continue
            
            # Progress update
            if line_count % 1000000 == 0:
                elapsed = time.time() - start_time
                print(f"  {line_count:,} lines processed ({edge_count:,} edges) - {elapsed:.1f}s")
    
    print()
    print(f"✓ Graph loaded:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Time:  {time.time() - start_time:.1f}s")
    print()
    
    # Save pickle
    print(f"Saving to {output_file}...")
    save_start = time.time()
    
    with open(output_file, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Saved in {time.time() - save_start:.1f}s")
    print()
    
    # Verify
    import os
    file_size = os.path.getsize(output_file) / (1024**3)  # GB
    print(f"File size: {file_size:.2f} GB")
    print()
    
    print("=" * 70)
    print("✓ CONVERSION COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Run standard mode benchmark")
    print("2. Run batch processing benchmark")
    print("3. Compare results")
    print()

if __name__ == "__main__":
    main()
