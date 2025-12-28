"""
Analyze and compare benchmark results.
Shows which mode completed and performance metrics.
"""

import pickle
import os
import json
from pathlib import Path

def main():
    print("=" * 70)
    print("BENCHMARK RESULTS ANALYSIS")
    print("=" * 70)
    print()
    
    # Check for results
    standard_path = 'results/standard_results.pkl'
    batch_path = 'results/batch_results.pkl'
    
    standard_exists = os.path.exists(standard_path)
    batch_exists = os.path.exists(batch_path)
    
    print("COMPLETION STATUS:")
    print(f"  Standard Mode:  {'✓ COMPLETED' if standard_exists else '✗ FAILED/CRASHED'}")
    print(f"  Batch Mode:     {'✓ COMPLETED' if batch_exists else '✗ FAILED/CRASHED'}")
    print()
    
    # Analyze standard mode
    if standard_exists:
        print("STANDARD MODE RESULTS:")
        try:
            with open(standard_path, 'rb') as f:
                standard_patterns = pickle.load(f)
            
            print(f"  Patterns found: {len(standard_patterns)}")
            
            # Count by size
            from collections import Counter
            sizes = Counter([len(p) for p in standard_patterns])
            print(f"  Pattern sizes:")
            for size in sorted(sizes.keys()):
                print(f"    Size {size}: {sizes[size]} patterns")
            
            file_size = os.path.getsize(standard_path) / (1024**2)  # MB
            print(f"  Output file: {file_size:.2f} MB")
        except Exception as e:
            print(f"  Error loading results: {e}")
        print()
    
    # Analyze batch mode
    if batch_exists:
        print("BATCH PROCESSING MODE RESULTS:")
        try:
            with open(batch_path, 'rb') as f:
                batch_patterns = pickle.load(f)
            
            print(f"  Patterns found: {len(batch_patterns)}")
            
            # Count by size
            from collections import Counter
            sizes = Counter([len(p) for p in batch_patterns])
            print(f"  Pattern sizes:")
            for size in sorted(sizes.keys()):
                print(f"    Size {size}: {sizes[size]} patterns")
            
            file_size = os.path.getsize(batch_path) / (1024**2)  # MB
            print(f"  Output file: {file_size:.2f} MB")
        except Exception as e:
            print(f"  Error loading results: {e}")
        print()
    
    # Extract timing from logs
    print("TIMING INFORMATION:")
    
    if os.path.exists('standard_mode.log'):
        print("  Standard Mode:")
        with open('standard_mode.log', 'r') as f:
            log = f.read()
            if 'Total time:' in log:
                for line in log.split('\n'):
                    if 'Total time:' in line:
                        print(f"    {line.strip()}")
            else:
                print("    No timing info found (likely crashed)")
    
    if os.path.exists('batch_mode.log'):
        print("  Batch Mode:")
        with open('batch_mode.log', 'r') as f:
            log = f.read()
            if 'Total time:' in log:
                for line in log.split('\n'):
                    if 'Total time:' in line:
                        print(f"    {line.strip()}")
            else:
                print("    No timing info found")
    
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if not standard_exists and batch_exists:
        print("✓ Batch processing ENABLES mining on large graphs")
        print("  Standard mode crashed, batch mode completed successfully")
    elif standard_exists and batch_exists:
        print("✓ Both modes completed")
        print("  Compare timing above to see performance difference")
    elif standard_exists and not batch_exists:
        print("⚠ Unexpected: Standard completed but batch failed")
        print("  Check batch_mode.log for errors")
    else:
        print("✗ Both modes failed")
        print("  Check logs for errors")
    
    print()

if __name__ == "__main__":
    main()
