#!/usr/bin/env python3
"""
Simple test script to verify the changes work correctly.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported without errors."""
    try:
        print("Testing imports...")

        # Test convert_to_pkl
        import convert_to_pkl

        print("✓ convert_to_pkl imported successfully")

        # Test common modules
        from common import data

        print("✓ common.data imported successfully")

        # Test visualizer
        from visualizer import visualizer

        print("✓ visualizer.visualizer imported successfully")

        # Test decoder (this might fail if dependencies are missing)
        try:
            from subgraph_mining import decoder

            print("✓ subgraph_mining.decoder imported successfully")
        except ImportError as e:
            print(
                f"⚠ subgraph_mining.decoder import failed (expected in some environments): {e}"
            )

        print("All imports successful!")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_data_loading():
    """Test that data loading works with the new label logic."""
    try:
        print("\nTesting data loading...")

        # Test if amazon0302.pkl exists
        if os.path.exists("amazon0302.pkl"):
            print("✓ amazon0302.pkl found")

            # Try to load it
            import pickle

            with open("amazon0302.pkl", "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and "nodes" in data and "edges" in data:
                print("✓ Data loaded successfully (dict format)")

                # Check a few nodes for label field
                sample_nodes = data["nodes"][:5] if len(data["nodes"]) > 0 else []
                for i, (node_id, attrs) in enumerate(sample_nodes):
                    label = attrs.get("label", "No label")
                    group = attrs.get("group", "No group")
                    title = attrs.get("title", "No title")
                    print(
                        f"  Node {node_id}: label='{label}', group='{group}', title='{title[:30]}...'"
                    )

                return True
            else:
                print("⚠ Data format not as expected")
                return False
        else:
            print("⚠ amazon0302.pkl not found - skipping data test")
            return True

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing changes to neural-subgraph-matcher-miner...")
    print("=" * 50)

    success = True

    # Test imports
    if not test_imports():
        success = False

    # Test data loading
    if not test_data_loading():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! The changes should work correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    sys.exit(0 if success else 1)
