# Fair Benchmark - Step by Step Guide

## 🎯 Goal
Run a **fair, unbiased comparison** of standard vs batch processing on a large graph.

---

## ⚖️ **Fairness Principles**

1. ✅ **Identical parameters** for both modes
2. ✅ **Same dataset** for both runs
3. ✅ **Same hardware** (your server)
4. ✅ **Only difference**: `streaming_workers` setting

---

## 📋 **Step 1: Prepare Your Server**

### SSH into your server:
```bash
ssh your-server
cd /path/to/neural-subgraph-matcher-miner
```

### Create a benchmark directory:
```bash
mkdir -p benchmark
cd benchmark
```

---

## 📥 **Step 2: Download LiveJournal Dataset**

```bash
# Download (1.7GB compressed, ~5 minutes)
wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz

# Decompress (~2 minutes)
gunzip soc-LiveJournal1.txt.gz

# Verify
ls -lh soc-LiveJournal1.txt
# Should show: ~2.5GB, 68M lines
```

---

## 🔄 **Step 3: Convert to Pickle**

Create `convert.py`:
```python
import networkx as nx
import pickle
import time

print("Converting LiveJournal to pickle format...")
start = time.time()

G = nx.DiGraph()

with open('soc-LiveJournal1.txt', 'r') as f:
    for i, line in enumerate(f):
        if not line.startswith('#'):
            try:
                u, v = line.strip().split()
                G.add_edge(int(u), int(v))
            except:
                continue
        
        if i % 1000000 == 0:
            print(f"  {i:,} lines processed...")

print(f"\n✓ Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"  Time: {time.time()-start:.1f}s")

print("\nSaving pickle...")
with open('livejournal.pkl', 'wb') as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✓ Done!")
```

Run it:
```bash
python convert.py
# Takes ~10-15 minutes
```

**✅ Checkpoint:** You should now have `livejournal.pkl` (~1.5GB)

---

## 🧪 **Step 4: Run Standard Mode**

```bash
docker run --rm \
  -v $(pwd):/app \
  -e PYTHONUNBUFFERED=1 \
  --name standard_benchmark \
  samribahta/decoder-image:latest \
  bash -c "
    echo '=== STANDARD MODE BENCHMARK ==='
    echo 'Start time:' \$(date)
    
    python -m subgraph_mining.decoder \
      --dataset=/app/livejournal.pkl \
      --model_path=ckpt/model.pt \
      --n_neighborhoods=50000 \
      --n_trials=1000 \
      --min_pattern_size=3 \
      --max_pattern_size=8 \
      --min_neighborhood_size=50 \
      --max_neighborhood_size=100 \
      --search_strategy=greedy \
      --out_batch_size=3 \
      --node_anchored \
      --streaming_workers=1 \
      --auto_streaming_threshold=10000000 \
      --out_path=/app/results/standard_results.pkl
    
    echo 'End time:' \$(date)
  " 2>&1 | tee standard_mode.log
```

**What to expect:**
- Either crashes with OOM
- Or takes a very long time
- Monitor with: `docker stats standard_benchmark` (in another terminal)

**⏱️ Time estimate:** 30-60 minutes (or crash)

---

## 🚀 **Step 5: Run Batch Processing Mode**

**IMPORTANT:** Use **identical parameters** as standard mode!

```bash
docker run --rm \
  -v $(pwd):/app \
  -e PYTHONUNBUFFERED=1 \
  --name batch_benchmark \
  samribahta/decoder-image:latest \
  bash -c "
    echo '=== BATCH PROCESSING MODE BENCHMARK ==='
    echo 'Start time:' \$(date)
    
    python -m subgraph_mining.decoder \
      --dataset=/app/livejournal.pkl \
      --model_path=ckpt/model.pt \
      --n_neighborhoods=50000 \
      --n_trials=1000 \
      --min_pattern_size=3 \
      --max_pattern_size=8 \
      --min_neighborhood_size=50 \
      --max_neighborhood_size=100 \
      --search_strategy=greedy \
      --out_batch_size=3 \
      --node_anchored \
      --streaming_workers=4 \
      --auto_streaming_threshold=100000 \
      --out_path=/app/results/batch_results.pkl
    
    echo 'End time:' \$(date)
  " 2>&1 | tee batch_mode.log
```

**What to expect:**
- Completes successfully
- Constant memory usage
- Monitor with: `docker stats batch_benchmark` (in another terminal)

**⏱️ Time estimate:** 45-90 minutes

---

## 📊 **Step 6: Compare Results**

```bash
# Check which runs completed
ls -lh results/

# Compare logs
echo "=== STANDARD MODE ==="
tail -20 standard_mode.log

echo "=== BATCH MODE ==="
tail -20 batch_mode.log

# Extract timing info
grep "Total time" standard_mode.log
grep "Total time" batch_mode.log
```

---

## 📈 **Step 7: Analyze Performance**

Create `analyze_results.py`:
```python
import pickle
import os

print("=" * 70)
print("BENCHMARK RESULTS COMPARISON")
print("=" * 70)

# Check standard mode
standard_exists = os.path.exists('results/standard_results.pkl')
batch_exists = os.path.exists('results/batch_results.pkl')

print(f"\nStandard Mode: {'✓ COMPLETED' if standard_exists else '✗ FAILED'}")
print(f"Batch Mode:    {'✓ COMPLETED' if batch_exists else '✗ FAILED'}")

if standard_exists:
    with open('results/standard_results.pkl', 'rb') as f:
        standard_patterns = pickle.load(f)
    print(f"\nStandard Mode found {len(standard_patterns)} patterns")

if batch_exists:
    with open('results/batch_results.pkl', 'rb') as f:
        batch_patterns = pickle.load(f)
    print(f"Batch Mode found {len(batch_patterns)} patterns")

print("\n" + "=" * 70)
```

Run it:
```bash
python analyze_results.py
```

---

## ⚖️ **What Makes This Fair?**

| Aspect | Standard | Batch | Fair? |
|--------|----------|-------|-------|
| Dataset | livejournal.pkl | livejournal.pkl | ✅ Same |
| Neighborhoods | 50,000 | 50,000 | ✅ Same |
| Trials | 1,000 | 1,000 | ✅ Same |
| Pattern sizes | 3-8 | 3-8 | ✅ Same |
| Neighborhood size | 50-100 | 50-100 | ✅ Same |
| Hardware | Your server | Your server | ✅ Same |
| **Only difference** | workers=1 | workers=4 | ⚖️ This is what we're testing! |

---

## 🎯 **Expected Outcomes**

### **Scenario 1: Standard Mode Crashes**
```
Standard Mode: ✗ FAILED (out of memory)
Batch Mode:    ✓ COMPLETED in 60 minutes

Conclusion: Batch processing enables mining on large graphs
```

### **Scenario 2: Both Complete**
```
Standard Mode: ✓ COMPLETED in 90 minutes
Batch Mode:    ✓ COMPLETED in 60 minutes

Conclusion: Batch processing is 33% faster
```

### **Scenario 3: Standard Mode Slower**
```
Standard Mode: ✓ COMPLETED in 120 minutes
Batch Mode:    ✓ COMPLETED in 60 minutes

Conclusion: Batch processing is 2x faster
```

---

## 📝 **What to Report**

After both runs, you can say:

> "Tested on LiveJournal (4.8M nodes, 69M edges) with identical parameters:
> - Standard mode: [result]
> - Batch processing: [result]
> 
> Batch processing [enables/improves] mining on large graphs."

**This is unbiased, measurable proof!** 🎯

---

## 🔧 **Troubleshooting**

### Standard mode crashes too quickly?
- Reduce to `--n_neighborhoods=10000`
- This still proves the point!

### Want faster testing?
- Use `--n_trials=500` for both
- Use `--n_neighborhoods=10000` for both
- Still fair as long as both use same values!

### Monitor memory usage:
```bash
# In another terminal
watch -n 1 'docker stats --no-stream'
```

---

## ✅ **Success Checklist**

- [ ] Downloaded LiveJournal dataset
- [ ] Converted to pickle format
- [ ] Ran standard mode (noted result)
- [ ] Ran batch mode (noted result)
- [ ] Compared results
- [ ] Have proof of difference!

**Now you have fair, unbiased evidence!** 🚀
