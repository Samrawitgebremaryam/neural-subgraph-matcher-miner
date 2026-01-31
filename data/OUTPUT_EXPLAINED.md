# What the decoder output means

## What we are assessing

We are **assessing SPMiner** (the neural subgraph miner) using a **synthetic bio dataset**:

- **Graph:** 2500 nodes (500 each: TF, Gene, mRNA, Enzyme, Metabolite), 899 directed edges with labels (regulates, transcribes, translates, catalyzes).
- **Planted ground truth:** 100 copies of the 5-node **golden motif** (Central Dogma chain):  
  **TF --regulates→ Gene --transcribes→ mRNA --translates→ Enzyme --catalyzes→ Metabolite**.
- **Question:** Does SPMiner discover this motif and rank it as a top frequent pattern?

The decoder output tells us: (1) which patterns SPMiner found, (2) how many instances of each, (3) whether the golden motif appears and at what rank.

---

## Summary of this run

SPMiner loaded the bio graph, sampled **500 neighborhoods** (2–5 nodes each), ran **200 greedy trials**, and ranked patterns by **count** (how often each appeared). It found **9 pattern types** (2 at size 3, 3 at size 4, 4 at size 5) and **84 unique instances** total. Edge labels appear in the HTML visualizations under `plots/cluster/size_*_rank_*/`.

---

## What each part of the log means

| Log line | Meaning |
|----------|--------|
| **Created directed graph ... 2500 nodes and 899 edges** | Your bio graph was loaded; 899 unique edges after deduplication. |
| **Adaptive Mode: Standard Sequential Processing** | Graph is small enough that streaming/batch mode is not used. |
| **500/500** (progress bar) | 500 neighborhoods were sampled (each 2–5 nodes). |
| **Number of graphs not multiple of batch size** | Harmless warning: 500 is not divisible by default batch size. |
| **200/200** (trials) | 200 greedy search trials finished. |
| **Ranking patterns of size 3/4/5 using method: 'counts'** | Patterns are ordered by how many times they appeared (support). |
| **Total time: ~8s** | Full mining run time. |

---

## Size 3 (3-node patterns)

| Rank | Meaning |
|------|--------|
| **size_3_rank_1** (27 instances) | Most frequent 3-node motif: **Enzyme–Gene–mRNA** with edges **transcribes, translates** (Gene→mRNA→Enzyme). |
| **size_3_rank_2** (7 instances) | Second motif: **Enzyme–Metabolite** with edge **catalyzes**. |

---

## Size 4 (4-node patterns)

| Rank | Meaning |
|------|--------|
| **size_4_rank_1** (17 instances) | **Enzyme–Gene–Metabolite–mRNA** with **catalyzes, transcribes, translates** — a 4-step chain. |
| **size_4_rank_2** (6 instances) | **Enzyme–Metabolite–mRNA** with **catalyzes, translates**. |
| **size_4_rank_3** (4 instances) | Same node set, **catalyzes, translates**. |

---

## Size 5 (5-node patterns) — Golden motif

| Rank | Instances (example run) | Meaning |
|------|--------------------------|--------|
| **size_5_rank_1** (e.g. 7) | 4-step chain (Enzyme–Gene–Metabolite–mRNA) — **catalyzes, transcribes, translates** (no TF). |
| **size_5_rank_2** (e.g. 6) | **Golden motif:** *Enzyme–Gene–Metabolite–TF–mRNA* — **catalyzes, regulates, transcribes, translates**. All 5 node types and all 4 edge types: TF→Gene→mRNA→Enzyme→Metabolite. |
| **size_5_rank_3** (e.g. 5) | 4-step chain, 3 edge types. |
| **size_5_rank_4** (e.g. 2) | Enzyme–Gene–TF–mRNA, **regulates, transcribes, translates** (no Metabolite). |
| **size_5_rank_5** (if present) | Other 5-node patterns. |

**size_5_rank_2** is the full Central Dogma chain with edge labels. Rank 1 is often a more frequent 4-step subchain (no TF).

---

## Output files and artifacts

| Output | Meaning |
|--------|--------|
| **results/patterns.pkl** | One representative graph per pattern (9 in this run). |
| **results/patterns.json** | Same representatives in JSON (nodes, edges, labels). |
| **results/patterns_all_instances.pkl** | All 84 unique discovered instances (this run). |
| **results/patterns_all_instances.json** | Same instances in JSON. |
| **plots/cluster/size_3_rank_1/** | Folder for size-3 rank-1 pattern: **index.html** (overview), **representative.html** (one example), **instance_0001.html** … **instance_0025.html** (each discovered instance). |
| **plots/cluster/size_5_rank_2/** | Same for the **golden motif** (5-node Central Dogma chain): representative + 6 instance HTMLs in this run. |
| **decoder-plots.zip** (GitHub artifact) | Zipped **plots/** and **results/** (e.g. 106 files): all HTML visualizations and pattern PKL/JSON. Download from the Actions run to inspect. |

**Total discoveries / unique instances:** e.g. "Total discoveries: 86, Unique instances: 84" means 86 subgraph hits before deduping, 84 after; these are **across all pattern types**, not just the golden motif.

---

## This run's assessment (example)

- **Golden motif:** **size_5_rank_2** (6 instances in this run).
- **Golden motif recovery rate:** 6 / 100 = **6%** (6 of the 100 planted copies were found as instances of this pattern).
- **Did SPMiner discover the motif?** **Yes.** The full Central Dogma chain (TF→Gene→mRNA→Enzyme→Metabolite with all four edge types) appears as the **second** 5-node pattern (rank 2). Rank 1 at size 5 is a 4-step subchain (no TF) that appeared slightly more often in the sampled neighborhoods.
- **84 unique instances** = total across all pattern types (3-, 4-, and 5-node); do **not** use 84/100 as the golden-motif recovery rate.

---

## Golden motif check

The **golden motif** is the 5-node chain:  
**TF --regulates→ Gene --transcribes→ mRNA --translates→ Enzyme --catalyzes→ Metabolite**.

**size_5_rank_2** (`dir_5-2_nodes-Enzyme-Gene-Metabolite-TF-mRNA_edges-catalyzes-regulates-transcribes-translates_...`) is that pattern: all four edge types in the correct biological order. SPMiner recovered the planted Central Dogma “sentence”; it appears as the **second** 5-node pattern (rank 2); rank 1 is an even more frequent 4-step subchain.

---

## Assessment: correct recovery rate (do not use 82%)

**We planted:** 100 copies of the **same** 5-node golden chain (same structure, different node IDs). So there are 100 instances of the golden motif in the graph. Planting 100 copies of one pattern does **not** violate anything; it is the intended design.

**"82 unique instances":** This is the **total** number of unique subgraph instances discovered across **all** pattern types (sizes 3, 4, and 5). It is **not** the number of golden motifs recovered.

- 82 = sum of (size_3 instances) + (size_4 instances) + (size_5 instances) for all ranks.
- So 82 includes e.g. 27+7 (size 3) + 17+6+4 (size 4) + 10+5+3+2+1 (size 5).

**Golden motif recovery:** Only **size_5_rank_2** is the full golden motif. Its instance count (e.g. 5 or 10 in a run) is the number of **recovered golden motifs**.

- **Correct recovery rate** = (size_5_rank_2 instances) / 100.
- Example: 5 instances → 5% recovery; 10 instances → 10% recovery.
- **Do not use** 82/100 = 82% — that wrongly treats total discoveries as golden-motif recoveries.

---

## Data validity: is the graph we gave SPMiner correct?

**Yes.** The synthetic bio graph is correct and appropriate for assessing SPMiner.

- **Structure:** Directed heterogeneous graph (2,500 nodes, 5 types, edge types: regulates, transcribes, translates, catalyzes). Matches what SPMiner expects (e.g. `--graph_type directed`).
- **Balance:** 500 nodes per type and noise edges that follow biological order (t → t+1 only). The golden motif is the only 5-node pattern we intentionally repeated; it is not an artifact of skewed node counts.
- **The 100 “copies” are intentionally identical (isomorphic).**  
  SPMiner finds **frequent isomorphic subgraphs**: one pattern type with many **instances**. So 100 copies of the same 5-node chain (same structure, same node/edge labels) = **one** pattern with **100 instances** in the graph. That is the intended design. If the 100 copies had different structure (e.g. different edge directions or types), they would be different patterns and would not be counted as one frequent motif. Our setup is therefore correct for the question: “Does SPMiner re-discover the planted Central Dogma motif?”
- **Bias:** The graph is not biased in a misleading way. Rank 1 at size 5 often being a 4-step subchain (no TF) is expected: that subgraph appears in more sampled neighborhoods (golden chains + noise), so SPMiner correctly reports it as more frequent. The full golden motif (size_5_rank_2) is still discovered and ranked as the second 5-node pattern.
