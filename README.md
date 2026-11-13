# Advanced Information Retrieval: Coursework

Coursework for a university **Advanced Information Retrieval** course (winter semester 2025/26). It contains guided lab tutorials and two graded projects that build **multi-stage neural ranking pipelines** with [PyTerrier](https://github.com/terrier-org/pyterrier) and [lightning-ir](https://github.com/webis-de/lightning-ir).

| Folder | What it is |
|---|---|
| [`tutorials/`](tutorials/) | Lab notebooks on PyTerrier, PyTorch, lightning-ir, cascade rankers and word embeddings |
| [`project1/`](project1/) | **Project 1:** a cascade ranker (BM25 → cross-encoder) on TREC DL 2020 |
| [`advanced-ir-project-part2/`](advanced-ir-project-part2/) | **Project 2:** a retrieval pipeline for noisy web documents (ClueWeb12 subsample), with ablations, LLM query expansion and energy measurement |

Each project folder holds the notebook (code, outputs and written discussion) and the exported PDF that was submitted.

---

## Highlights

### Project 1: cascade ranking on TREC DL 2020

A two-stage retrieve-then-rerank pipeline:

```
BM25 (top 1000)  →  fetch passage text  →  cross-encoder webis/monoelectra-base  →  top 100
```

Evaluated on `msmarco-passage/trec-dl-2020/judged` (54 queries):

| System | RR | nDCG@10 | nDCG@100 | AP@100 |
|---|---|---|---|---|
| BM25 (cut at 100) | 0.793 | 0.481 | 0.488 | 0.302 |
| **BM25 >> Cross-Encoder** | **0.937** | **0.743** | **0.683** | **0.508** |

### Project 2: retrieval for noisy web documents

The corpus is a ClueWeb12 subsample: 84k web pages averaging about 2,100 words, full of navigation bars, footers and spam. The notebook includes a configurable pipeline builder, `run_toggled_experiment(...)`, that switches each stage on or off:

1. **First stage:** BM25 or BM25F
2. **Query expansion:** RM3 (optional)
3. **Dense retrieval:** bi-encoder `webis/bert-bi-encoder`, used either alone or fused with BM25 through Reciprocal Rank Fusion (RRF)
4. **Re-ranking:** a cross-encoder (`webis/monoelectra-base` or `-large`) over **sliding windows** of 300 tokens with a stride of 150, aggregated with max-passage scoring

Ablation results (50 topics):

| Pipeline | RR | P@10 | nDCG@10 | nDCG@100 |
|---|---|---|---|---|
| BM25 baseline | 0.550 | 0.504 | 0.306 | 0.363 |
| Project 1 cascade (no passaging) | 0.587 | 0.350 | 0.254 | 0.294 |
| Bi-encoder only | 0.523 | 0.314 | 0.230 | 0.250 |
| BM25 + BiEnc (RRF) + CE-base | 0.625 | 0.424 | 0.307 | 0.359 |
| BM25 + BiEnc (RRF) + CE-large | 0.639 | 0.428 | 0.316 | 0.362 |
| BM25 + RM3 + BiEnc (RRF) + CE-large | 0.592 | 0.390 | 0.264 | 0.303 |
| Run-level RRF of (BM25+CE) and BiEnc | 0.709 | 0.454 | 0.342 | 0.374 |
| LLM query expansion (HyDE, Llama-3.1-70B) + CE-large | 0.596 | 0.362 | 0.254 | 0.290 |
| **BM25 (top 100) + CE-large, sliding window** *(final)* | **0.719** | **0.526** | **0.375** | **0.390** |

Findings:
- Without passaging, the Project 1 cascade does *worse* than BM25 on long web pages. Splitting pages into sliding windows and max-pooling the passage scores is what makes the cross-encoder work here.
- The bi-encoder, trained on clean MS MARCO questions, does not transfer well to noisy web text, and adding it through fusion did not help.
- Neither RM3 nor LLM-generated query expansion helped.
- **Efficiency cost:** measured with [CodeCarbon](https://github.com/mlco2/codecarbon), the final pipeline takes about **4.7 s per query** and emitted about 13.7 g CO₂ over the 50-query run.

The notebook's discussion section covers the design reasoning, limitations and the effectiveness/efficiency trade-offs in detail.

---

## Repository layout

```
.
├── tutorials/
│   ├── tutorial1/advanced-ir-lab-pyterrier.ipynb        # PyTerrier basics: indexing, retrieval, evaluation
│   ├── tutorial2/advanced-ir-lab-pytorch.ipynb          # Intro to PyTorch
│   ├── tutorial2/advanced-ir-lab-lightning-ir.ipynb     # Neural rankers with lightning-ir
│   ├── tutorial3/advanced-ir-lab-cascade-rankers.ipynb  # Combining rankers into cascades
│   └── tutorial3/advanced-ir-lab-word-embeddings.ipynb  # BOW → word2vec → transformer embeddings
├── project1/
│   ├── advanced-ir-project-part1.ipynb
│   └── lightning_ir_pyterrier.py
└── advanced-ir-project-part2/
    ├── advanced-ir-project-part2.ipynb
    └── lightning_ir_pyterrier.py
```

`lightning_ir_pyterrier.py` is a small adapter, provided with the course, that wraps lightning-ir `BiEncoder` / `CrossEncoder` models as PyTerrier transformers so they can be chained with `>>`. Each folder has its own copy with small version differences (for example, the Project 2 copy adds `@torch.no_grad()` and keeps the query order stable). Use the copy next to the notebook you are running.

---

## Running the notebooks

### Requirements

- Python 3.12 (the notebooks were run in a conda environment)
- Java 11 or newer (needed by PyTerrier / Terrier)
- A GPU is strongly recommended for the cross-encoder stages. The code picks CUDA, then Apple MPS, then CPU.

### Install

Each notebook's first cell installs what it needs. Roughly:

```bash
pip install python-terrier pyterrier-alpha ir-datasets lightning-ir torch transformers \
            sentence-transformers lsr-benchmark openai codecarbon
# Project 2 dataset loader:
pip install "git+https://github.com/tira-io/tira@d914806859f487037e7193a0773895937f39e8de#subdirectory=python-client"
```

### Data

- **Project 1** downloads its data automatically through `ir_datasets` (`msmarco-passage`, `trec-dl-2020`). The full MS MARCO passage corpus is large, so building the BM25 index takes a while the first time.
- **Project 2** uses a **ClueWeb12 subsample that is not publicly redistributable**. It was loaded from a course-provided `__shared/clueweb12.zip` on the university's JupyterHub cluster. Without access to ClueWeb12 you can read the code and results but not re-run the notebook.

Indexes (`./index_*`, `./indices/`) and run files (`run_*.res`, `results_*.csv`) are written next to the notebook and are not part of the repository.

### LLM query expansion (Project 2, optional)

The HyDE / keyword-expansion experiment calls an OpenAI-compatible endpoint (GWDG Academic Cloud). To run it, create a file named `secret.py` next to the notebook:

```python
GWDG_API_KEY = "your-key-here"
```

**Never commit this file.** All other experiments work without it.

---

## Acknowledgements

- Assignment prompts, tutorial notebooks and the `lightning_ir_pyterrier.py` adapter were provided by the course instructors. The project implementations, experiments and written discussion are my own.
- Models: [`webis/monoelectra-base`](https://huggingface.co/webis/monoelectra-base), [`webis/monoelectra-large`](https://huggingface.co/webis/monoelectra-large) and [`webis/bert-bi-encoder`](https://huggingface.co/webis/bert-bi-encoder) from the Webis group.
- Libraries: PyTerrier, lightning-ir, ir_datasets, TIRA and CodeCarbon.
