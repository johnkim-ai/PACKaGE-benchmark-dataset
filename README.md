# PACKaGE: Paragraph-level Annotated Causal Knowledge and Graph–tExt

This repository contains the official implementation and dataset construction pipeline for **PACKaGE**, a benchmark designed to evaluate bidirectional causal reasoning in Large Language Models (LLMs).

Unlike existing sentence-level resources, PACKaGE operates at the **paragraph level**, pairing extended textual narratives with complex, multi-hop Directed Acyclic Graphs (DAGs). The benchmark supports four primary tasks: Text-to-Graph (T2G), Graph-to-Text (G2T), and their cyclic counterparts (T2G2T, G2T2G).

## 📂 Project Structure

- **`data/`**: Released dataset, its uncleaned predecessor, the raw corpus, and the intermediate paragraph file (see [Dataset](#-dataset)).
- **`dataset_build.py`**: Causal pair extraction, graph induction, node condensation (SBERT embeddings + ABTT + FAISS), and paragraph generation (Plain, Role-annotated, Role+Relationship-annotated).
- **`retag_claude.py`**: Places `<source>` / `<leaf>` role tags (and `direct cause of` / `direct effect of` relation headers) on smoothed paragraphs by locating each graph node in the text.
- **`retag_gemini.py`**: Alternative tagger that transfers the tags of the raw RR paragraph onto the smoothed paragraph by fuzzy span matching.
- **`count_rr_tags.py`**: Quality check that counts role tags and header targets per RR paragraph.
- **`strip_filler.py`**: Final cleaning step that removes meta/filler sentences (e.g. "No further details are provided.") from all three paragraph variants.
- **`run_tasks.py`**: Benchmark runner for T2G and G2T with multiple LLM backends (OpenAI, Anthropic, Google, xAI, Groq, local Hugging Face).
- **`configs/g2t2g_ft.yaml`**: Configuration template for defining tasks, models, and runtime environments.

## 📦 Dataset

| File | Rows | Description |
| --- | --- | --- |
| `data/package_dataset_example.csv` | 4,566 | **Released PACKaGE dataset** (filler sentences removed) |
Columns of `package_dataset_example.csv`:

| Column | Content |
| --- | --- |
| `graph_json` | Causal DAG. Rows 0–4164: `{"nodes": [{"id": ...}], "edges": [{"source": ..., "target": ...}]}`. Rows 4165–4565: `{"nodes": [...], "edges": [[u, v], ...]}` |
| `paragraph_plain` | Paragraph without markup |
| `paragraph_role` | Same paragraph with role tags: `<source> ... </source>`, `<leaf> ... </leaf>` (row 0 also uses `<intermediate>`) |
| `paragraph_rr` | Role + relationship tags: `<source: direct cause of X> ... </source>`, `<leaf: direct effect of Y> ... </leaf>` |
| `row_index` | Original row id; filled for 1,617 rows only |

Graphs have 3.7 nodes and 2.8 edges on average (max 24 nodes / 20 edges); the median plain paragraph is 16 words.

## 🚀 Dataset Construction

The pipeline has five stages. Stages 3–4 were partly done outside these scripts, so the outputs of those stages are provided as files.

1. **Collection**: Causality-labeled sentences with `<ARG0>` / `<ARG1>` spans (`data/Papers 2021-2024.csv` for OpenAlex; the Causal News Corpus is not included and must be obtained from its original release).
2. **Graph Induction & Node Condensation** (`dataset_build.py`): Builds cause→effect edges from the `<ARG0>` / `<ARG1>` spans, embeds node texts with SBERT, applies All-but-the-Top (ABTT) reduction, merges near-duplicate nodes with FAISS (cosine ≥ `--dedup_thr`), and writes one paragraph per connected component in three variants (`paragraph_{plain,role,rr}_{raw,smoothed}`). Smoothing calls the OpenAI API when `OPENAI_API_KEY` is set and falls back to rule-based cleanup otherwise.
3. **Second smoothing pass** (script not included): Produces the `*_smoothed_smooth` columns of `data/path_paragraphs_final3.csv`.
4. **Role tagging & curation**: `retag_claude.py` (or `retag_gemini.py`) adds `paragraph_role_smoothed_smooth` / `paragraph_rr_smoothed_smooth`, and `count_rr_tags.py` reports tag statistics. Filtering to 4,566 rows and the final tag revisions were not scripted; the result is `data/package_dataset 4_uncleaned.csv` (4,565 of its 4,566 graphs also appear in `path_paragraphs_final3.csv`).
5. **Filler removal** (`strip_filler.py`): Removes sentences that only comment on the text, such as "No further details are provided.", "The context is unclear." and "These are separate statements." (1,963 sentences per column). Sentences with a causal verb (gave rise to, resulted in, led to, ...) are never removed, and node tags inside a removed sentence are dropped together with it so the remaining tags stay balanced. Running it on `package_dataset 4_uncleaned.csv` reproduces `package_dataset 4.csv` byte for byte.

### Build Commands

```bash
# Stage 2: graphs + paragraph variants from the raw corpus
python3 dataset_build.py \
  --input "data/Papers 2021-2024.csv" \
  --text_col causal_text_w_pairs \
  --out_csv path_paragraphs.csv \
  --sbert_model sentence-transformers/all-MiniLM-L6-v2 \
  --abtt_dim 2 \
  --dedup_thr 0.84 \
  --openai_model gpt-5

# Stage 4: role / relationship tags on the smoothed paragraphs, then tag statistics
python3 retag_claude.py data/path_paragraphs_final3.csv path_paragraphs_tagged.csv
python3 count_rr_tags.py \
  --input_csv path_paragraphs_tagged.csv \
  --output_csv path_paragraphs_tagged_rrcount.csv \
  --rr_col paragraph_rr_raw,paragraph_rr_smoothed_smooth

# Stage 5: filler removal (omit --output for a dry run)
python3 strip_filler.py \
  --input "data/package_dataset 4_uncleaned.csv" \
  --output "data/package_dataset 4.csv"
```

## 🧪 Evaluation Tasks

The benchmark evaluates models on the following tasks:

| Task | Description | Input | Output |
| --- | --- | --- | --- |
| **T2G** | Text to Graph | Paragraph Text | Causal Graph (JSON) |
| **G2T** | Graph to Text | Causal Graph (JSON) | Descriptive Paragraph |
| **G2T2G** | Graph → Text → Graph | Ground Truth Graph | Reconstructed Graph |
| **T2G2T** | Text → Graph → Text | Ground Truth Text | Reconstructed Text |

## ⚡ Running Experiments

`run_tasks.py` in this folder runs T2G and G2T on a CSV/JSON/JSONL file with `text` and `graph` fields and writes the predictions to JSONL:

```bash
export OPENAI_API_KEY="sk-..."
python3 run_tasks.py --task t2g --input data.jsonl --provider openai --model gpt-5 --output_dir runs
```

> **Note:** The YAML-driven runner described below (all four tasks, automatic metrics) is not part of this folder yet; `run_tasks.py` does not accept `--config`.

### 1. Configuration (`configs/g2t2g_ft.yaml`)

Define your target task and models in a YAML file:

```yaml
task: g2t2g
data:
  graph_csv: "data/package_dataset 4.csv"
  graph_col: graph_json
models:
  - provider: openai
    name: gpt-5
  - provider: xai
    name: grok4
runtime:
  max_workers: 16
```

### 2. Execution

```bash
export OPENAI_API_KEY="sk-..."
python3 run_tasks.py --config configs/g2t2g_ft.yaml
```

## 📊 Metrics

The framework calculates the following metrics automatically:

* **Structural Metrics** (for T2G, G2T2G): Precision, Recall, F1, and Exact Match Accuracy for both Nodes and Edges.
* **Textual Metrics** (for G2T, T2G2T):
  * **ROUGE-L**: Measures lexical overlap.
  * **SBERT-cos**: Measures semantic similarity using `all-MiniLM-L6-v2` embeddings.
  * **SBERT-FAISS**: Nearest neighbor semantic retrieval score.

## 📝 Citation

If you use this dataset or code, please cite our paper:

```bibtex
@article{anonymous2025package,
  title={From Text to Structure: The PACKaGE Benchmark for Bidirectional Causal Reasoning in Paragraphs and Graphs},
  author={Anonymous},
  journal={Under Review},
  year={2025}
}
```

## ⚖️ License

The code is released under the MIT License. The dataset leverages data from OpenAlex (CC BY 4.0) and the Causal News Corpus (CC0).
