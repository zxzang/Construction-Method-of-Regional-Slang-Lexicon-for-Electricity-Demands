# Construction Method of Regional Slang Lexicon for Electricity Demands

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A statistical-anchor-driven active learning framework for constructing a regional slang lexicon from noisy short texts in low-resource dialect settings. Built for power-grid customer service NLP.

## Overview

Power-grid customer service increasingly relies on WeChat groups where users report outages, billing issues, and equipment faults in regional dialect (Yuxi, Yunnan). Standard NLP tools fail on dialect morphology, and manual lexicon construction is prohibitively expensive.

This framework constructs a power-domain dialect lexicon through a four-stage pipeline:

1. **Candidate Generation** — PMI-based bigram extraction anchored on 16 Yuxi dialect markers with soft-label bidirectional anchoring
2. **Semantic Classification** — SBERT + wordhood scoring with FAISS-accelerated similarity search
3. **Active Learning** — Three query strategies (Margin / TD-CAL / Random) × two oracle types (Rule / RAG-LLM)
4. **Dual-Track Evaluation** — Independent LLM evaluator calibrated against human expert annotations via Cohen's κ

## Directory Structure

```
.
├── experiments/              # Experiment code and results
│   ├── config.py             # Hyperparameters, anchors, seeds, experiment definitions
│   ├── data_loader.py        # CSV loading, noise cleaning, vocabulary parsing
│   ├── candidate_generator.py # PMI/NPMI extraction, bidirectional anchoring, fragment filter
│   ├── semantic_mapper.py    # SBERT + FAISS + wordhood scoring
│   ├── oracle.py             # Rule-based and RAG-LLM oracles, independent evaluator
│   ├── active_learning.py    # AL loop orchestrator (Margin / TD-CAL / Random)
│   ├── evaluation.py         # DP/PDP metrics, Cohen's κ, calibration sampling
│   ├── run_all.py            # Main pipeline (Phase 1–6), post-hoc evaluation
│   ├── requirements.txt      # Python dependencies
│   ├── .env.example          # Environment variable template
│   └── knowledge_base/       # Power-domain KB (auto-generated cache)
│
└── figures/                  # Architecture diagrams
```

## Installation

### Prerequisites

- Python 3.10+
- [pandoc](https://pandoc.org/) (optional, for PDF generation)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd <project-dir>

# Install dependencies
pip install -r experiments/requirements.txt

# Configure API keys
cp experiments/.env.example experiments/.env
# Edit experiments/.env and fill in your OPENAI_API_KEY
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | DeepSeek API key (shared by oracle and evaluator) |
| `OPENAI_BASE_URL` | No | `https://api.deepseek.com` | API endpoint |
| `LLM_MODEL` | No | `deepseek-v4-flash` | Oracle model |
| `EVAL_LLM_MODEL` | No | `deepseek-v4-pro` | Evaluator model |
| `EVAL_LLM_API_KEY` | No | — | Separate key for cross-model evaluation |
| `EVAL_LLM_BASE_URL` | No | `https://api.deepseek.com` | Evaluator endpoint |
| `PYTHONHASHSEED` | Yes | `42` | Set before starting Python for reproducibility |
| `HF_ENDPOINT` | No | `https://hf-mirror.com` | HuggingFace mirror (China mainland) |

Load with: `set -a && source experiments/.env && set +a`

## Usage

### Quick Start

```bash
# List available experiment configurations
python experiments/run_all.py --list

# Run all experiments + evaluation
python experiments/run_all.py

# Run a single experiment
python experiments/run_all.py --exp proposed_tdcal_llm

# Run experiments only (skip LLM evaluation)
python experiments/run_all.py --skip-eval

# Run with custom random seed
python experiments/run_all.py --seed 123
```

### Calibration (Cohen's Kappa)

```bash
# Step 1: Run experiments + Phase 6 (generates calibration_template.csv)
python experiments/run_all.py

# Step 2: Have a human expert annotate experiments/results/calibration_template.csv
#         Save as experiments/results/calibration_annotated.csv

# Step 3: Compute Cohen's Kappa
python experiments/run_all.py --annotations experiments/results/calibration_annotated.csv
```

### Diagnostics

```bash
# Enable Phase 3.5 oracle labeling (diagnosis mode — costs extra API calls)
python experiments/run_all.py --diagnose-phase35
```

## Experiment Configurations

| Key | AL Strategy | Oracle | Description |
|-----|------------|--------|-------------|
| `baseline_margin_rule` | Margin | Rule | Soft-label baseline |
| `ablation_margin_llm` | Margin | RAG-LLM | Isolates LLM oracle contribution |
| `ablation_tdcal_rule` | TD-CAL | Rule | Isolates TD-CAL contribution |
| `random_llm` | Random | RAG-LLM | Active learning lower bound |
| `proposed_tdcal_llm` | TD-CAL | RAG-LLM | Full proposed method |
| `ablation_hard_gate_rule` | Margin | Rule | Hard power-gate ablation |

## Metrics

| Metric | Definition | Evaluated On |
|--------|-----------|-------------|
| **DP** (Dialect Precision) | Proportion of terms with dialect features | `_dialect_lexicon.json` |
| **PDP** (Power-Dialect Precision) | Proportion with both dialect and power relevance | `_power_dialect_lexicon.json` |
| **\|L_d\|** | Dialect lexicon size | `_dialect_lexicon.json` |
| **\|L_pd\|** | Power-dialect lexicon size | `_power_dialect_lexicon.json` |
| **Cohen's κ** | LLM-human agreement (dialect + power dimensions) | Calibration set (Phase 6) |

## Key Results (Run #26)

| Model | DP | PDP | L_d | L_pd | Lexicon | Queries |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| Margin + Rule (Soft-Label baseline) | 1.0000 | 0.1600 | 25 | 4 | 25 | 15 |
| Margin + RAG-LLM Oracle | 1.0000 | 0.1250 | 32 | 4 | 32 | 15 |
| TD-CAL + Rule Oracle | 1.0000 | 0.1481 | 27 | 4 | 27 | 15 |
| TD-CAL + RAG-LLM Oracle (Proposed) | 1.0000 | 0.1250 | 32 | 4 | 32 | 15 |
| Random + RAG-LLM Oracle | 1.0000 | 0.1212 | 33 | 4 | 33 | 15 |
| Margin + Rule (Hard Gate) | 1.0000 | 0.6250 | 8 | **5** | 8 | 5 |

**Key findings:**
- **DP = 1.0** across all configurations — no non-dialect false positives.
- **AL ceiling effect confirmed:** all soft-label strategies find exactly 4 power-dialect terms (|L_pd| = 4), regardless of query strategy or oracle type.
- **Hard gate achieves highest |L_pd| (5)** with only 8 terms and 5 queries — the precision-coverage tradeoff quantified.

**Soft-label vs Hard gate ablation:**

| | Hard Gate | Soft Label | Δ |
|---|:---:|:---:|:---:|
| Candidates extracted | 41 | 349 | +751% |
| After fragment filter | 9 | 58 | +544% |
| Final lexicon | 8 | 25 | +213% |
| DP | 1.0000 | 1.0000 | — |
| PDP | 0.6250 | 0.1600 | — |

**Cohen's Kappa (n=46):**
- κ_dialect = 1.0 (degenerate — all 46 terms annotated as dialect by both judges; no variance to assess κ meaningfully)
- κ_power = 0.8487, 95% CI [0.64, 1.0] — strong agreement

## Output Files

Each experiment produces under `experiments/results/`:

| File | Content |
|------|---------|
| `{exp}.json` | Summary metrics, config, round history |
| `{exp}_lexicon.json` | Full lexicon with oracle labels |
| `{exp}_candidates.json` | Candidate pool with Phase 2 classification |
| `{exp}_eval.json` | Phase 6 LLM evaluation with DP/PDP |
| `{exp}_dialect_lexicon.json` | Dialect terms (L_d) with full metadata |
| `{exp}_power_dialect_lexicon.json` | Power-dialect terms (L_pd) with full metadata |
| `calibration_template.csv` | Human annotation template |
| `calibration_results.json` | LLM evaluator results on calibration set |
| `calibration_kappa.json` | Cohen's κ with CI and confusion matrices |
| `comparison_table.md` | Markdown comparison of all configurations |

## Citation

```bibtex
@article{chang2025slang,
  title={Constructing a Regional Slang Lexicon for Electricity Demands via Soft-Label Bidirectional Anchoring and Active Learning},
  author={Chang, Rong and Shi, Lijing and Zang, Zhaoxiang and Mao, Cunli},
  journal={[Journal]},
  year={2025},
  note={Under review}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.