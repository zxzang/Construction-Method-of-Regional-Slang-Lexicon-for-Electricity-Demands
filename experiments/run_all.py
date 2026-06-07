# -*- coding: utf-8 -*-
"""
Main experiment runner.
Executes all 5 configurations from EXPERIMENT_PLAN.md and saves results.

Usage:
    python run_all.py                  # Run all 5 experiments + Phase 6 evaluation
    python run_all.py --exp KEY        # Run a single experiment + Phase 6
    python run_all.py --skip-eval      # Run experiments only, skip Phase 6 (dev)
    python run_all.py --seed N         # Run with a specific random seed
    python run_all.py --list           # List available experiment keys

    # Calibration (after human annotation is complete):
    python run_all.py --annotations results/calibration_annotated.csv
                                       # Compute Cohen's Kappa from existing results
                                       # (skips experiments, no API calls)
    python run_all.py --exp KEY --annotations results/calibration_annotated.csv
                                       # Run experiment + Kappa in one shot
"""
import sys
import os
import json
import time
import logging
import argparse
import random
import numpy as np
from typing import List

# Ensure experiments/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DATA_FILES, VOCAB_FILE, RESULTS_DIR, EXPERIMENTS,
                    RANDOM_SEED, AL_MAX_ROUNDS, TDCAL_TOP_K)


def _set_global_seed(seed: int):
    """Fix random seeds for reproducibility across all libraries.
    Note: PYTHONHASHSEED must be set BEFORE Python starts (e.g.
    ``export PYTHONHASHSEED=42``).  Setting it here has no effect."""
    random.seed(seed)
    np.random.seed(seed)


# ── Heavy imports (sentence-transformers / faiss / jieba / openai) are
# deferred into the functions that actually need them.  This keeps ``--list``
# and ``--annotations`` (Kappa-only) lightweight — they only need config and
# stdlib, not the full ML stack.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(
            os.path.dirname(__file__), 'experiment.log'),
            encoding='utf-8'),
    ]
)
logger = logging.getLogger('RunAll')


def run_single_experiment(exp_key: str, exp_config: dict,
                          corpus: list, reference_vocab: dict,
                          corpus_words: set = None,
                          extended_anchors: List[str] = None,
                          shared_cand_gen = None,
                          shared_mapper = None,
                          global_seed: int = 42,
                          diagnose_phase35: bool = False) -> str:
    """
    Run a single experiment configuration end-to-end.
    Returns path to the saved result JSON.

    Args:
        shared_cand_gen: Pre-built CandidateGenerator (optional).  If provided,
            its frequency model is reused instead of re-tokenizing the corpus.
        shared_mapper: Pre-loaded SemanticMapper (optional).  If provided,
            the SBERT model is reused instead of reloading from disk.
        diagnose_phase35: If True, run oracle labeling on auto-accepted
            candidates for post-hoc diagnosis (wastes API calls; off by default).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {exp_config['name']}")
    logger.info(f"Strategy: {exp_config['al_strategy']}, "
                f"Oracle: {exp_config['oracle_type']}, "
                f"NPMI: {exp_config['use_npmi']}")
    logger.info(f"{'='*60}")
    t_start = time.time()

    # --- Lazy heavy imports (deferred from module level) ---
    from candidate_generator import CandidateGenerator, _is_grammatical_fragment
    from semantic_mapper import SemanticMapper
    from oracle import RuleBasedOracle, RAGLLMOracle
    from active_learning import ActiveLearningLoop
    from evaluation import (compute_reference_overlap,
                            compute_cumulative_round_metrics, save_results)

    # --- Phase 1: Candidate Generation ---
    logger.info("[Phase 1] Generating candidates...")
    if shared_cand_gen is not None:
        cand_gen = shared_cand_gen
        logger.info("  (reusing shared frequency model)")
    else:
        cand_gen = CandidateGenerator(extended_anchors=extended_anchors)
        cand_gen.build_frequency_model(corpus)
    candidates = cand_gen.extract_candidates(
        use_npmi=exp_config['use_npmi'],
        hard_power_gate=exp_config.get('hard_power_gate', False))
    logger.info(f"  -> {len(candidates)} candidates extracted")

    # --- Phase 1.5: Fragment Filter (two-stage: remove before SBERT) ---
    n_before = len(candidates)
    candidates = [c for c in candidates
                  if not _is_grammatical_fragment(c['word'])]
    n_filtered = n_before - len(candidates)
    logger.info(f"[Phase 1.5] Fragment filter: removed {n_filtered} "
                f"grammatical fragments ({len(candidates)} remaining)")

    # --- Phase 2: Semantic Classification ---
    logger.info("[Phase 2] Semantic mapping with SBERT + FAISS "
                "(combined = α*sim + β*wordhood)...")
    if shared_mapper is not None:
        mapper = shared_mapper
        logger.info("  (reusing shared SBERT model)")
    else:
        mapper = SemanticMapper()
    accepted, uncertain, rejected = mapper.classify_candidates(candidates)

    # --- Phase 3: Oracle Setup ---
    logger.info("[Phase 3] Setting up oracle...")
    if exp_config['oracle_type'] == 'rag_llm':
        # ground_truth is NOT passed here — it is only needed for mock mode
        # (development/testing).  Real LLM oracle calls go to the API.
        oracle = RAGLLMOracle(sbert_model=mapper.model)
    else:
        oracle = RuleBasedOracle()

    # --- Phase 3.5: Auto-accepted candidates pass through (no oracle gate) ---
    # Candidates with SBERT+wordhood score >= SIM_HIGH bypass the AL loop
    # entirely (soft-label design: defer relevance decisions).  By default
    # they are NOT labeled by the oracle — that would waste API calls without
    # affecting the lexicon.  Use --diagnose-phase35 to enable oracle labeling
    # of auto-accepted terms for post-hoc diagnosis.
    if diagnose_phase35:
        logger.info(f"[Phase 3.5] Oracle-labeling {len(accepted)} auto-accepted "
                    f"candidates (diagnosis mode — all kept regardless)")
        judged_accepted = oracle.judge_batch(accepted)
        for c in judged_accepted:
            c['phase35_oracle_label'] = c.get('oracle_label', False)
            c['phase35_oracle_reason'] = c.get('oracle_reason', '')
        verified_count = sum(1 for c in judged_accepted
                             if c.get('oracle_label', False))
        logger.info(f"  -> {verified_count}/{len(accepted)} would be verified "
                    f"(all {len(accepted)} kept for lexicon)")
    else:
        logger.info(f"[Phase 3.5] {len(accepted)} auto-accepted candidates "
                    f"→ keeping all (oracle labeling skipped; use "
                    f"--diagnose-phase35 to enable)")
        judged_accepted = list(accepted)

    # --- Phase 4: Active Learning Loop ---
    logger.info("[Phase 4] Running active learning loop...")
    al_loop = ActiveLearningLoop(
        semantic_mapper=mapper,
        oracle=oracle,
        candidate_generator=cand_gen if exp_config['al_strategy'] == 'tdcal' else None,
        al_strategy=exp_config['al_strategy'],
        max_rounds=AL_MAX_ROUNDS,
        top_k=TDCAL_TOP_K,
        seed=global_seed,
    )
    lexicon, round_history = al_loop.run(judged_accepted, uncertain)

    elapsed = round(time.time() - t_start, 2)

    # --- Phase 5: Reference Vocabulary Overlap (auxiliary metric) ---
    logger.info("[Phase 5] Computing reference vocabulary overlap...")
    ref_overlap = compute_reference_overlap(lexicon, reference_vocab)
    logger.info(f"  Reference overlap: {ref_overlap['exact_matches']} exact, "
                f"{ref_overlap['token_overlap_matches']} token-overlap")

    # Compute per-round cumulative metrics
    round_history = compute_cumulative_round_metrics(
        judged_accepted, round_history, reference_vocab, corpus_words)

    # Summary metrics
    metrics = {
        'predicted_size': len(lexicon),
        'oracle_queries': oracle.query_count,
        'runtime_seconds': elapsed,
    }

    logger.info(f"  Lexicon:     {len(lexicon)} terms")
    logger.info(f"  Queries:     {oracle.query_count}")
    logger.info(f"  Ref overlap: {ref_overlap['exact_matches']} exact "
                f"/ {ref_overlap['token_overlap_matches']} token")
    logger.info(f"  Time:        {elapsed}s")

    # --- Save (summary + full lexicon + candidate pool) ---
    result_path = save_results(
        experiment_name=exp_key,
        metrics=metrics,
        lexicon=lexicon,
        round_history=round_history,
        config_info=exp_config,
        all_candidates=candidates,
        accepted=accepted,
        uncertain=uncertain,
        rejected=rejected,
        pre_accepted=judged_accepted,
        reference_overlap=ref_overlap,
    )
    return result_path


def run_posthoc_evaluation(result_files: List[str]):
    """Phase 6: Post-hoc LLM evaluation after all experiments complete.

    Step 6a: Merge lexicons → sample calibration set (~100 terms).
    Step 6b: Generate ONE human annotation template (not per-experiment).
    Step 6c: LLM evaluator runs on the same calibration set.
    Step 6d: Batch LLM evaluation on every experiment's lexicon.
    Step 6e: Generate final comparison table with DP/PDP/FER/AvgConf.

    Human annotation happens OFFLINE after this function.  Once annotated,
    the user runs ``compute_llm_human_agreement()`` to compute Cohen's Kappa
    and validate the evaluator's reliability.
    """
    from oracle import RAGLLMOracle, EvaluationOracle
    from evaluation import (sample_for_human_annotation,
                            generate_human_annotation_template)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 6: POST-HOC LLM EVALUATION")
    logger.info("=" * 60)

    # ── Load all lexicons from saved files ──
    all_lexicons = {}   # {exp_key: [lexicon entries]}
    for fpath in result_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        exp_key = data['experiment']
        lex_path = os.path.join(RESULTS_DIR, f"{exp_key}_lexicon.json")
        if os.path.exists(lex_path):
            with open(lex_path, 'r', encoding='utf-8') as f:
                all_lexicons[exp_key] = json.load(f)
            logger.info(f"  Loaded {len(all_lexicons[exp_key])} terms "
                        f"from {exp_key}_lexicon.json")
        else:
            logger.warning(f"  Lexicon file not found: {lex_path}")
            all_lexicons[exp_key] = []

    # ── Phase 6a: Merge and sample calibration set ──
    logger.info("[Phase 6a] Building calibration set from merged lexicons...")
    merged_pool = {}
    for exp_key, lexicon in all_lexicons.items():
        for c in lexicon:
            w = c['word']
            if w not in merged_pool:
                merged_pool[w] = c
            else:
                # Keep the entry with higher combined_score
                existing_score = merged_pool[w].get('combined_score',
                                                    merged_pool[w].get('similarity', 0))
                new_score = c.get('combined_score', c.get('similarity', 0))
                if new_score > existing_score:
                    merged_pool[w] = c

    merged_list = list(merged_pool.values())
    logger.info(f"  Merged pool: {len(merged_list)} unique terms "
                f"(from {sum(len(l) for l in all_lexicons.values())} total)")

    cal_samples = sample_for_human_annotation(merged_list)
    logger.info(f"  Calibration set: {len(cal_samples)} terms "
                f"for human annotation")

    # ── Phase 6b: Generate human annotation template ──
    logger.info("[Phase 6b] Generating human annotation template...")
    cal_template_path = os.path.join(RESULTS_DIR, 'calibration_template.csv')
    generate_human_annotation_template(cal_samples, cal_template_path)

    # ── Build evaluator ──
    logger.info("[Phase 6c] Initializing LLM evaluator (deepseek-v4-pro)...")
    eval_kb = RAGLLMOracle._build_default_kb()
    eval_oracle = EvaluationOracle(kb_entries=eval_kb)

    # ── Phase 6c: Evaluate ALL unique terms ONCE ──
    # Instead of evaluating each experiment's lexicon independently (which
    # would duplicate work for terms appearing in multiple lexicons), we
    # evaluate the merged pool once and then look up per-experiment metrics.
    # The calibration set is a subset of merged_list, so its results are
    # included here automatically.
    logger.info("[Phase 6c] Evaluating ALL unique terms ONCE (merged pool "
                "= calibration set + all lexicons deduplicated)...")
    logger.info(f"  Total unique terms to evaluate: {len(merged_list)}")
    logger.info(f"  (Of which {len(cal_samples)} are in the calibration set)")

    # Evaluate all unique terms — log progress every 20 terms
    all_eval_results = []
    total = len(merged_list)
    for i, c in enumerate(merged_list):
        r = eval_oracle._evaluate_one(c)
        all_eval_results.append(r)
        if (i + 1) % 20 == 0 or (i + 1) == total:
            logger.info(f"  ... {i + 1}/{total} terms evaluated")

    eval_by_word = {r['word']: r for r in all_eval_results}

    logger.info(f"  Evaluation complete: {len(all_eval_results)} terms assessed "
                f"({eval_oracle.query_count} LLM calls)")

    # ── Extract calibration subset ──
    cal_eval_data = [eval_by_word[c['word']] for c in cal_samples
                     if c['word'] in eval_by_word]

    # Compute calibration quality metrics
    n = len(cal_eval_data)
    n_dialect = sum(1 for r in cal_eval_data if r.get('eval_has_dialect', False))
    n_power = sum(1 for r in cal_eval_data if r.get('eval_power_relevant', False))
    n_power_dialect = sum(1 for r in cal_eval_data
                          if r.get('eval_power_relevant', False)
                          and r.get('eval_has_dialect', False))
    cal_detail = {'n_total': n, 'n_dialect': n_dialect,
                  'n_power': n_power, 'n_power_dialect': n_power_dialect}

    # Save calibration results
    cal_path = os.path.join(RESULTS_DIR, 'calibration_results.json')
    with open(cal_path, 'w', encoding='utf-8') as f:
        json.dump({
            'n_samples': len(cal_samples),
            'detail': cal_detail,
            'llm_evaluations': cal_eval_data,
            'instruction': (
                'Human annotation template: calibration_template.csv\n'
                '1. Fill in the 2 empty columns (has_dialect_features, '
                'power_domain_relevance)\n'
                '2. Save as calibration_annotated.csv\n'
                '3. Run: compute_llm_human_agreement(llm_results, human_annotations)'
            ),
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"  Calibration results saved to {cal_path}")

    # ── Phase 6d: Per-experiment metrics via lookup (no extra API calls) ──
    logger.info("[Phase 6d] Computing per-experiment metrics from cached evaluations...")
    eval_summaries = {}
    for exp_key, lexicon in all_lexicons.items():
        if not lexicon:
            eval_summaries[exp_key] = None
            continue

        # Look up cached evaluations for this experiment's terms
        exp_evals = [eval_by_word[c['word']] for c in lexicon
                     if c['word'] in eval_by_word]
        n = len(exp_evals)
        if n == 0:
            eval_summaries[exp_key] = None
            continue

        n_dialect = sum(1 for r in exp_evals if r.get('eval_has_dialect', False))
        n_power = sum(1 for r in exp_evals if r.get('eval_power_relevant', False))
        n_power_dialect = sum(1 for r in exp_evals
                              if r.get('eval_power_relevant', False)
                              and r.get('eval_has_dialect', False))
        avg_conf = round(np.mean([r.get('eval_confidence', 0.5)
                                  for r in exp_evals]), 4) if exp_evals else 0.0

        cat_dist = {}
        for r in exp_evals:
            cat = r.get('eval_category', 'OTHER')
            cat_dist[cat] = cat_dist.get(cat, 0) + 1

        eval_summaries[exp_key] = {
            'dialect_precision': round(n_dialect / n, 4),
            'power_dialect_precision': round(n_power_dialect / n, 4),
            'avg_confidence': avg_conf,
            'category_distribution': cat_dist,
            'detail': {'n_total': n, 'n_dialect': n_dialect,
                       'n_power': n_power, 'n_power_dialect': n_power_dialect},
            'evaluator_queries': 0,
            # Dual-lexicon output (direction B):
            # L_d = dialect lexicon (eval_has_dialect=True) → DP computed on this
            # L_pd = power-dialect lexicon (dialect AND power) → measured by |L_pd|
            'dialect_lexicon_size': n_dialect,
            'power_dialect_lexicon_size': n_power_dialect,
        }

        # Save per-experiment eval file
        eval_path = os.path.join(RESULTS_DIR, f"{exp_key}_eval.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment': exp_key,
                'summary': eval_summaries[exp_key],
                'evaluations': exp_evals,
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"  {exp_key}: DP={eval_summaries[exp_key]['dialect_precision']:.4f} "
                    f"PDP={eval_summaries[exp_key]['power_dialect_precision']:.4f} "
                    f"|L_d|={n_dialect} |L_pd|={n_power_dialect} "
                    f"({n} total) -> {exp_key}_eval.json")

        # ── Export dual lexicons (direction B) ──
        # L_d:  dialect terms (eval_has_dialect=True)
        # L_pd: power-dialect terms (dialect AND power_relevant)
        # Merge original candidate metadata (pmi/similarity/freq/...) with
        # evaluator results for auditability.
        orig_by_word = {c['word']: c for c in lexicon}
        dialect_lexicon = [{**orig_by_word.get(r['word'], {}), **r}
                           for r in exp_evals
                           if r.get('eval_has_dialect', False)]
        power_dialect_lexicon = [{**orig_by_word.get(r['word'], {}), **r}
                                 for r in exp_evals
                                 if r.get('eval_has_dialect', False)
                                 and r.get('eval_power_relevant', False)]
        for suffix, lex in [('_dialect_lexicon', dialect_lexicon),
                            ('_power_dialect_lexicon', power_dialect_lexicon)]:
            lex_path = os.path.join(RESULTS_DIR, f"{exp_key}{suffix}.json")
            with open(lex_path, 'w', encoding='utf-8') as f:
                json.dump(lex, f, ensure_ascii=False, indent=2)
            logger.info(f"    -> {exp_key}{suffix}.json ({len(lex)} terms)")

    # ── Save combined evaluation summary ──
    eval_summary_path = os.path.join(RESULTS_DIR, 'evaluation_summary.json')
    with open(eval_summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'calibration': {
                'n_samples': len(cal_samples),
                'template': 'calibration_template.csv',
                'results': 'calibration_results.json',
            },
            'experiments': eval_summaries,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"  Evaluation summary saved to {eval_summary_path}")

    # ── Phase 6e: Generate final comparison table ──
    logger.info("[Phase 6e] Generating final comparison table...")
    table = _build_comparison_table(result_files, eval_summaries)
    print(table)

    table_path = os.path.join(RESULTS_DIR, 'comparison_table.md')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write("# Experiment Comparison (Phase 6 LLM Evaluation)\n\n")
        f.write(table)
        f.write("\n")
    logger.info(f"  Comparison table saved to {table_path}")

    # Append to experiment log
    log_path = os.path.join(os.path.dirname(__file__), 'experiment.log')
    with open(log_path, 'a', encoding='utf-8') as lf:
        lf.write("\n# Phase 6 Evaluation\n\n")
        lf.write(table)
        lf.write("\n")
    logger.info("  Comparison table appended to experiment.log")

    logger.info("\nPhase 6 complete.")
    logger.info("Next step: have a human expert annotate 'calibration_template.csv',")
    logger.info("  then run compute_llm_human_agreement() to validate the evaluator.")


def _build_comparison_table(result_files: List[str],
                            eval_summaries: dict) -> str:
    """Build Markdown comparison table from post-hoc evaluation results."""
    rows = []
    for fpath in result_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        exp_key = data['experiment']
        name = data['config'].get('name', exp_key)
        m = data.get('metrics', {})
        ref = data.get('reference_overlap', {})
        es = eval_summaries.get(exp_key) or {}

        rows.append({
            'name': name,
            'DP': es.get('dialect_precision', '—'),
            'PDP': es.get('power_dialect_precision', '—'),
            'avg_conf': es.get('avg_confidence', '—'),
            'ref_exact': ref.get('exact_matches', '—'),
            'ref_token': ref.get('token_overlap_matches', '—'),
            'size': m.get('predicted_size', 0),
            'queries': m.get('oracle_queries', 'N/A'),
            'size_d': es.get('dialect_lexicon_size', 0),
            'size_pd': es.get('power_dialect_lexicon_size', 0),
        })

    lines = [
        "| Model | DP | PDP | L_d | L_pd | AvgConf | Ref(Exact) | Ref(Token) | Lexicon | Queries |",
        "|-------|----|-----|-------|--------|---------|------------|------------|---------|---------|",
    ]

    def _fmt(v):
        if isinstance(v, str):
            return v
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for r in sorted(rows, key=lambda x: (x['size_pd'] if isinstance(x['size_pd'], int) else 0), reverse=True):
        lines.append(
            f"| {r['name']} | {_fmt(r['DP'])} | {_fmt(r['PDP'])} | "
            f"{r['size_d']} | {r['size_pd']} | "
            f"{_fmt(r['avg_conf'])} | "
            f"{_fmt(r['ref_exact'])} | {_fmt(r['ref_token'])} | "
            f"{r['size']} | {r['queries']} |")

    return "\n".join(lines)


def _run_kappa_calibration(annotations_path: str):
    """Compute Cohen's Kappa from existing LLM evaluations and human annotations.

    Reads calibration_results.json and the annotated CSV, computes κ for
    2 dimensions (has_dialect_features, power_domain_relevance), reports PASS/FAIL per dimension, and saves the report to calibration_kappa.json.
    Does NOT run any experiments or make any API calls.
    """
    from evaluation import load_human_annotations, compute_llm_human_agreement

    cal_results_path = os.path.join(RESULTS_DIR, 'calibration_results.json')
    if not os.path.exists(cal_results_path):
        raise FileNotFoundError(
            f"calibration_results.json not found at {cal_results_path}. "
            f"Run experiments + Phase 6 first to generate it.")

    logger.info("\n" + "=" * 60)
    logger.info("COHEN'S KAPPA CALIBRATION")
    logger.info("=" * 60)

    try:
        with open(cal_results_path, 'r', encoding='utf-8') as f:
            cal_data = json.load(f)
        llm_evaluations = cal_data.get('llm_evaluations', [])
        human_annotations = load_human_annotations(annotations_path)

        kappa = compute_llm_human_agreement(llm_evaluations, human_annotations)

        logger.info(f"  Compared: {kappa['n_compared']} terms")
        logger.info(f"  Kappa (has_dialect_features): {kappa['kappa_dialect']}")
        logger.info(f"  Kappa (power_domain_relevance):{kappa['kappa_power']}")

        threshold = 0.6
        for dim, key in [("has_dialect_features", "kappa_dialect"),
                          ("power_domain_relevance", "kappa_power")]:
            val = kappa.get(key)
            if val is None:
                status = "N/A (no overlap)"
            elif val >= threshold:
                status = "PASS"
            else:
                status = f"FAIL (κ={val:.3f} < {threshold})"
            logger.info(f"    {dim}: {status}")

        kappa_path = os.path.join(RESULTS_DIR, 'calibration_kappa.json')
        with open(kappa_path, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': threshold,
                'n_compared': kappa['n_compared'],
                'kappa_dialect': kappa['kappa_dialect'],
                'kappa_dialect_degenerate': kappa.get('kappa_dialect_degenerate'),
                'kappa_dialect_ci95': kappa.get('kappa_dialect_ci95'),
                'kappa_power': kappa['kappa_power'],
                'kappa_power_degenerate': kappa.get('kappa_power_degenerate'),
                'kappa_power_ci95': kappa.get('kappa_power_ci95'),
                'matrix_dialect': kappa.get('kappa_dialect_matrix'),
                'matrix_power': kappa.get('kappa_power_matrix'),
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"  Kappa report saved to {kappa_path}")

    except Exception as e:
        logger.error(f"Kappa computation FAILED: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Regional Slang Lexicon Experiment Runner')
    parser.add_argument('--exp', type=str, default=None,
                        help='Run specific experiment key (e.g., proposed_tdcal_llm)')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip Phase 6 (LLM evaluation). '
                             'Only run experiments + pre-calibration table. '
                             'Useful for quick iteration during development.')
    parser.add_argument('--diagnose-phase35', action='store_true',
                        help='Run oracle labeling on Phase-3.5 auto-accepted '
                             'candidates (costs extra API calls; for post-hoc '
                             'diagnosis only — does not affect the lexicon).')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to human-annotated CSV (calibration_annotated.csv). '
                             'If provided, computes Cohen\'s Kappa between LLM '
                             'evaluator and human judgments after Phase 6.')
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for key, cfg in EXPERIMENTS.items():
            print(f"  {key:30s} -> {cfg['name']}")
        return

    # ── Fast path: --annotations only, skip experiments ──
    # When the user only wants Kappa on existing Phase 6 results, don't
    # re-run experiments.  If calibration_results.json doesn't exist yet,
    # warn and exit.
    if args.annotations and not args.exp:
        cal_results_path = os.path.join(RESULTS_DIR, 'calibration_results.json')
        if os.path.exists(cal_results_path):
            _run_kappa_calibration(args.annotations)
            return
        else:
            logger.error(
                "calibration_results.json not found at %s. "
                "Run experiments + Phase 6 first, then retry with --annotations.",
                cal_results_path)
            sys.exit(1)

    # Apply random seed for reproducibility
    _set_global_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    if 'PYTHONHASHSEED' not in os.environ:
        logger.warning(
            "PYTHONHASHSEED is not set — dict/set iteration order may vary "
            "across runs.  Set it before starting Python:\n"
            "  export PYTHONHASHSEED=42 && python experiments/run_all.py\n"
            "  (or add PYTHONHASHSEED=42 to your .env file)")
    else:
        logger.info(f"PYTHONHASHSEED={os.environ['PYTHONHASHSEED']} "
                    "(dict/set order locked)")

    # --- Lazy heavy imports (deferred from module level) ---
    from data_loader import (DataPreprocessor, load_reference_vocabulary,
                             get_corpus_dialect_anchors)
    from candidate_generator import CandidateGenerator
    from semantic_mapper import SemanticMapper
    from evaluation import generate_comparison_table

    # ===== Shared Data Loading (once) =====
    logger.info("Loading and preprocessing corpus...")
    corpus = DataPreprocessor.load_and_preprocess(DATA_FILES)
    logger.info(f"Corpus: {len(corpus)} clean messages")

    logger.info("Loading reference vocabulary...")
    reference_vocab = load_reference_vocabulary(VOCAB_FILE)
    logger.info(f"Reference vocabulary: {len(reference_vocab)} dialect terms "
                "(general Yuxi dialect dictionary — NOT ground truth)")

    # Inject reference dialect words into jieba so they are tokenized as whole words
    # (e.g., "扎实" stays as one token instead of being split into "扎"+"实").
    # This is tokenization calibration, not label leakage — we only tell jieba
    # that these character sequences are words, not which ones are correct answers.
    import jieba
    # Force-load jieba's default dictionary (lazy init: FREQ is empty until
    # the first cut/add_word call triggers initialize()).  We must snapshot
    # AFTER initialization so the word list is complete.
    jieba.initialize()
    # Snapshot jieba's default dictionary before adding reference words.
    # Used by get_corpus_dialect_anchors() to exclude standard Chinese words
    # that happen to contain dialect-looking characters (e.g. 冬瓜 contains 冬).
    # gen_pfdict() writes real words (freq > 0) AND prefix fragments (freq = 0)
    # for Trie construction.  We only want the genuine dictionary entries.
    _JIEBA_DEFAULT_WORDS = {w for w, f in jieba.dt.FREQ.items() if f > 0}
    _JIEBA_DICT_SIZE = len(_JIEBA_DEFAULT_WORDS)
    if _JIEBA_DICT_SIZE < 200_000:
        # jieba's default dict.txt should contain ~349k words.  A count this
        # low indicates the dictionary file was not loaded (missing, corrupt,
        # or wrong jieba installation).  Without a proper default word set,
        # the standard-Chinese exclusion in get_corpus_dialect_anchors()
        # would be silently ineffective — dialect anchors could include
        # common standard words.
        logger.error(
            "jieba default dictionary has only %d entries (expected ~349k). "
            "The standard-Chinese exclusion filter in "
            "get_corpus_dialect_anchors() will be DISABLED to avoid "
            "incorrectly filtering dialect terms on an unreliable word list. "
            "Check your jieba installation (dict.txt may be missing).",
            _JIEBA_DICT_SIZE)
        _JIEBA_DEFAULT_WORDS = None   # disable the filter
    else:
        logger.info(
            "jieba default dictionary snapshot: %d words (filter enabled "
            "for get_corpus_dialect_anchors)", _JIEBA_DICT_SIZE)
    gt_injected = 0
    for word in reference_vocab:
        if len(word) >= 2:
            jieba.add_word(word)
            gt_injected += 1
    logger.info(f"Injected {gt_injected} reference words into jieba dictionary")

    # Build set of all words in corpus for reference filtering (after jieba calibration)
    corpus_words = set()
    for msg in corpus:
        corpus_words.update(jieba.lcut(msg))
    ref_found = sum(1 for w in reference_vocab if w in corpus_words)
    logger.info(f"Corpus vocabulary: {len(corpus_words)} unique words, "
                f"{ref_found}/{len(reference_vocab)} reference terms found in corpus")

    # Extract dialect content-word anchors from reference vocabulary
    # (Improvement 2: enriches bigram extraction beyond function words)
    extended_anchors = get_corpus_dialect_anchors(corpus, reference_vocab,
                                                  jieba_default_words=_JIEBA_DEFAULT_WORDS)
    logger.info(f"Extended anchors: {len(extended_anchors)} dialect content words "
                f"from reference vocabulary")

    # ===== Shared Pipeline Components (created once, reused across exps) =====
    logger.info("Initializing shared pipeline components...")
    shared_cand_gen = CandidateGenerator(extended_anchors=extended_anchors)
    shared_cand_gen.build_frequency_model(corpus)
    shared_mapper = SemanticMapper()
    logger.info("Shared components ready (frequency model + SBERT loaded once).")

    # ===== Run Experiments =====
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_files = []
    failed_experiments = []

    if args.exp:
        # Run single experiment
        if args.exp not in EXPERIMENTS:
            matching = [k for k in EXPERIMENTS if args.exp in k]
            if len(matching) == 1:
                args.exp = matching[0]
            else:
                logger.error(f"Unknown experiment: {args.exp}")
                logger.info(f"Available: {list(EXPERIMENTS.keys())}")
                return
        path = run_single_experiment(args.exp, EXPERIMENTS[args.exp],
                                     corpus, reference_vocab, corpus_words,
                                     extended_anchors,
                                     shared_cand_gen, shared_mapper,
                                     global_seed=args.seed,
                                     diagnose_phase35=args.diagnose_phase35)
        result_files.append(path)
    else:
        # Run all experiments
        for key, config in EXPERIMENTS.items():
            try:
                path = run_single_experiment(key, config, corpus, reference_vocab,
                                             corpus_words, extended_anchors,
                                             shared_cand_gen, shared_mapper,
                                             global_seed=args.seed,
                                             diagnose_phase35=args.diagnose_phase35)
                result_files.append(path)
            except Exception as e:
                logger.error(f"Experiment {key} FAILED: {e}", exc_info=True)
                failed_experiments.append(key)

    # ===== Pre-Calibration Comparison Table =====
    # Quick overview for development iteration — basic metrics only.
    # Final paper metrics (DP/PDP/FER/AvgConf) come from Phase 6 below.
    if result_files:
        logger.info("\n" + "=" * 60)
        logger.info("PRE-CALIBRATION COMPARISON")
        logger.info("=" * 60)
        pre_table = generate_comparison_table(result_files)
        print(pre_table)

    # ===== Phase 6: Post-hoc LLM Evaluation =====
    # Generates calibration set, human template, batch evaluation,
    # and final comparison table with DP/PDP/FER/AvgConf.
    phase6_failed = False
    if args.skip_eval:
        logger.info("\nPhase 6 skipped (--skip-eval).")
    elif result_files:
        try:
            run_posthoc_evaluation(result_files)
        except Exception as e:
            logger.error(f"Post-hoc evaluation FAILED: {e}", exc_info=True)
            phase6_failed = True

    # ===== Cohen's Kappa Calibration (optional) =====
    if args.annotations:
        if phase6_failed:
            logger.error(
                "Phase 6 failed — refusing to compute Kappa on potentially "
                "stale calibration_results.json.  Re-run experiments+Phase 6 "
                "first, then retry --annotations.")
        else:
            _run_kappa_calibration(args.annotations)
    elif result_files and not args.skip_eval:
        logger.info("\nHuman annotation calibration not run.")
        logger.info("To calibrate: annotate 'calibration_template.csv', save as")
        logger.info("  'calibration_annotated.csv', then run:")
        logger.info("  python run_all.py --annotations results/calibration_annotated.csv")

    if failed_experiments:
        logger.error(
            f"\n{len(failed_experiments)} experiment(s) FAILED: "
            f"{', '.join(failed_experiments)}")
    parts = [f"{len(result_files)} succeeded"]
    if failed_experiments:
        parts.append(f"{len(failed_experiments)} failed")
    if phase6_failed:
        parts.append("Phase 6 FAILED")
    logger.info(f"\nExperiments finished ({', '.join(parts)}).")
    if failed_experiments or phase6_failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
