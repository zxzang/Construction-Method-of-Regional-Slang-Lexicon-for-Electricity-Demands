# -*- coding: utf-8 -*-
"""
Evaluation module: LLM-based quality assessment, reference vocabulary overlap,
human annotation calibration, and comparison table generation.
"""
import json
import os
import csv
import logging
import random
import numpy as np
from typing import List, Dict, Set, Tuple

from config import (RESULTS_DIR, EVAL_BOUNDARY_SAMPLE, EVAL_RANDOM_SAMPLE,
                    EVAL_BOUNDARY_MARGIN)

logger = logging.getLogger(__name__)


def evaluate_lexicon_quality(lexicon: List[dict], eval_oracle) -> dict:
    """Run independent LLM evaluation on all lexicon terms.

    The evaluator model (deepseek-v4-pro) is separate from the selection
    oracle (deepseek-v4-flash), so metrics are not self-fulfilling.

    Returns dict with computed quality metrics and per-term evaluation results.
    """
    if not lexicon:
        logger.warning("Lexicon is empty — no evaluation to run")
        return {
            'dialect_precision': 0.0,
            'power_dialect_precision': 0.0,
            'avg_confidence': 0.0,
            'category_distribution': {},
            'evaluator_queries': 0,
            'eval_results': [],
        }

    logger.info(f"Evaluating {len(lexicon)} lexicon terms with independent LLM...")
    eval_results = eval_oracle.evaluate(lexicon)

    n = len(eval_results)
    n_dialect = sum(1 for r in eval_results if r.get('eval_has_dialect', False))
    n_power = sum(1 for r in eval_results if r.get('eval_power_relevant', False))
    n_power_dialect = sum(1 for r in eval_results
                          if r.get('eval_power_relevant', False)
                          and r.get('eval_has_dialect', False))
    avg_conf = round(np.mean([r.get('eval_confidence', 0.5)
                              for r in eval_results]), 4)

    category_dist = {}
    for r in eval_results:
        cat = r.get('eval_category', 'OTHER')
        category_dist[cat] = category_dist.get(cat, 0) + 1

    metrics = {
        'dialect_precision': round(n_dialect / n, 4) if n else 0.0,
        'power_dialect_precision': round(n_power_dialect / n, 4) if n else 0.0,
        'avg_confidence': avg_conf,
        'category_distribution': category_dist,
        'evaluator_queries': eval_oracle.query_count,
        'detail': {
            'n_total': n,
            'n_dialect': n_dialect,
            'n_power': n_power,
            'n_power_dialect': n_power_dialect,
        },
    }

    logger.info(
        f"Quality metrics: DP={metrics['dialect_precision']:.4f} "
        f"PDP={metrics['power_dialect_precision']:.4f} "
        f"AvgConf={avg_conf:.4f}"
    )
    logger.info(f"Category distribution: {category_dist}")

    return {**metrics, 'eval_results': eval_results}


def compute_reference_overlap(lexicon: List[dict],
                              reference_vocab: dict) -> dict:
    """Compute overlap between predicted lexicon and reference vocabulary.

    This is an AUXILIARY metric — the reference vocabulary (词汇汇编-v2.md)
    is a general Yuxi dialect dictionary used for understanding dialect
    patterns, NOT a ground-truth answer key. Overlap is informative but
    does not directly measure lexicon quality.

    Returns exact-match and token-overlap counts and term lists.
    """
    import jieba

    if not lexicon or not reference_vocab:
        return {'exact_matches': 0, 'exact_terms': [],
                'token_overlap_matches': 0, 'token_overlap_terms': []}

    pred_words = {c['word'] for c in lexicon}
    ref_words = set(reference_vocab.keys())

    # Exact match
    exact = sorted(pred_words & ref_words)

    # Token overlap (jieba)
    ref_toks = {rw: set(jieba.lcut(rw)) for rw in ref_words}
    tok_overlap = []
    for pw in pred_words:
        ptoks = set(jieba.lcut(pw))
        for rw, rtoks in ref_toks.items():
            if ptoks & rtoks:
                tok_overlap.append(pw)
                break

    logger.info(f"Reference overlap: {len(exact)} exact, "
                f"{len(tok_overlap)} token-overlap "
                f"(out of {len(ref_words)} ref, {len(pred_words)} pred)")

    return {
        'exact_matches': len(exact),
        'exact_terms': exact,
        'token_overlap_matches': len(tok_overlap),
        'token_overlap_terms': tok_overlap,
        'reference_size': len(ref_words),
    }


def sample_for_human_annotation(pool: List[dict],
                                n_boundary: int = EVAL_BOUNDARY_SAMPLE,
                                n_random: int = EVAL_RANDOM_SAMPLE,
                                margin: float = EVAL_BOUNDARY_MARGIN,
                                seed: int = 42) -> List[dict]:
    """Sample terms for human annotation: boundary + random.

    Boundary terms are those closest to the decision boundary
    (|combined_score - SIM_BOUNDARY| < margin).  Random terms are
    uniformly sampled from the remaining pool.
    """
    from config import SIM_BOUNDARY

    rng = random.Random(seed)
    boundary, others = [], []
    for c in pool:
        score = c.get('combined_score', c.get('similarity', 0.5))
        if abs(score - SIM_BOUNDARY) < margin:
            boundary.append(c)
        else:
            others.append(c)

    # Sort boundary by distance to boundary (closest first)
    boundary.sort(key=lambda x: abs(
        x.get('combined_score', x.get('similarity', 0.5)) - SIM_BOUNDARY))

    n_bound = min(n_boundary, len(boundary))
    n_rand = min(n_random, len(others))

    sampled_boundary = boundary[:n_bound] if n_bound > 0 else []
    sampled_random = rng.sample(others, n_rand) if n_rand > 0 else []

    logger.info(f"Human annotation sample: {len(sampled_boundary)} boundary "
                f"+ {len(sampled_random)} random = "
                f"{len(sampled_boundary) + len(sampled_random)} total")

    return sampled_boundary + sampled_random


def generate_human_annotation_template(samples: List[dict],
                                       output_path: str) -> str:
    """Generate a CSV template for human annotation.

    Columns: word, has_dialect_features, power_domain_relevance, notes.
    Annotation columns are left blank for the human expert to fill in.
    is_fixed_expression removed after κ=0.29 calibration.
    source_hint removed (M6: blinding — oracle metadata must not bias annotators).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for c in samples:
        rows.append({
            'word': c.get('word', ''),
            'has_dialect_features': '',
            'power_domain_relevance': '',
            'notes': '',
        })

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'word', 'has_dialect_features', 'power_domain_relevance', 'notes'])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Human annotation template saved to {output_path} "
                f"({len(rows)} terms)")
    return output_path


def load_human_annotations(file_path: str) -> List[dict]:
    """Load completed human annotations from CSV.

    Expects columns matching generate_human_annotation_template output.
    Boolean columns parsed as: 'true'/'True'/'1'/'yes' → True.
    """
    annotations = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            def _parse_bool(val):
                v = str(val).strip().lower()
                if v in ('true', '1', 'yes'):
                    return True
                if v in ('false', '0', 'no'):
                    return False
                if v == '':
                    return None  # blank → treat as missing, not False
                logger.warning(
                    f"Unrecognised annotation value for '{row.get('word','?')}': "
                    f"{repr(val)} — treating as None")
                return None
            h_dialect = _parse_bool(row['has_dialect_features'])
            h_power = _parse_bool(row['power_domain_relevance'])
            if h_dialect is None or h_power is None:
                logger.warning(
                    f"Skipping '{row.get('word','?')}': incomplete annotation "
                    f"(dialect={h_dialect}, power={h_power})")
                continue
            annotations.append({
                'word': row['word'].strip(),
                'has_dialect_features': h_dialect,
                'power_domain_relevance': h_power,
                'notes': row.get('notes', '').strip(),
            })
    logger.info(f"Loaded {len(annotations)} human annotations from {file_path}")
    return annotations


def compute_llm_human_agreement(llm_results: List[dict],
                                human_annotations: List[dict]) -> dict:
    """Compute Cohen's Kappa between LLM evaluator and human annotations.

    Computes Kappa for 2 dimensions (has_dialect_features,
    power_domain_relevance) independently.
    Returns kappa values, 95% CI, degenerate flags, and per-dimension
    confusion matrices.
    """
    human_by_word = {a['word']: a for a in human_annotations}
    llm_by_word = {r['word']: r for r in llm_results}

    common = set(human_by_word.keys()) & set(llm_by_word.keys())
    if not common:
        logger.warning("No overlap between LLM and human annotations")
        return {'kappa_dialect': None, 'kappa_power': None,
                'n_compared': 0}

    dims = [
        ('kappa_dialect', 'has_dialect_features', 'eval_has_dialect'),
        ('kappa_power', 'power_domain_relevance', 'eval_power_relevant'),
    ]

    result = {'n_compared': len(common)}
    for kappa_key, human_key, llm_key in dims:
        a_true = b_true = a_false = b_false = 0
        for word in common:
            h = human_by_word[word].get(human_key, False)
            l = llm_by_word[word].get(llm_key, False)
            if h and l:
                a_true += 1
            elif h and not l:
                a_false += 1
            elif not h and l:
                b_true += 1
            else:
                b_false += 1

        n = len(common)
        p_o = (a_true + b_false) / n  # observed agreement
        # Expected agreement
        p_yes = ((a_true + a_false) / n) * ((a_true + b_true) / n)
        p_no = ((b_true + b_false) / n) * ((a_false + b_false) / n)
        p_e = p_yes + p_no

        degenerate = (p_e >= 0.999)  # near-degenerate prevalence
        if degenerate:
            # When all (or nearly all) samples share the same label, κ is
            # not informative even if observed agreement is perfect.
            kappa = 1.0 if p_o >= 0.999 else (p_o - p_e) / (1.0 - p_e)
            logger.warning(
                f"κ_{kappa_key.split('_')[1]}: degenerate prevalence "
                f"(p_e={p_e:.4f}, n_true={a_true + a_false}, "
                f"n_false={b_true + b_false}).  κ={kappa:.4f} should be "
                f"interpreted with caution.")
        else:
            kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 0 else 0.0

        result[kappa_key] = round(kappa, 4)
        result[f'{kappa_key}_degenerate'] = degenerate
        result[f'{kappa_key}_matrix'] = {
            'llm_true_human_true': a_true,
            'llm_false_human_true': a_false,
            'llm_true_human_false': b_true,
            'llm_false_human_false': b_false,
        }

        # ── 95% Confidence Interval (asymptotic SE) ──
        # SE(κ) ≈ sqrt(p_o * (1-p_o) / (n * (1-p_e)^2))
        # CI = κ ± 1.96 * SE, clamped to [-1, 1]
        if (1.0 - p_e) > 0 and n > 0:
            se = (p_o * (1.0 - p_o) / (n * (1.0 - p_e) ** 2)) ** 0.5
            ci_low = max(-1.0, round(kappa - 1.96 * se, 4))
            ci_high = min(1.0, round(kappa + 1.96 * se, 4))
        else:
            ci_low, ci_high = None, None
        result[f'{kappa_key}_ci95'] = [ci_low, ci_high]

    logger.info(
        f"LLM-Human agreement (n={len(common)}): "
        f"κ_dialect={result['kappa_dialect']} "
        f"CI95={result.get('kappa_dialect_ci95')}, "
        f"κ_power={result['kappa_power']} "
        f"CI95={result.get('kappa_power_ci95')}"
    )
    return result


def compute_cumulative_round_metrics(initial_accepted: List[dict],
                                     round_history: List[dict],
                                     reference_vocab: dict,
                                     corpus_words: set = None) -> List[dict]:
    """Compute cumulative reference overlap at each AL round.

    Tracks lexicon growth and overlap with the reference vocabulary
    (auxiliary metric only — reference is NOT a ground-truth answer key).
    """
    lexicon_words = set()
    for c in initial_accepted:
        lexicon_words.add(c['word'] if isinstance(c, dict) else str(c))

    for entry in round_history:
        for w in entry.get('verified_words', []):
            lexicon_words.add(w)

        overlap = compute_reference_overlap(
            [{'word': w} for w in lexicon_words], reference_vocab)
        entry['cumulative_lexicon_size'] = len(lexicon_words)
        entry['cumulative_ref_exact'] = overlap['exact_matches']
        entry['cumulative_ref_token'] = overlap['token_overlap_matches']
    return round_history


def save_results(experiment_name: str, metrics: dict, lexicon: List[dict],
                 round_history: List[dict], config_info: dict,
                 all_candidates: List[dict] = None,
                 accepted: List[dict] = None,
                 uncertain: List[dict] = None,
                 rejected: List[dict] = None,
                 pre_accepted: List[dict] = None,
                 reference_overlap: dict = None):
    """Save per-experiment results (Phase 1–5 output).

    Produces:
      - {name}.json              — summary metrics, config, round history
      - {name}_lexicon.json      — complete final lexicon
      - {name}_candidates.json   — all candidates with classification

    Phase 6 (LLM evaluation) runs post-hoc across all experiments.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Build lexicon sample for summary ──
    lexicon_sample = []
    for c in lexicon[:50]:
        entry = {'word': c['word'],
                 'category': c.get('category', ''),
                 'similarity': c.get('similarity', 0),
                 'wordhood': c.get('wordhood'),
                 'combined_score': c.get('combined_score'),
                 'oracle_label': c.get('oracle_label')}
        lexicon_sample.append(entry)

    # ── Main result file (summary) ──
    result = {
        'experiment': experiment_name,
        'config': config_info,
        'metrics': metrics,
        'reference_overlap': reference_overlap,
        'round_history': round_history,
        'lexicon_size': len(lexicon),
        'lexicon_sample': lexicon_sample,
    }

    fpath = os.path.join(RESULTS_DIR, f"{experiment_name}.json")
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary saved to {fpath}")

    # ── Full lexicon file ──
    lexicon_export = []
    for c in lexicon:
        entry = {'word': c['word']}
        for key in ('category', 'similarity', 'wordhood', 'combined_score',
                     'pmi', 'npmi', 'freq',
                     'power_related', 'has_dialect_anchor',
                     'oracle_label', 'oracle_reason', 'oracle_category',
                     'oracle_has_dialect', 'oracle_power_relevant',
                     'oracle_confidence',
                     'phase35_oracle_label', 'phase35_oracle_reason', 'source'):
            if key in c:
                entry[key] = c[key]
        lexicon_export.append(entry)

    lex_path = os.path.join(RESULTS_DIR, f"{experiment_name}_lexicon.json")
    with open(lex_path, 'w', encoding='utf-8') as f:
        json.dump(lexicon_export, f, ensure_ascii=False, indent=2)
    logger.info(f"Full lexicon ({len(lexicon_export)} terms) saved to {lex_path}")

    # ── Candidate pool file ──
    if all_candidates is not None:
        lexicon_words = {c['word'] for c in lexicon}
        candidate_export = _build_candidate_export(
            all_candidates, accepted, uncertain, rejected, pre_accepted,
            lexicon_words)
        cand_path = os.path.join(RESULTS_DIR, f"{experiment_name}_candidates.json")
        with open(cand_path, 'w', encoding='utf-8') as f:
            json.dump(candidate_export, f, ensure_ascii=False, indent=2)
        logger.info(f"Candidate pool ({len(all_candidates)} total) saved to {cand_path}")

    return fpath


def _build_candidate_export(all_candidates, accepted, uncertain, rejected,
                             pre_accepted, lexicon_words: set = None) -> dict:
    """Build a comprehensive candidate export with classification metadata."""
    accepted_words = {c['word']: c for c in (accepted or [])}
    uncertain_words = {c['word']: c for c in (uncertain or [])}
    rejected_words = {c['word']: c for c in (rejected or [])}
    pre_acc_words = {c['word']: c for c in (pre_accepted or [])}
    final_words = lexicon_words or set()

    export_list = []
    stats = {'total': len(all_candidates), 'accepted': 0, 'uncertain': 0,
             'rejected': 0, 'pre_accepted_verified': 0, 'pre_accepted_rejected': 0}

    for c in all_candidates:
        w = c['word']
        entry = {
            'word': w,
            'pmi': c.get('pmi', 0),
            'npmi': c.get('npmi', 0),
            'freq': c.get('freq', 0),
            'source': c.get('source', 'bigram'),
            'power_related': c.get('power_related'),
            'has_dialect_anchor': c.get('has_dialect_anchor'),
        }
        if w in accepted_words:
            entry['phase2'] = 'accepted'
            entry['similarity'] = accepted_words[w].get('similarity')
            entry['wordhood'] = accepted_words[w].get('wordhood')
            entry['combined_score'] = accepted_words[w].get('combined_score')
            entry['category'] = accepted_words[w].get('category')
            if w in pre_acc_words:
                lbl = pre_acc_words[w].get('phase35_oracle_label')
                entry['phase35_oracle'] = lbl
                entry['phase35_reason'] = pre_acc_words[w].get('phase35_oracle_reason')
                if lbl is True:
                    stats['pre_accepted_verified'] += 1
                elif lbl is False:
                    stats['pre_accepted_rejected'] += 1
                # else None: oracle was not run (--diagnose-phase35 not set)
            stats['accepted'] += 1
        elif w in uncertain_words:
            entry['phase2'] = 'uncertain'
            entry['similarity'] = uncertain_words[w].get('similarity')
            entry['wordhood'] = uncertain_words[w].get('wordhood')
            entry['combined_score'] = uncertain_words[w].get('combined_score')
            entry['category'] = uncertain_words[w].get('category')
            stats['uncertain'] += 1
        elif w in rejected_words:
            entry['phase2'] = 'rejected'
            entry['similarity'] = rejected_words[w].get('similarity')
            entry['wordhood'] = rejected_words[w].get('wordhood')
            entry['combined_score'] = rejected_words[w].get('combined_score')
            stats['rejected'] += 1

        entry['in_final_lexicon'] = w in final_words
        export_list.append(entry)

    phase_order = {'accepted': 0, 'uncertain': 1, 'rejected': 2}
    export_list.sort(key=lambda x: (
        phase_order.get(x.get('phase2', 'rejected'), 2),
        -(x.get('similarity') or 0)
    ))

    return {'stats': stats, 'candidates': export_list}


def generate_comparison_table(result_files: List[str]) -> str:
    """Generate a pre-calibration comparison table (basic metrics only).

    Shows lexicon size, oracle queries, and reference overlap.
    Used for quick iteration before running Phase 6 LLM evaluation.
    For final paper metrics (DP/PDP/FER/AvgConf), see _build_comparison_table
    in run_all.py which runs after post-hoc LLM evaluation.
    """
    rows = []
    for fpath in result_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ref = data.get('reference_overlap') or {}
        m = data.get('metrics') or {}

        rows.append({
            'name': data['config'].get('name', data['experiment']),
            'size': m.get('predicted_size', 0),
            'queries': m.get('oracle_queries', 'N/A'),
            'ref_exact': ref.get('exact_matches', '—'),
            'ref_token': ref.get('token_overlap_matches', '—'),
        })

    lines = [
        "| Model | Lexicon | Queries | Ref(Exact) | Ref(Token) |",
        "|-------|---------|---------|------------|------------|",
    ]
    for r in sorted(rows, key=lambda x: x['size'], reverse=True):
        lines.append(
            f"| {r['name']} | {r['size']} | {r['queries']} | "
            f"{r['ref_exact']} | {r['ref_token']} |")

    return "\n".join(lines)
