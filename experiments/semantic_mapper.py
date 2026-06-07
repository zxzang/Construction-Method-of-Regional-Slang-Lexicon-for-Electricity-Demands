# -*- coding: utf-8 -*-
"""
Semantic Mapper using Sentence-BERT and FAISS.
Implements standard cosine similarity and TD-CAL contrastive scoring.
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

import os
# Network / mirror configuration for HF model downloads.
# These are set here so they take effect before the SentenceTransformer import
# below.  In production deployments, prefer to set these in the environment
# before starting the process (HF_ENDPOINT, CURL_CA_BUNDLE, TRANSFORMERS_VERBOSITY).
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "TRANSFORMERS_VERBOSITY" not in os.environ:
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# Ignore SSL certificate issues (compatible with older versions of Linux systems)
os.environ["CURL_CA_BUNDLE"] = ""
# NOTE: CURL_CA_BUNDLE="" disables SSL certificate verification entirely.
# Only set this when running on legacy systems with outdated CA bundles.
# Do NOT set it unconditionally — prefer updating the system CA certificates.
from sentence_transformers import SentenceTransformer

import faiss
import jieba
from config import (SBERT_MODEL, POWER_SEEDS, SIM_HIGH, SIM_LOW,
                    SIM_BOUNDARY, TDCAL_TOP_K, WORDHOOD_ALPHA, WORDHOOD_BETA,
                    DIALECT_PARTICLE_CHARS)

logger = logging.getLogger(__name__)


class WordhoodScorer:
    """Computes lexical-fixedness signals complementary to SBERT similarity.

    Measures whether a candidate resists jieba segmentation (indicating
    it's a fixed expression) vs. being split into particles (indicating
    a grammatical fragment). Score range [0, 1], higher = more word-like.
    """

    def score(self, word: str) -> float:
        """Compute wordhood score for a single candidate."""
        tokens = jieba.lcut(word)
        n_tokens = len(tokens)
        n_chars = len(word)

        # Segmentation resistance: 1.0 when jieba keeps it as one piece,
        # drops toward 0.0 as more pieces are produced relative to word length.
        seg_score = max(0.0, 1.0 - (n_tokens - 1) / max(n_chars, 1))

        # Penalty for each pure-particle token in the segmentation
        particle_count = sum(1 for t in tokens if t in DIALECT_PARTICLE_CHARS)
        particle_penalty = 0.30 * particle_count / max(n_tokens, 1)

        return round(max(0.0, min(1.0, seg_score - particle_penalty)), 4)

    def score_batch(self, words: List[str]) -> List[float]:
        return [self.score(w) for w in words]


class SemanticMapper:
    """Maps dialect candidates to power domain categories via SBERT + FAISS.

    Uses combined scoring: WORDHOOD_ALPHA * SBERT_similarity
                          + WORDHOOD_BETA * wordhood_score
    to distinguish lexical items from grammatical fragments.
    """

    def __init__(self, model_name: str = SBERT_MODEL):
        logger.info(f"Loading SBERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.seed_categories = POWER_SEEDS
        self.seed_labels: List[str] = []
        self.seed_embeddings = None
        self.faiss_index = None
        self.wordhood = WordhoodScorer()
        self._build_seed_index()

    def _build_seed_index(self):
        all_words, self.seed_labels = [], []
        for cat, words in self.seed_categories.items():
            for w in words:
                all_words.append(w)
                self.seed_labels.append(cat)
        self.seed_embeddings = self.model.encode(all_words, normalize_embeddings=True)
        dim = self.seed_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.seed_embeddings.astype(np.float32))
        logger.info(f"FAISS index: {len(all_words)} seeds, dim={dim}")

    def encode_candidates(self, candidates: List[str]) -> np.ndarray:
        return self.model.encode(candidates, normalize_embeddings=True,
                                 show_progress_bar=False)

    def classify_candidates(self, candidates: List[dict]):
        """Split candidates into accepted/uncertain/rejected by combined score.

        combined = WORDHOOD_ALPHA * SBERT_similarity + WORDHOOD_BETA * wordhood_score

        The wordhood component penalizes candidates that jieba segments into
        grammatical particles (e.g. "V+着", "N+呢") and boosts candidates that
        resist segmentation (fixed expressions).
        """
        words = [c['word'] for c in candidates]
        if not words:
            return [], [], []
        embs = self.encode_candidates(words)
        scores, indices = self.faiss_index.search(embs.astype(np.float32), k=1)
        wh_scores = self.wordhood.score_batch(words)

        accepted, uncertain, rejected = [], [], []
        for i, cand in enumerate(candidates):
            sim = float(scores[i, 0])
            wh = wh_scores[i]
            combined = WORDHOOD_ALPHA * sim + WORDHOOD_BETA * wh
            cat = self.seed_labels[int(indices[i, 0])]
            r = {**cand,
                 'similarity': round(sim, 4),
                 'wordhood': round(wh, 4),
                 'combined_score': round(combined, 4),
                 'category': cat}
            if combined >= SIM_HIGH:
                accepted.append(r)
            elif combined >= SIM_LOW:
                uncertain.append(r)
            else:
                rejected.append(r)

        logger.info(f"Classified: {len(accepted)} accept, "
                     f"{len(uncertain)} uncertain, {len(rejected)} reject "
                     f"(using combined = {WORDHOOD_ALPHA}*sim + {WORDHOOD_BETA}*wordhood)")
        return accepted, uncertain, rejected

    def tdcal_select(self, pool: List[dict], cand_gen, top_k=TDCAL_TOP_K):
        """TD-CAL: density * uncertainty scoring.

        Uses PMI as the density signal — stronger dialect-anchor association
        implies higher semantic density.  PMI varies meaningfully even in
        small corpora (range ~2-12 for attested bigrams), unlike tf-idf
        entropy which collapses to near-constant for rare dialect pairs.

        Uncertainty is computed from combined_score to be consistent with
        classify_candidates().
        """
        # 1. Use PMI as raw density (fallback: 0 for unigrams without PMI)
        for c in pool:
            c['raw_density'] = c.get('pmi', 0.0)

        # 2. Min-max normalize densities to 0-1
        max_d = max([c['raw_density'] for c in pool]) if pool else 1.0
        min_d = min([c['raw_density'] for c in pool]) if pool else 0.0
        range_d = max_d - min_d if max_d > min_d else 1.0

        for c in pool:
            c['density'] = round((c['raw_density'] - min_d) / range_d, 6)
            # 3. Uncertainty from combined_score (consistent with classification)
            combined = c.get('combined_score', c.get('similarity', 0.5))
            margin = abs(combined - SIM_BOUNDARY)
            max_margin = max(SIM_HIGH - SIM_BOUNDARY, SIM_BOUNDARY - SIM_LOW, 0.01)
            uncertainty = 1.0 - min(margin / max_margin, 1.0)
            # 4. TD-CAL score: balanced sum of density and uncertainty
            c['tdcal_score'] = round(0.5 * c['density'] + 0.5 * uncertainty, 6)

        pool.sort(key=lambda x: x['tdcal_score'], reverse=True)
        return pool[:top_k]

    def margin_select(self, pool: List[dict], top_k=TDCAL_TOP_K):
        """Standard margin sampling: closest to decision boundary.

        Uses combined_score to be consistent with classify_candidates().
        """
        for c in pool:
            combined = c.get('combined_score', c.get('similarity', 0.5))
            c['margin_dist'] = abs(combined - SIM_BOUNDARY)
        pool.sort(key=lambda x: x['margin_dist'])
        return pool[:top_k]

    def random_select(self, pool: List[dict], top_k=TDCAL_TOP_K, seed=42):
        """Random sampling baseline."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pool), size=min(top_k, len(pool)), replace=False)
        return [pool[i] for i in idx]
