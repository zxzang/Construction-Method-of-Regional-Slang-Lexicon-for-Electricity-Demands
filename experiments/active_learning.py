# -*- coding: utf-8 -*-
"""
Active Learning loop orchestrating candidate selection and oracle verification.
"""
import logging
import copy
from typing import List, Dict, Tuple
from config import AL_MAX_ROUNDS, TDCAL_TOP_K

logger = logging.getLogger(__name__)


class ActiveLearningLoop:
    """
    Iterative AL loop: select uncertain candidates -> query oracle -> update lexicon.
    Supports margin, tdcal, and random selection strategies.
    """

    def __init__(self, semantic_mapper, oracle, candidate_generator=None,
                 al_strategy: str = 'margin', max_rounds: int = AL_MAX_ROUNDS,
                 top_k: int = TDCAL_TOP_K, seed: int = 42):
        self.mapper = semantic_mapper
        self.oracle = oracle
        self.cand_gen = candidate_generator  # Needed for TD-CAL density
        self.strategy = al_strategy
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.seed = seed
        self.history: List[dict] = []  # Per-round stats

    def run(self, accepted: List[dict], uncertain_pool: List[dict]
            ) -> Tuple[List[dict], List[dict]]:
        """
        Run the active learning loop.
        
        Args:
            accepted: initially auto-accepted candidates
            uncertain_pool: candidates needing oracle verification
            
        Returns:
            final_lexicon: all verified slang terms
            round_history: per-round statistics
        """
        lexicon = list(accepted)
        pool = copy.deepcopy(uncertain_pool)
        self.history = []

        for round_num in range(1, self.max_rounds + 1):
            if not pool:
                logger.info(f"Round {round_num}: pool empty, stopping")
                break

            # 1. Select candidates using the configured strategy
            if self.strategy == 'tdcal' and self.cand_gen:
                selected = self.mapper.tdcal_select(
                    pool, self.cand_gen, top_k=self.top_k)
            elif self.strategy == 'margin':
                selected = self.mapper.margin_select(pool, top_k=self.top_k)
            elif self.strategy == 'random':
                selected = self.mapper.random_select(
                    pool, top_k=self.top_k, seed=self.seed + round_num)
            else:
                raise ValueError(
                    f"Unknown AL strategy: '{self.strategy}'. "
                    f"Valid options: margin, tdcal, random.")

            # 2. Query oracle
            judged = self.oracle.judge_batch(selected)
            verified = [c for c in judged if c.get('oracle_label', False)]
            rejected = [c for c in judged if not c.get('oracle_label', False)]

            # 3. Update lexicon and pool
            lexicon.extend(verified)
            selected_words = {c['word'] for c in selected}
            pool = [c for c in pool if c['word'] not in selected_words]

            # 4. Record stats (with full detail for analysis)
            round_stats = {
                'round': round_num,
                'pool_size': len(pool) + len(selected),
                'selected': len(selected),
                'verified': len(verified),
                'rejected': len(rejected),
                'lexicon_size': len(lexicon),
                'oracle_queries_total': self.oracle.query_count,
                'verified_words': [c['word'] for c in verified],
                # Detailed per-candidate records
                'selected_detail': [
                    {'word': c['word'],
                     'similarity': c.get('similarity'),
                     'wordhood': c.get('wordhood'),
                     'combined_score': c.get('combined_score'),
                     'category': c.get('category'),
                     'tdcal_score': c.get('tdcal_score'),
                     'margin_dist': c.get('margin_dist')}
                    for c in selected
                ],
                'judged_detail': [
                    {'word': c['word'],
                     'oracle_label': c.get('oracle_label', False),
                     'oracle_reason': c.get('oracle_reason', ''),
                     'oracle_category': c.get('oracle_category', ''),
                     'oracle_has_dialect': c.get('oracle_has_dialect'),
                     'oracle_power_relevant': c.get('oracle_power_relevant'),
                     'oracle_confidence': c.get('oracle_confidence')}
                    for c in judged
                ],
            }
            self.history.append(round_stats)
            logger.info(f"Round {round_num}: selected={len(selected)}, "
                        f"verified={len(verified)}, lexicon={len(lexicon)}, "
                        f"pool_remaining={len(pool)}")

        return lexicon, self.history
