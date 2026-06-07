# -*- coding: utf-8 -*-
"""
Candidate Generator using NPMI (Normalized PMI) with Edit-Distance filtering.
Implements both raw PMI (baseline) and NPMI+EditDist (proposed) modes.
"""
import math
import re
import logging
from typing import List, Dict, Tuple, Set
from collections import defaultdict

import jieba
import os
import sys
import tempfile
# Fix jieba cache permission error on shared servers
# os.getuid() is Unix-only; use getpass on Windows
if sys.platform == 'win32':
    import getpass
    _uid = getpass.getuser()
else:
    _uid = str(os.getuid())
_tmp_dir = os.path.join(tempfile.gettempdir(), f'jieba_cache_{_uid}')
os.makedirs(_tmp_dir, exist_ok=True)
jieba.dt.tmp_dir = _tmp_dir
import Levenshtein

from config import (YUXI_ANCHORS, ALL_SEEDS, PMI_THRESHOLD,
                    NPMI_THRESHOLD, MIN_BIGRAM_FREQ, EDIT_DISTANCE_MAX,
                    DIALECT_PURE_PARTICLE_COMBOS, DIALECT_PARTICLE_CHARS)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DEPRECATED: Dialect function words and dual NPMI thresholds.
# All current experiments use use_npmi=False (PMI mode), so the dual-threshold
# NPMI logic (FUNC_NPMI_THRESHOLD, DIALECT_FUNCTION_WORDS, is_typo_of_seed,
# compute_npmi) is not exercised.  Kept for historical reference.
# ---------------------------------------------------------------------------
DIALECT_FUNCTION_WORDS = {
    '咋个', '呢', '着', '给有', '噶', '嘛', '哈',
    '了嘛', '得嘛', '挨', '冇', '木', '是呢', '得了',
}

# NPMI threshold for bigrams where the anchor is a dialect function word.
# Standard threshold (0.15) is too permissive for function words because
# they co-occur with nearly everything, producing low-quality bigrams.
FUNC_NPMI_THRESHOLD = 0.12
# Unified with NPMI_THRESHOLD: bidirectional anchoring already filters
# power-irrelevant bigrams regardless of anchor type, so a separate
# higher threshold for function-word anchors is no longer needed.

# Dialect suffixes / prefixes used for unigram fragment detection
_DIALECT_SUFFIXES = ['呢', '噶', '嘛', '哈', '着', '了', '得']
_DIALECT_PREFIX_WORDS = ['咋个', '给有', '挨', '冇', '木']

# Known genuine dialect expressions that contain particles — these should
# NOT be filtered out even if they match fragment patterns.
_DIALECT_KNOWN_GOOD = {
    '真呢', '好呢', '好好呢', '等着', '等哈', '不来电呢', '冇交',
    '冇住', '停电哈', '多收呢', '着电', '着钱', '着戳', '线路老火',
    '我家呢', '换着', '早说嘛', '挨我们', '挨电费',
    # Base dialect content-word anchors (2-char) — exempted from the
    # len(word)<=2 fragment check so they can enter the lexicon via the
    # anchor-injection or unigram paths.
    '扎实', '鬼火', '老火', '得行', '咋个', '给有', '是呢', '咋整',
}


def _is_grammatical_fragment(word: str) -> bool:
    """Detect candidates that are grammatical fragments, not lexical items.

    Catches several patterns common in Yuxi dialect bigram extraction:
    1. Pure function-word combos (e.g. "呢嘛", "了嘛") — never lexical
    2. Short-stem suffix/prefix fragments (e.g. "烂呢") — already caught
    3. Jieba-segmentable constructions where one token is a pure particle
       (e.g. "拿着" → "拿"+"着", "电呢" → "电"+"呢")
    """
    # Whitelisted known-good expressions
    if word in _DIALECT_KNOWN_GOOD:
        return False

    if len(word) <= 2:
        return True

    # 1. Pure function-word combos (never lexical items)
    if word in DIALECT_PURE_PARTICLE_COMBOS:
        return True

    # 2. Short-stem suffix/prefix fragments (existing logic, now >= 2 check)
    for sfx in _DIALECT_SUFFIXES:
        if word.endswith(sfx):
            stem_len = len(word) - len(sfx)
            if stem_len < 2:
                return True
    for pfx in _DIALECT_PREFIX_WORDS:
        if word.startswith(pfx):
            rest_len = len(word) - len(pfx)
            if rest_len < 2:
                return True

    # 3. Jieba segmentation check: if jieba splits the word and at least
    #    one resulting token is a pure dialect particle, it's a fragment.
    #    e.g. "拿着" → ["拿", "着"] → particle "着" → fragment
    #    e.g. "着电" → ["着电"] (injected) → no split → keep
    tokens = jieba.lcut(word)
    if len(tokens) > 1:
        has_particle = any(t in DIALECT_PARTICLE_CHARS for t in tokens)
        if has_particle:
            return True

    return False


class CandidateGenerator:
    """
    Extract dialect slang candidates from corpus using PMI/NPMI
    anchored by known dialect markers.
    """

    # Power-domain content characters — used for bidirectional anchoring check
    # (same set as RuleBasedOracle._POWER_CHARS)
    _POWER_CHARS = set('电线路闸表费火跳停修炸漏烧压杆变容开关供配')

    def __init__(self, anchors: List[str] = None, seeds: List[str] = None,
                 extended_anchors: List[str] = None):
        self.anchors = set(anchors or YUXI_ANCHORS)
        if extended_anchors:
            self.anchors.update(extended_anchors)
            # Reference vocabulary terms are legitimate dialect words — exempt
            # them from fragment filtering.  Some end with particles (哈/嘛/噶)
            # and otherwise match fragment patterns.
            _DIALECT_KNOWN_GOOD.update(extended_anchors)
        self.seeds = set(seeds or ALL_SEEDS)
        # Register known words with jieba
        for word in self.anchors | self.seeds:
            jieba.add_word(word)

        self.unigram_freq: Dict[str, int] = defaultdict(int)
        self.bigram_freq: Dict[Tuple[str, str], int] = defaultdict(int)
        self.total_tokens: int = 0
        self._tokenized_corpus: List[List[str]] = []

    # Short power seed terms (≤3 chars) for substring matching in bigrams.
    # Populated once at class load; catches terms like "负荷", "来电" that
    # don't contain the character-level power chars but are legitimate power
    # domain vocabulary.
    _POWER_TERMS_SHORT = frozenset(
        s for s in ALL_SEEDS if 2 <= len(s) <= 3
    )

    @staticmethod
    def _is_power_related(word: str) -> bool:
        """Check if a word contains power-domain characters or terms."""
        # Character-level check (fast, catches 80%+ of power terms)
        if any(c in CandidateGenerator._POWER_CHARS for c in word):
            return True
        # Term-level check (catches "负荷", "来电" etc. that lack power chars)
        for term in CandidateGenerator._POWER_TERMS_SHORT:
            if term in word:
                return True
        return False

    # Dialect-specific characters for unigram dialect-feature detection.
    # These are uniquely characteristic of Yuxi dialect morphology.
    # IMPORTANT: do NOT include common power-domain characters (电/线/闸/...)
    # here, otherwise standard Chinese power terms pass the dialect filter.
    # Dialect-specific characters for unigram filtering.
    # Excludes DIALECT_PARTICLE_CHARS members '过' (as in "莫过于") and
    # content chars '老' (as in "老百姓") — these appear in too many
    # standard Chinese words.  Legitimate dialect words with these chars
    # (老火/得过) are captured by the anchor path, not unigram_cooccur.
    _DIALECT_CHARS = (
        (DIALECT_PARTICLE_CHARS - {'过'})
        | {'咋', '冇', '木', '挨',      # dialect function words
           '扎', '实', '鬼', '火'}      # dialect content (扎实/鬼火)
    )

    @staticmethod
    def _has_dialect_features(word: str) -> bool:
        """Check if a word exhibits Yuxi dialect morphological features.

        Returns True when the word contains uniquely dialect characters
        (particles, function words, dialect content chars).  Excludes
        characters shared with standard Chinese power vocabulary ('电' etc.)
        to avoid passing words like "供电所" or "用电量".
        """
        return any(c in CandidateGenerator._DIALECT_CHARS for c in word)

    def build_frequency_model(self, corpus: List[str]):
        """Tokenize corpus and count unigram/bigram frequencies."""
        logger.info("Building frequency model from corpus...")
        self.unigram_freq.clear()
        self.bigram_freq.clear()
        self.total_tokens = 0
        self._tokenized_corpus = []

        for text in corpus:
            tokens = [t for t in jieba.cut(text) if t.strip()]
            self._tokenized_corpus.append(tokens)
            for i, w in enumerate(tokens):
                self.unigram_freq[w] += 1
                self.total_tokens += 1
                if i < len(tokens) - 1:
                    self.bigram_freq[(w, tokens[i + 1])] += 1

        logger.info(f"Vocabulary: {len(self.unigram_freq)} unigrams, "
                     f"{len(self.bigram_freq)} bigrams, "
                     f"{self.total_tokens} total tokens")

    def compute_pmi(self, w1: str, w2: str) -> float:
        """Compute Pointwise Mutual Information for a word pair."""
        p_w1 = self.unigram_freq[w1] / self.total_tokens
        p_w2 = self.unigram_freq[w2] / self.total_tokens
        p_w1w2 = self.bigram_freq[(w1, w2)] / self.total_tokens
        if p_w1 * p_w2 == 0 or p_w1w2 == 0:
            return 0.0
        return math.log2(p_w1w2 / (p_w1 * p_w2))

    def compute_npmi(self, w1: str, w2: str) -> float:
        """
        Compute Normalized PMI (range: -1 to 1).
        NPMI = PMI / -log2(P(w1, w2))
        """
        p_w1w2 = self.bigram_freq[(w1, w2)] / self.total_tokens
        if p_w1w2 == 0:
            return -1.0
        pmi = self.compute_pmi(w1, w2)
        h = -math.log2(p_w1w2)
        if h == 0:
            return 0.0
        return pmi / h

    @staticmethod
    def is_typo_of_seed(candidate: str, seeds: Set[str],
                        max_dist: int = EDIT_DISTANCE_MAX) -> bool:
        """
        Check if a candidate is a typo/OCR error of a known seed word
        (edit distance <= max_dist). If so, it's noise, not novel slang.
        """
        # For short Chinese words (len <= 2), edit distance 1 is too aggressive
        # as it filters out completely different words sharing one character.
        if len(candidate) <= 2:
            return False

        for seed in seeds:
            # Skip when length difference >= 2: a 4-char compound is not a
            # typo of a 2-char seed (e.g. "线路老火" ⊃ "线路", not a typo).
            if abs(len(candidate) - len(seed)) >= 2:
                continue
            if Levenshtein.distance(candidate, seed) <= max_dist and candidate != seed:
                return True
        return False

    def extract_candidates(self, use_npmi: bool = True,
                            hard_power_gate: bool = False) -> List[dict]:
        """
        Extract dialect slang candidates from the corpus.

        Args:
            use_npmi: If True, use Normalized PMI + edit-distance filter.
                      If False, use raw PMI (baseline mode).
            hard_power_gate: If True, filter out power-irrelevant bigrams
                      (old behavior before soft-label). Default False.
                      Used for soft-vs-hard ablation study.

        Returns:
            List of dicts: [{'word': str, 'pmi': float, 'npmi': float, 'freq': int}, ...]
        """
        logger.info(f"Extracting candidates (use_npmi={use_npmi})...")
        candidates = []
        seen = set()

        for (w1, w2), freq in self.bigram_freq.items():
            if freq < MIN_BIGRAM_FREQ:
                continue
            # At least one word must be a dialect anchor
            if w1 not in self.anchors and w2 not in self.anchors:
                continue

            merged = w1 + w2
            if merged in seen:
                continue
            # Must contain Chinese characters
            if not re.search(r'[\u4e00-\u9fa5]{2,}', merged):
                continue

            pmi_val = self.compute_pmi(w1, w2)
            npmi_val = self.compute_npmi(w1, w2)

            if use_npmi:
                # Dual-threshold: function-word anchors get stricter NPMI
                # because they co-occur with everything and produce fragments.
                threshold = FUNC_NPMI_THRESHOLD if (
                    w1 in DIALECT_FUNCTION_WORDS or w2 in DIALECT_FUNCTION_WORDS
                ) else NPMI_THRESHOLD
                if npmi_val < threshold:
                    continue
                if self.is_typo_of_seed(merged, self.seeds):
                    continue
            else:
                # Baseline: raw PMI threshold only
                if pmi_val < PMI_THRESHOLD:
                    continue

            # ── Bidirectional anchoring ──
            # Two independent dimensions stored for ablation analysis:
            #   has_dialect_anchor: anchor component is a content word
            #   power_related:       word contains power-domain chars/terms
            w1_content = (w1 in self.anchors
                          and w1 not in DIALECT_FUNCTION_WORDS)
            w2_content = (w2 in self.anchors
                          and w2 not in DIALECT_FUNCTION_WORDS)
            has_dialect_anchor = (w1_content or w2_content)
            power_related = (self._is_power_related(w1)
                             or self._is_power_related(w2)
                             or self._is_power_related(merged))

            if hard_power_gate and not power_related:
                continue

            seen.add(merged)
            candidates.append({
                'word': merged,
                'pmi': round(pmi_val, 4),
                'npmi': round(npmi_val, 4),
                'freq': freq,
                'power_related': power_related,
                'has_dialect_anchor': has_dialect_anchor,
            })

        # --- Unigram candidates: words co-occurring with power seeds ---
        # This captures standalone dialect words (e.g., "扎实", "咋个")
        # that appear in the same message as power domain terms.
        cooccur_words = defaultdict(int)
        for tokens in self._tokenized_corpus:
            token_set = set(tokens)
            has_seed = bool(token_set & self.seeds)
            if has_seed:
                for t in tokens:
                    if (len(t) >= 2
                            and re.search(r'[\u4e00-\u9fa5]{2,}', t)
                            and t not in self.seeds):
                        cooccur_words[t] += 1

        for word, freq in cooccur_words.items():
            if word in seen or freq < 5:
                continue
            # Skip purely grammatical function-word fragments
            if _is_grammatical_fragment(word):
                continue
            # Skip very common Chinese words (stopword-like)
            if self.unigram_freq[word] > self.total_tokens * 0.01:
                continue
            if use_npmi and self.is_typo_of_seed(word, self.seeds):
                continue
            # ── Dialect-feature filter ──
            # Unigrams extracted via co-occurrence with power seeds are
            # predominantly standard Chinese vocabulary ("搞清楚",
            # "便民服务").  Require dialect morphology to keep only
            # terms with genuine Yuxi dialect character features.
            if not self._has_dialect_features(word):
                continue
            # Hard-gate check for unigrams: must have power-domain content.
            if hard_power_gate and not self._is_power_related(word):
                continue
            seen.add(word)
            candidates.append({
                'word': word,
                'pmi': 0.0,
                'npmi': 0.0,
                'freq': freq,
                'source': 'unigram_cooccur',
                'power_related': self._is_power_related(word),
                'has_dialect_anchor': False,
            })

        # Also add anchor words themselves (skip function-word anchors).
        # Threshold freq >= 2 (not 3): extended anchors from the reference
        # vocabulary include rare dialect expressions ("不来电呢", "着电")
        # that may appear only twice.  The jieba registration of these
        # anchors also causes them to be tokenized as single unigrams,
        # preventing their extraction via the bigram path.
        for word in self.anchors:
            if word not in seen and self.unigram_freq[word] >= 2:
                if len(word) >= 2 and not _is_grammatical_fragment(word):
                    # Hard-gate check for anchor-injected candidates.
                    if hard_power_gate and not self._is_power_related(word):
                        continue
                    seen.add(word)
                    candidates.append({
                        'word': word, 'pmi': 0.0, 'npmi': 0.0,
                        'freq': self.unigram_freq[word],
                        'source': 'anchor',
                        'power_related': self._is_power_related(word),
                        'has_dialect_anchor': word in self.anchors
                            and word not in DIALECT_FUNCTION_WORDS,
                    })

        # Sort by NPMI descending (bigrams first), then freq
        candidates.sort(key=lambda x: (x['npmi'], x['freq'], x['word']), reverse=True)
        logger.info(f"Extracted {len(candidates)} candidates "
                     f"({len([c for c in candidates if c.get('source') == 'unigram_cooccur'])} unigrams)")
        return candidates

    def get_tfidf_entropy(self, word: str) -> Tuple[float, float]:
        """
        Compute TF-IDF-like score and entropy for a word.
        Used by TD-CAL for token-density scoring.
        """
        tf = self.unigram_freq.get(word, 0) / max(self.total_tokens, 1)
        # IDF: log(N_docs / (1 + df))  where df = number of messages containing word
        n_docs = len(self._tokenized_corpus)
        df = sum(1 for tokens in self._tokenized_corpus if word in tokens)
        idf = math.log((n_docs + 1) / (df + 1)) + 1

        # Entropy of word distribution across documents
        if df == 0:
            entropy = 0.0
        else:
            p = df / n_docs
            entropy = -(p * math.log2(p + 1e-10) + (1 - p) * math.log2(1 - p + 1e-10))

        tfidf = tf * idf
        return tfidf, entropy
