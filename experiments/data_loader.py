# -*- coding: utf-8 -*-
"""
Data loading and preprocessing module.
Handles WeChat CSV noise: XML, emojis, wxid, URLs, system messages.
"""
import pandas as pd
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Robust preprocessor for extremely noisy WeChat short texts."""

    # Noise patterns ordered by priority
    NOISE_PATTERNS = [
        (r'标题:.*?描述:.*', ''),               # XML formatted bot messages
        (r'无法解析的XML消息', ''),               # Unparseable XML
        (r'<msg>.*?</msg>', ''),                  # System XML messages (dotall)
        (r'wxid_[a-z0-9]+', ''),                  # User unique IDs
        (r'"wxid_[a-z0-9]+"', ''),                # Quoted user IDs
        (r'\[.*?\]', ''),                          # WeChat emojis [捂脸] etc.
        (r'http[s]?://\S+', ''),                  # URL links
        (r'@\S+', ''),                            # @mentions
        (r'拍了拍', ''),                           # Platform interaction
        (r'邀请.*?加入了群聊', ''),                # Join messages
        (r'撤回了一条消息', ''),                   # Recall messages
        (r'[\d\-:.\s]{10,25}', ' '),              # Redundant timestamps
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean a single message text."""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        for pattern, replacement in cls.NOISE_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.DOTALL)
        # Keep Chinese chars, letters, digits, and minimal punctuation
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！、]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @classmethod
    def load_csv(cls, file_path: str) -> pd.DataFrame:
        """Load a single CSV file with robust encoding handling.

        Handles two WeChat export formats:
        - File 1: standard comma-separated CSV with 6 columns.
        - File 2: tab-separated rows wrapped in double-quotes with trailing
          commas (``\"col1\\tcol2\\t...\",,``).  Pandas' default ``sep='\\t'``
          treats the whole quoted line as one field; we strip the outer quotes
          and trailing commas first, then re-read with tab separator.
        """
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig', on_bad_lines='skip')
            df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]

            # Detect tab-delimited single-column format (file 2)
            if len(df.columns) <= 3 and '\t' in str(df.columns[0]):
                # First attempt: simple tab re-read
                df = pd.read_csv(file_path, encoding='utf-8-sig',
                                 sep='\t', on_bad_lines='skip')
                df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]

                # If still a single column, the file uses quoted-tab format
                # (whole row wrapped in double-quotes).  Strip outer quotes
                # and trailing commas, then re-parse with tab separator.
                if df.shape[1] == 1 and '\t' in str(df.columns[0]):
                    import io
                    with open(file_path, 'r', encoding='utf-8-sig') as fh:
                        raw = fh.read()
                    # Strip outer double-quotes and trailing commas (WeChat
                    # export artifact: each tab-delimited row is wrapped in
                    # double-quotes with trailing commas, e.g.
                    # "col1\tcol2\tcol3",, → col1\tcol2\tcol3
                    cleaned_lines = []
                    for line in raw.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        # Remove trailing commas first (they sit outside the
                        # closing quote), then strip the enclosing quotes.
                        line = line.rstrip(',').strip()
                        if line.startswith('"') and line.endswith('"'):
                            line = line[1:-1]
                        if line:
                            cleaned_lines.append(line)
                    cleaned = '\n'.join(cleaned_lines)
                    df = pd.read_csv(io.StringIO(cleaned), sep='\t',
                                     quoting=3,  # QUOTE_NONE
                                     on_bad_lines='skip',
                                     engine='python')
                    df.columns = [c.replace('\ufeff', '').strip()
                                  for c in df.columns]

            return df
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return pd.DataFrame()

    @classmethod
    def load_and_preprocess(cls, file_paths: List[str]) -> List[str]:
        """
        Load multiple CSV files, extract and clean message texts.
        Returns a list of cleaned non-empty strings.
        """
        logger.info(f"Loading {len(file_paths)} data files...")
        all_messages = []

        for fpath in file_paths:
            df = cls.load_csv(fpath)
            if df.empty:
                continue

            # Find the message column (usually named '消息' or index 3)
            msg_col = None
            for c in df.columns:
                if '消息' in str(c):
                    msg_col = c
                    break
            if msg_col is None and len(df.columns) > 3:
                msg_col = df.columns[3]
            elif msg_col is None:
                msg_col = df.columns[-1]

            messages = df[msg_col].dropna().astype(str).tolist()
            all_messages.extend(messages)
            logger.info(f"  Loaded {len(messages)} messages from {fpath}")

        # Apply cleaning
        cleaned = [cls.clean_text(msg) for msg in all_messages]
        # Filter out very short texts (< 2 Chinese chars)
        cleaned = [t for t in cleaned if len(t) >= 2
                    and re.search(r'[\u4e00-\u9fa5]', t)]

        logger.info(f"Preprocessing complete: {len(cleaned)} valid messages "
                     f"from {len(all_messages)} raw messages")
        return cleaned


def load_reference_vocabulary(vocab_file: str) -> dict:
    """
    Parse the Yuxi dialect vocabulary compilation (词汇汇编-v2.md)
    as a REFERENCE resource for understanding dialect patterns and
    characteristics.  This is NOT a ground-truth answer key — it is a
    general dialect dictionary used for auxiliary overlap analysis.
    Returns dict: {dialect_word: standard_meaning}
    """
    gt = {}
    try:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # Extract bold terms: **word**: ... meaning
        pattern = r'\*\*(.+?)\*\*[:：].*?[）)](.*?)(?:\n|$)'
        matches = re.findall(pattern, content)
        for word, meaning in matches:
            word = word.strip()
            meaning = meaning.strip()
            if len(word) >= 1 and len(word) <= 6:
                gt[word] = meaning
        logger.info(f"Extracted {len(gt)} dialect terms from vocabulary file")
    except Exception as e:
        logger.warning(f"Failed to parse vocabulary file: {e}")
    return gt


def get_corpus_dialect_anchors(corpus: List[str],
                               reference_vocab: dict,
                               jieba_default_words: set = None) -> List[str]:
    """Extract reference vocabulary terms that appear in the corpus
    and can serve as additional dialect content-word anchors.

    The default YUXI_ANCHORS are dominated by function words (呢/着/咋个/...).
    Adding content words from the reference vocabulary (扎实/老火/鬼火/...)
    enriches bigram extraction with dialect-content + power-word pairings.

    Filters:
    - 2-4 character terms only (longer phrases don't anchor bigrams well)
    - Must appear in the corpus (otherwise they can't anchor anything)
    - Must NOT be in jieba's default dictionary (excludes standard Chinese
      words like 冬瓜, 不得了 that happen to contain dialect-looking chars)
    - Must contain dialect morphology cues (particle chars, content chars,
      or be a known dialect whole-word expression)
    """
    import re
    from config import DIALECT_FULL_WORDS
    # Dialect content characters — distinct from pure function particles
    dialect_content_chars = set('扎实鬼火老冇木挨电着炸')
    dialect_particle_chars = set('着呢嘛噶哈了得过')

    # Build corpus word frequency
    import jieba
    corpus_words = set()
    for msg in corpus:
        corpus_words.update(jieba.lcut(msg))

    extended = []
    ref_words = set(reference_vocab.keys())
    for word in ref_words:
        if not (2 <= len(word) <= 4):
            continue
        if word not in corpus_words:
            continue
        # Must have dialect morphology: particle chars, content chars,
        # or be a known dialect whole-word expression.
        has_dialect = (
            any(c in dialect_particle_chars for c in word)
            or any(c in dialect_content_chars for c in word)
            or word in DIALECT_FULL_WORDS
        )
        if not has_dialect:
            continue
        # Exclude standard Chinese words already in jieba's default dictionary
        # (e.g. 冬瓜, 不得了), unless they are verified dialect expressions.
        if (jieba_default_words and word in jieba_default_words
                and word not in DIALECT_FULL_WORDS):
            continue
        # Exclude pure number/time combos that happen to match
        if re.match(r'^[\d一二三四五六七八九十]+', word):
            continue
        extended.append(word)

    logger.info(
        "Extracted %d dialect content anchors from reference vocabulary "
        "(appearing in corpus)", len(extended))
    return extended
