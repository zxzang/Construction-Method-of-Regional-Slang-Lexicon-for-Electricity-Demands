# -*- coding: utf-8 -*-
"""
Oracle implementations: Rule-Based and RAG-LLM.
"""
import logging
import json
import os
import re
import time
import numpy as np
from typing import List, Dict, Optional
from config import (DIALECT_MORPHOLOGICAL_CUES, LLM_MODEL, LLM_TEMPERATURE,
                    LLM_MAX_TOKENS, LLM_SEED, LLM_JSON_MODE, RAG_TOP_K,
                    KB_DIR, LLM_API_KEY, LLM_BASE_URL,
                    LLM_FEW_SHOT_EXAMPLES, LLM_ACCEPTANCE_MIN_CONFIDENCE,
                    EVAL_LLM_MODEL, EVAL_LLM_API_KEY, EVAL_LLM_BASE_URL,
                    EVAL_LLM_TEMPERATURE, EVAL_LLM_MAX_TOKENS,
                    EVAL_LLM_SEED, EVAL_LLM_JSON_MODE,
                    DIALECT_FULL_WORDS)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key setup — fill in config.py or set env vars before running.
# Priority: environment variable > config.py value
# ---------------------------------------------------------------------------
if LLM_API_KEY and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY
if LLM_BASE_URL and "OPENAI_BASE_URL" not in os.environ:
    os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL


# ---------------------------------------------------------------------------
# Robust JSON extraction for LLM outputs
# ---------------------------------------------------------------------------

def _extract_json_robust(text: str) -> dict:
    """Multi-strategy JSON extraction for 2-dimension LLM output.

    Strategy 1: Standard json.loads (markdown block → greedy {} →
                trailing-comma cleanup).
    Strategy 2: Field-by-field regex extraction, immune to unescaped
                double quotes inside definition strings (common LLM
                artifact when the model embeds quoted phrases like
                "给有" directly in JSON string values).

    Returns dict with keys: has_dialect_features, power_domain_relevance,
    category, definition, confidence.
    Raises ValueError when no content can be extracted at all.
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM response")

    text = text.strip()

    # ── Strategy 1: Standard json.loads ──
    # 1a. Markdown code block
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # 1b. Greedy outermost {}
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    json_str = json_match.group() if json_match else text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 1c. Trailing comma cleanup
    cleaned = re.sub(r',\s*}', '}', json_str)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 1d. Pre-process: escape unescaped quotes inside the definition field.
    # The LLM often writes Chinese quotes ("好") inside the definition
    # value without escaping them.  Since we know the definition is between
    # "definition": " and ", "confidence": (confidence is always last),
    # we can safely escape all " within that interval.
    def_match = re.search(r'("definition"\s*:\s*")', text)
    conf_match = re.search(r'("\s*,\s*"confidence"\s*:)', text)
    if def_match and conf_match:
        prefix = text[:def_match.end()]
        middle = text[def_match.end():conf_match.start()]
        suffix = text[conf_match.start():]
        # Escape all " in the definition value
        # First normalize any already-escaped quotes, then escape all
        middle_escaped = middle.replace('\\"', '"').replace('"', '\\"')
        cleaned = prefix + middle_escaped + suffix
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # ── Strategy 2: Field-by-field extraction ──
    # Exploits the known JSON structure {bool, bool, str, str, float}
    # where "confidence" is always the last key.
    logger.warning("JSON parse failed, falling back to field-by-field extraction "
                   "(raw: %.120s)", text)
    return {
        'has_dialect_features': _extract_bool_field(text, 'has_dialect_features'),
        'power_domain_relevance': _extract_bool_field(text, 'power_domain_relevance'),
        'category': _extract_cat_field(text),
        'definition': _extract_def_field(text),
        'confidence': _extract_float_field(text, 'confidence'),
    }


def _extract_bool_field(text: str, key: str) -> bool:
    """Extract a boolean field from malformed JSON.  Default False."""
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*(true|false)', text, re.I)
    return m.group(1).lower() == 'true' if m else False


def _extract_float_field(text: str, key: str) -> float:
    """Extract a float field from malformed JSON.  Default 0.5."""
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*([0-9]+\.?[0-9]*)', text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.5


def _extract_cat_field(text: str) -> str:
    """Extract the category field.  Category values never contain quotes
    (they are enum-like: OUTAGE, REPAIR, ...).  Default 'OTHER'."""
    m = re.search(r'"category"\s*:\s*"([^"]*)"', text)
    return m.group(1) if m else 'OTHER'


def _extract_def_field(text: str) -> str:
    """Extract the definition field, robust to unescaped double quotes
    inside the value (the dominant LLM JSON artifact).

    Strategy: find "definition": " and then search for the closing
    marker ", "confidence": which is guaranteed unique because
    "confidence" is always the last key in our prompt schema.
    """
    start_m = re.search(r'"definition"\s*:\s*"', text)
    if not start_m:
        return ''
    start = start_m.end()

    # Search for the closing marker from after the definition start.
    # "confidence" is always the last key, so ", "confidence": is the
    # unique end-of-definition delimiter regardless of embedded quotes.
    tail = text[start:]
    end_m = re.search(r'"\s*,\s*"confidence"\s*:', tail)
    if end_m:
        definition = tail[:end_m.start()]
        # Unescape any escaped quotes the model DID produce correctly
        definition = definition.replace('\\"', '"')
        return definition

    # Truncated response — "confidence" not present.
    # Try closing "} at the end.
    end_brace = re.search(r'"\s*\}', tail)
    if end_brace:
        definition = tail[:end_brace.start()]
        return definition.replace('\\"', '"')

    # Truncated mid-definition — return whatever we have, stripping trailing
    # whitespace and dangling characters.
    return tail.rstrip().rstrip('"').rstrip()


class RuleBasedOracle:
    """Simulates expert via dialect morphological pattern matching.

    Uses three pattern types:
    - full_words: candidate contains a known dialect word as substring
    - suffixes: candidate ends with a Yuxi dialect suffix, stem >= 2 chars,
      AND word contains power-domain content characters
    - prefixes: candidate starts with a Yuxi dialect prefix (and has content after)
    """

    # Power-domain content characters — suffix matches also require the word
    # to contain at least one of these, to distinguish dialect particles
    # attached to power terms ("来电呢") from standard vocab + particle ("详细呢").
    _POWER_CHARS = set('电线路闸表费火跳停修炸漏烧压杆变容开关供配')

    def __init__(self, cues: dict = None):
        self.cues = cues or DIALECT_MORPHOLOGICAL_CUES
        self.query_count = 0

    def judge(self, candidate: dict) -> dict:
        """Return candidate with 'oracle_label' (True/False) and 'oracle_reason'."""
        self.query_count += 1
        word = candidate['word']
        reasons = []

        # 1. Full-word match: candidate contains a known dialect word
        for fw in self.cues.get('full_words', []):
            if fw in word:
                reasons.append(f'full:{fw}')
                break  # one full-word match is enough

        # 2. Suffix pattern: word ends with dialect suffix, stem has >= 2 chars,
        #    AND word contains power-domain content (excludes "详细呢" etc.)
        if not reasons:
            for sfx in self.cues.get('suffixes', []):
                if word.endswith(sfx) and len(word) >= 3:
                    stem = word[:-len(sfx)]
                    if len(stem) >= 2 and any(
                            c in self._POWER_CHARS for c in word):
                        reasons.append(f'sfx:{sfx}')
                        break

        # 3. Prefix pattern: word starts with dialect prefix, has content after
        if not reasons:
            for pfx in self.cues.get('prefixes', []):
                if word.startswith(pfx) and len(word) > len(pfx) + 1:
                    reasons.append(f'pfx:{pfx}')
                    break

        is_valid = len(reasons) > 0
        return {**candidate,
                'oracle_label': is_valid,
                'oracle_reason': '+'.join(reasons) if is_valid else 'no_pattern_match'}

    def judge_batch(self, candidates: List[dict]) -> List[dict]:
        return [self.judge(c) for c in candidates]


class RAGLLMOracle:
    """
    RAG-Augmented LLM Oracle.
    Retrieves power-domain KB passages, prompts LLM for definition,
    then scores by SBERT similarity.

    When OPENAI_API_KEY is not set, falls back to MOCK MODE that uses the
    ground-truth dictionary directly. Results from mock mode are NOT valid
    for paper claims — they assume a perfect oracle. Set OPENAI_API_KEY
    (and optionally OPENAI_BASE_URL) to use a real LLM.
    """

    def __init__(self, kb_dir: str = None, sbert_model=None,
                 ground_truth: dict = None, mock: bool = False):
        self.kb_dir = kb_dir or KB_DIR
        self.query_count = 0
        self.sbert = sbert_model
        # 'ground_truth' kept as parameter name for backward compat;
        # it receives reference_vocab — a reference resource, NOT an answer key.
        self.ground_truth = ground_truth
        self.kb_entries: List[dict] = []
        self.kb_embeddings: Optional[np.ndarray] = None
        self._mock_mode = mock
        self._load_knowledge_base()

        if self._mock_mode:
            if ground_truth is None:
                raise ValueError(
                    "RAGLLMOracle: mock=True requires ground_truth (reference_vocab).")
            logger.warning(
                "RAGLLMOracle: running in MOCK MODE (ground-truth lookup). "
                "Results are NOT valid for paper claims.")
        else:
            if "OPENAI_API_KEY" not in os.environ:
                raise RuntimeError(
                    "RAGLLMOracle: OPENAI_API_KEY not set and mock=False. "
                    "Set the environment variable OPENAI_API_KEY to use a real "
                    "LLM oracle, or pass mock=True for development/testing only.")

    def _load_knowledge_base(self):
        """Load or create the power-domain knowledge base.

        A cached JSON file avoids rebuilding on every run.  The content hash of
        the code version (``_build_default_kb()``) is stored alongside the
        entries.  On load, the stored hash is compared against the current code:

        * **match** — code unchanged; load the cached file as-is (default path).
        * **mismatch** — code was updated (or the JSON was manually edited);
          regenerate from code, overwriting the JSON file.  Manual edits to
          ``power_domain_kb.json`` are not supported — edit
          ``_build_default_kb()`` instead.
        * **old format** — plain list without a hash key; migrate to the new
          format automatically.
        """
        kb_file = os.path.join(self.kb_dir, 'power_domain_kb.json')
        code_entries = self._build_default_kb()
        import hashlib
        code_hash = hashlib.sha256(
            json.dumps(code_entries, ensure_ascii=False, sort_keys=True)
            .encode('utf-8')).hexdigest()

        if os.path.exists(kb_file):
            with open(kb_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if isinstance(cached, dict) and '_content_hash' in cached:
                cached_hash = cached['_content_hash']
                self.kb_entries = cached['entries']
            else:
                logger.info("KB file is in old format (no content hash) — "
                            "migrating to hash-tracked format.")
                cached_hash = None
                self.kb_entries = cached

            if cached_hash != code_hash:
                if cached_hash is not None:
                    logger.warning(
                        "KB content hash mismatch — the cached JSON was "
                        "manually edited or _build_default_kb() changed.  "
                        "Regenerating from code (manual edits to "
                        "power_domain_kb.json will be overwritten).")
                self.kb_entries = code_entries
                os.makedirs(self.kb_dir, exist_ok=True)
                with open(kb_file, 'w', encoding='utf-8') as f:
                    json.dump({'_content_hash': code_hash,
                               'entries': code_entries},
                              f, ensure_ascii=False, indent=2)
        else:
            self.kb_entries = code_entries
            os.makedirs(self.kb_dir, exist_ok=True)
            with open(kb_file, 'w', encoding='utf-8') as f:
                json.dump({'_content_hash': code_hash,
                           'entries': code_entries},
                          f, ensure_ascii=False, indent=2)
        logger.info(f"KB loaded: {len(self.kb_entries)} entries "
                    f"(hash={code_hash[:8]}...)")

        if self.sbert and self.kb_entries:
            texts = [e['text'] for e in self.kb_entries]
            self.kb_embeddings = self.sbert.encode(texts, normalize_embeddings=True)

    @staticmethod
    def _build_default_kb() -> List[dict]:
        """Build default power-domain KB from domain knowledge."""
        entries = [
            {"id": "outage_01", "text": "停电是指电力供应中断，用户无法正常用电的状态",
             "category": "OUTAGE"},
            {"id": "outage_02", "text": "断电、没电、黑了都表示供电中断",
             "category": "OUTAGE"},
            {"id": "outage_03", "text": "计划停电是供电局提前通知的检修停电",
             "category": "OUTAGE"},
            {"id": "repair_01", "text": "报修是用户向供电局报告电力故障请求维修",
             "category": "REPAIR"},
            {"id": "repair_02", "text": "抢修是供电局紧急处理电力故障的行为",
             "category": "REPAIR"},
            {"id": "equip_01", "text": "变压器是改变电压等级的电力设备",
             "category": "EQUIPMENT"},
            {"id": "equip_02", "text": "电表是计量用电量的仪表装置",
             "category": "EQUIPMENT"},
            {"id": "equip_03", "text": "空气开关是一种低压断路器保护装置",
             "category": "EQUIPMENT"},
            {"id": "fault_01", "text": "跳闸是断路器自动断开电路的保护动作",
             "category": "FAULT"},
            {"id": "fault_02", "text": "短路是电路中电流不经过负载直接连通",
             "category": "FAULT"},
            {"id": "fault_03", "text": "漏电是电流通过非正常路径流入大地",
             "category": "FAULT"},
            {"id": "fault_04", "text": "电线老化破损可能导致漏电或短路事故",
             "category": "FAULT"},
            {"id": "bill_01", "text": "电费是用户按用电量支付的费用",
             "category": "BILLING"},
            {"id": "bill_02", "text": "欠费停电是因未缴纳电费而被停止供电",
             "category": "BILLING"},
            # ── Yuxi dialect words (vocabulary) ──
            {"id": "dialect_01", "text": "扎实在玉溪方言中表示程度严重非常厉害，例如扎实老火表示非常严重",
             "category": "DIALECT"},
            {"id": "dialect_02", "text": "鬼火在玉溪方言中表示愤怒烦躁发火，鬼火绿表示非常生气",
             "category": "DIALECT"},
            {"id": "dialect_03", "text": "给有在玉溪方言中是疑问句表示是否有，例如给有电表示有没有电",
             "category": "DIALECT"},
            {"id": "dialect_04", "text": "着炸在玉溪方言中表示被损坏短路烧毁，着电表示触电",
             "category": "DIALECT"},
            {"id": "dialect_05", "text": "木电电在玉溪方言中表示完全没有电了，木表示没有",
             "category": "DIALECT"},
            {"id": "dialect_06", "text": "咋个在玉溪方言中表示怎么为什么，例如咋个停电表示怎么停电了",
             "category": "DIALECT"},
            {"id": "dialect_07", "text": "老火在玉溪方言中表示严重、糟糕、厉害，例如线路老火表示线路故障严重",
             "category": "DIALECT"},
            {"id": "dialect_08", "text": "不来电呢在玉溪方言中表示没有来电，呢是语气助词",
             "category": "DIALECT"},
            {"id": "dialect_09", "text": "黑古隆冬在玉溪方言中形容一片漆黑、停电后的黑暗状态",
             "category": "DIALECT"},
            {"id": "dialect_10", "text": "不鬼火在玉溪方言中表示不生气、算了、没事了",
             "category": "DIALECT"},
            {"id": "dialect_11", "text": "冇在玉溪方言中表示没有，例如冇交表示没有交费，冇住表示没有住",
             "category": "DIALECT"},
            {"id": "dialect_12", "text": "挨在玉溪方言中表示把、被、和，例如挨电费交了表示把电费交了",
             "category": "DIALECT"},
            # ── Dialect morphology (fragment vs. word distinction) ──
            {"id": "morph_01", "text": "玉溪方言中「咋个」+X组合如「咋个整」「咋个怎」是疑问句式结构而非独立的方言词汇，不应收录",
             "category": "DIALECT"},
            {"id": "morph_02", "text": "玉溪方言中「X呢」组合如「烂呢」「摘呢」通常只是名词加语气助词，不构成独立的方言词汇",
             "category": "DIALECT"},
            {"id": "morph_03", "text": "玉溪方言中「给有」+X组合如「给有看看」「给有回」是疑问句式，不是独立词汇",
             "category": "DIALECT"},
            {"id": "morph_04", "text": "玉溪方言中「X噶」「X嘛」「X哈」结尾的组合是语气词附加，通常不是独立方言词汇",
             "category": "DIALECT"},
        ]
        return entries

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[dict]:
        """Retrieve top-k KB entries most similar to query."""
        if self.sbert is None or self.kb_embeddings is None:
            return self.kb_entries[:top_k]
        q_emb = self.sbert.encode([query], normalize_embeddings=True)
        scores = np.dot(self.kb_embeddings, q_emb.T).flatten()
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [self.kb_entries[i] for i in top_idx]

    def _build_prompt(self, candidate_word: str, context_entries: List[dict]) -> str:
        """Build the 2-dimension LLM prompt with few-shot examples."""
        kb_text = "\n".join(
            "- {} (类别: {})".format(e['text'], e['category'])
            for e in context_entries)

        # Format few-shot examples
        examples = []
        for ex in LLM_FEW_SHOT_EXAMPLES:
            examples.append(
                "「{}」→ {{\n"
                '  "has_dialect_features": {},\n'
                '  "power_domain_relevance": {},\n'
                '  "category": "{}",\n'
                '  "definition": "{}",\n'
                '  "confidence": {}\n'
                "}}".format(
                    ex['word'],
                    str(ex['has_dialect_features']).lower(),
                    str(ex['power_domain_relevance']).lower(),
                    ex['category'],
                    ex['definition'],
                    ex['confidence'],
                ))
        examples_text = "\n".join(examples)

        prompt = """你是一位云南玉溪方言和电力服务领域专家。

## 背景知识
{}

## 任务
对词语「{}」进行两个维度的独立判断：

1. **has_dialect_features**: 是否包含玉溪方言特有的形态特征？
   - true: 使用了方言语气词（呢/噶/嘛/哈/着）或方言词汇（咋个/给有/冇/木/挨/扎实/鬼火/老火）等
   - false: 纯标准汉语表达
2. **power_domain_relevance**: 是否与电力服务场景相关？
   - true: 涉及停电、报修、电费、设备、故障等电力业务
   - false: 与电力服务无关

## 判断指南
- 两个维度独立判断，一个表达可以同时满足两个维度
- "着"在玉溪方言中有两种角色：被动/遭受标记（着电=触电）vs. 体标记（拿着=拿着）
- 方言语气词附加在电力术语上构成方言特征（如"停电哈"、"给有停电"）

## 参考示例
{}

## 输出格式
严格按JSON输出。注意：definition 字符串中的双引号必须转义为 \"，例如 "触电" 应写为 \"触电\"。
{{"has_dialect_features": true/false, "power_domain_relevance": true/false, "category": "OUTAGE|REPAIR|EQUIPMENT|FAULT|BILLING|OTHER|NOT_POWER", "definition": "简短定义", "confidence": 0.0~1.0}}"""

        return prompt.format(kb_text, candidate_word, examples_text)

    def judge(self, candidate: dict) -> dict:
        """Query the RAG-LLM oracle for a single candidate.

        Uses 2-dimension assessment:
        - (has_dialect_features OR power_domain_relevance)
        - confidence >= LLM_ACCEPTANCE_MIN_CONFIDENCE
        """
        self.query_count += 1
        word = candidate['word']

        # Mock mode: use ground truth directly (development/testing only)
        if self._mock_mode:
            return self._mock_judge_with_gt(candidate)

        context = self.retrieve(word)
        prompt = self._build_prompt(word, context)

        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                result = self._call_llm(prompt)
                has_dialect = result.get('has_dialect_features', False)
                is_power = result.get('power_domain_relevance', False)
                confidence = result.get('confidence', 0.5)

                is_valid = (
                    (has_dialect or is_power)
                    and confidence >= LLM_ACCEPTANCE_MIN_CONFIDENCE
                )

                category = result.get('category', candidate.get('category', 'OTHER'))
                definition = result.get('definition', '')

                return {**candidate,
                        'oracle_label': is_valid,
                        'oracle_reason': definition,
                        'oracle_category': category,
                        'oracle_has_dialect': has_dialect,
                        'oracle_power_relevant': is_power,
                        'oracle_confidence': confidence}
            except (ValueError, ConnectionError, OSError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"LLM call attempt {attempt + 1}/{max_retries} failed "
                        f"for '{word}': {e} — retrying in {wait}s")
                    time.sleep(wait)
            except ImportError:
                raise  # missing openai package — retry won't help
        raise RuntimeError(
            f"RAGLLMOracle: all {max_retries} LLM attempts failed for "
            f"'{word}': {last_error}. "
            f"Check API connectivity and retry.") from last_error

    def _mock_judge_with_gt(self, candidate: dict) -> dict:
        """Mock LLM judge using the ground truth dictionary.

        If the candidate is found in ground_truth, label it as accepted.
        Otherwise fall back to character-level heuristic rules (power-domain
        and dialect character matching) for a rough assessment.
        """
        word = candidate['word']
        if word in self.ground_truth:
            return {**candidate,
                    'oracle_label': True,
                    'oracle_reason': "mock_llm_gt_lookup",
                    'oracle_category': "DIALECT"}
        # Fall back to heuristic for terms not in the reference vocabulary.
        heuristic = self._heuristic_judge(word)
        return {**candidate,
                'oracle_label': (
                    (heuristic.get('has_dialect_features', False)
                     or heuristic.get('power_domain_relevance', False))
                    and heuristic.get('confidence', 0.5) >= LLM_ACCEPTANCE_MIN_CONFIDENCE
                ),
                'oracle_reason': f"mock_heuristic:{heuristic.get('definition','')}",
                'oracle_category': heuristic.get('category', 'OTHER')}

    def _call_llm(self, prompt: str) -> dict:
        """Call the LLM API. Uses openai-compatible interface."""
        try:
            from openai import OpenAI
            client = OpenAI()
            kwargs = dict(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                seed=LLM_SEED,
            )
            if LLM_JSON_MODE:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content.strip()
            return _extract_json_robust(text)
        except ValueError:
            # Empty or unparseable response — let judge() fallback handle it
            raise
        except ImportError:
            raise RuntimeError(
                "RAGLLMOracle: 'openai' package is required. "
                "Install it with: pip install openai") from None

    def _heuristic_judge(self, prompt_or_word: str) -> dict:
        """Heuristic fallback when LLM API is unavailable.

        Accepts either a full LLM prompt (containing 「word」 brackets) or a
        plain word string.  When a prompt is passed, the word is extracted
        from the bracketed portion; otherwise the input is used as-is.
        """
        match = re.search(r'「(.+?)」', prompt_or_word)
        word = match.group(1) if match else prompt_or_word
        power_chars = set('电线闸表坏黑费火跳停修炸漏烧压杆变容开关')
        dialect_chars = set('扎实鬼给有着咋冇木噶挨')
        has_power = any(c in word for c in power_chars)
        has_dialect = any(c in word for c in dialect_chars)
        cat = "FAULT" if has_power else ("DIALECT" if has_dialect else "OTHER")
        return {
            "has_dialect_features": has_dialect,
            "power_domain_relevance": has_power,
            "category": cat,
            "definition": f"heuristic: power={has_power}, dialect={has_dialect}",
            "confidence": 0.5,
        }

    def judge_batch(self, candidates: List[dict]) -> List[dict]:
        return [self.judge(c) for c in candidates]


class EvaluationOracle:
    """Independent LLM evaluator using high-capability model.

    Evaluates ALL lexicon terms on 3 dimensions WITHOUT acceptance gating.
    This is purely for evaluation — it does NOT filter or select candidates.
    Uses a higher-capability model (EVAL_LLM_MODEL, default deepseek-v4-pro)
    for maximum assessment quality, separate from the selection oracle.

    Requires OPENAI_API_KEY or EVAL_LLM_API_KEY to be set in the environment.
    Heuristic fallback is NOT used — the evaluator raises on missing keys
    or API failures so that paper metrics are never produced from unreliable
    proxies.
    """

    def __init__(self, kb_entries: List[dict] = None):
        self.query_count = 0
        self.kb_entries = kb_entries or []
        has_key = (bool(EVAL_LLM_API_KEY)
                   or "OPENAI_API_KEY" in os.environ
                   or "EVAL_LLM_API_KEY" in os.environ)
        if not has_key:
            raise RuntimeError(
                "EvaluationOracle: neither EVAL_LLM_API_KEY nor "
                "OPENAI_API_KEY is set.  Set one of these environment "
                "variables to use the LLM evaluator.")

    def evaluate(self, candidates: List[dict]) -> List[dict]:
        """Evaluate all candidates on 2 dimensions (has_dialect_features,
        power_domain_relevance). No acceptance gating."""
        results = []
        for c in candidates:
            results.append(self._evaluate_one(c))
        return results

    def _evaluate_one(self, candidate: dict) -> dict:
        self.query_count += 1
        word = candidate['word']

        prompt = self._build_eval_prompt(word)
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                result = self._call_llm(prompt)
                return {
                    'word': word,
                    'eval_has_dialect': result.get('has_dialect_features', False),
                    'eval_power_relevant': result.get('power_domain_relevance', False),
                    'eval_category': result.get('category', 'OTHER'),
                    'eval_definition': result.get('definition', ''),
                    'eval_confidence': result.get('confidence', 0.5),
                }
            except (ValueError, ConnectionError, OSError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"Eval attempt {attempt + 1}/{max_retries} failed "
                        f"for '{word}': {e} — retrying in {wait}s")
                    time.sleep(wait)
        raise RuntimeError(
            f"EvaluationOracle: all {max_retries} attempts failed for "
            f"'{word}': {last_error}") from last_error

    def _build_eval_prompt(self, candidate_word: str) -> str:
        """Build the 2-dimension evaluation prompt with domain context."""
        kb_text = "\n".join(
            "- {} (类别: {})".format(e['text'], e['category'])
            for e in self.kb_entries
        ) if self.kb_entries else "（无背景知识条目）"

        examples = []
        for ex in LLM_FEW_SHOT_EXAMPLES:
            examples.append(
                "「{}」→ {{\n"
                '  "has_dialect_features": {},\n'
                '  "power_domain_relevance": {},\n'
                '  "category": "{}",\n'
                '  "definition": "{}",\n'
                '  "confidence": {}\n'
                "}}".format(
                    ex['word'],
                    str(ex['has_dialect_features']).lower(),
                    str(ex['power_domain_relevance']).lower(),
                    ex['category'],
                    ex['definition'],
                    ex['confidence'],
                ))
        examples_text = "\n".join(examples)

        prompt = """你是一位云南玉溪方言和电力服务领域专家。

## 背景知识
{}

## 任务
请对词语「{}」进行两个维度的独立评估：

1. **has_dialect_features**: 是否包含玉溪方言特有的形态特征？
   - true: 使用了方言语气词（呢/噶/嘛/哈/着）或方言词汇（咋个/给有/冇/木/挨/扎实/鬼火/老火）等
   - false: 纯标准汉语表达
2. **power_domain_relevance**: 是否与电力服务场景相关？
   - true: 涉及停电、报修、电费、设备、故障等电力业务
   - false: 与电力服务无关

## 判断指南
- 两个维度独立判断，一个表达可以同时满足两个维度
- "着"在玉溪方言中有两种角色：被动/遭受标记（着电=触电）vs. 体标记（拿着=拿着）
- 方言语气词附加在电力术语上构成方言特征（如"停电哈"、"给有停电"）

## 参考示例
{}

## 输出格式
严格按JSON输出。注意：definition 字符串中的双引号必须转义为 \"，例如 "触电" 应写为 \"触电\"。
{{"has_dialect_features": true/false, "power_domain_relevance": true/false, "category": "OUTAGE|REPAIR|EQUIPMENT|FAULT|BILLING|OTHER|NOT_POWER", "definition": "简短定义", "confidence": 0.0~1.0}}"""

        return prompt.format(kb_text, candidate_word, examples_text)

    def _call_llm(self, prompt: str) -> dict:
        """Call the LLM API with robust JSON extraction."""
        try:
            from openai import OpenAI
            eval_key = (EVAL_LLM_API_KEY
                        or os.environ.get("EVAL_LLM_API_KEY")
                        or os.environ.get("OPENAI_API_KEY"))
            eval_url = (EVAL_LLM_BASE_URL
                        or os.environ.get("EVAL_LLM_BASE_URL")
                        or os.environ.get("OPENAI_BASE_URL"))
            client = OpenAI(
                api_key=eval_key,
                base_url=eval_url,
            )
            kwargs = dict(
                model=EVAL_LLM_MODEL,
                temperature=EVAL_LLM_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=EVAL_LLM_MAX_TOKENS,
                seed=EVAL_LLM_SEED,
            )
            if EVAL_LLM_JSON_MODE:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content.strip()
            return _extract_json_robust(text)
        except ValueError:
            raise
        except ImportError:
            raise RuntimeError(
                "EvaluationOracle: 'openai' package is required. "
                "Install it with: pip install openai") from None
