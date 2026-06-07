# -*- coding: utf-8 -*-
"""
Global configuration for the Regional Slang Lexicon experiments.
All hyperparameters, paths, and seed words are centralized here.
"""
import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILES = [
    os.path.join(PROJECT_ROOT, '微信群聊天数据1_脱敏.csv'),
    os.path.join(PROJECT_ROOT, '微信群聊天数据2_脱敏.csv'),
]
VOCAB_FILE = os.path.join(PROJECT_ROOT, '词汇汇编-v2.md')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')
KB_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'knowledge_base')

# ============================================================
# Random Seed
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Yuxi Dialect Anchors (方言锚点)
# ============================================================
YUXI_ANCHORS = [
    '呢', '挨', '咋个', '给有', '是呢', '扎实', '鬼火', '噶',
    '木', '冇', '咋整', '着', '嘛', '哈', '得行', '老火',
]

# ============================================================
# Power Domain Seed Words (电力领域种子词)
# ============================================================
POWER_SEEDS = {
    'OUTAGE': ['停电', '没电', '黑了', '断电', '没得电', '黑古隆冬', '木电',
               '来电', '停过', '停了', '没来电', '一哈停', '停好几次',
               '经常停', '一早就停'],
    'REPAIR': ['报修', '抢修', '维修', '快来人', '处理哈', '联系下', '修理',
               '修好', '来修', '检查', '检修', '帮看看', '帮瞧'],
    'EQUIPMENT': ['变压器', '电线', '电表', '空气开关', '闸刀', '保险', '电杆',
                  '线路', '开关', '表箱', '杆子', '电闸'],
    'FAULT': ['跳闸', '短路', '漏电', '烧坏', '炸掉', '着火', '冒烟',
              '烧了', '炸了', '冒火', '漏了', '打火', '着炸', '坏掉'],
    'BILLING': ['电费', '缴费', '欠费', '度数', '抄表', '充值',
                '交费', '收钱', '多收', '退费', '收多'],
}

# Flat list of all seed words
ALL_SEEDS = []
for words in POWER_SEEDS.values():
    ALL_SEEDS.extend(words)

# ============================================================
# PMI / NPMI Thresholds
# ============================================================
PMI_THRESHOLD = 1.5
# ── DEPRECATED: NPMI mode is no longer used in current experiments.
# All configurations use use_npmi=False.  Kept for historical reference
# and possible future comparison; remove if permanently abandoned. ──
NPMI_THRESHOLD = 0.12
MIN_BIGRAM_FREQ = 2      # Minimum co-occurrence count
EDIT_DISTANCE_MAX = 2    # Max edit distance for typo filtering
# DEPRECATED: only used in NPMI mode

# ============================================================
# SBERT Configuration
# ============================================================
SBERT_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

# ============================================================
# Active Learning Configuration
# ============================================================
# Similarity thresholds for auto-accept / uncertain / reject
# Multilingual MiniLM tends to give high baseline cosine similarities.
SIM_HIGH = 0.78             # Auto-accept threshold (raised from 0.75:
                            # creates ~5 uncertain candidates for AL
                            # differentiation with current score distribution)
SIM_LOW = 0.45              # Below this -> reject as noise (raised from 0.35)
SIM_BOUNDARY = 0.60         # Decision boundary for margin sampling (raised from 0.45)

# TD-CAL specific
TDCAL_TOP_K = 5             # Number of uncertain candidates to query per round
                              # Lowered from 30: creates multi-round AL when
                              # uncertain pool > TOP_K, enabling strategy comparison
AL_MAX_ROUNDS = 3           # Maximum active learning iterations.
                              # Reduced from 10: early stopping at 3 rounds
                              # prevents full uncertain-pool exhaustion and
                              # enables AL strategy differentiation when
                              # uncertain pool > TOP_K × MAX_ROUNDS.

# ============================================================
# RAG-LLM Oracle Configuration
# ============================================================
# Fill in your API key below, or set the environment variable OPENAI_API_KEY.
# For non-OpenAI providers (DeepSeek, etc.), also set OPENAI_BASE_URL.
LLM_API_KEY = ""             # Set via environment variable OPENAI_API_KEY
LLM_BASE_URL = "https://api.deepseek.com"            # <-- FILL ME (optional): e.g. "https://api.deepseek.com"
LLM_MODEL = os.environ.get('LLM_MODEL', 'deepseek-v4-flash')  # Model name for the RAG-LLM oracle; override via env
LLM_TEMPERATURE = 0.0           # 0 = deterministic (required for reproducible experiments) 0.0-0.3
LLM_MAX_TOKENS = 800            # Increased from 400: eliminates truncation
                                # warnings on longer definitions
LLM_SEED = 42                   # Fixed seed for reproducibility
LLM_JSON_MODE = True            # Enforce JSON output (set False if provider doesn't support it)
RAG_TOP_K = 5                   # Number of KB passages to retrieve
LLM_ACCEPTANCE_MIN_CONFIDENCE = 0.6  # Minimum confidence for oracle acceptance

# Few-shot examples for the 2-dimension LLM oracle prompt.
# Each entry: (word, has_dialect, power_relevant, category, definition, confidence)
LLM_FEW_SHOT_EXAMPLES = [
    {
        'word': '不来电呢',
        'has_dialect_features': True,
        'power_domain_relevance': True,
        'category': 'OUTAGE',
        'definition': '玉溪方言固定表达，表示没有来电、供电中断，带有方言语气词"呢"',
        'confidence': 0.95,
    },
    {
        'word': '着电',
        'has_dialect_features': True,
        'power_domain_relevance': True,
        'category': 'FAULT',
        'definition': '玉溪方言固定表达，表示触电或被电到，带有方言被动标记"着"',
        'confidence': 0.90,
    },
    {
        'word': '线路老火',
        'has_dialect_features': True,
        'power_domain_relevance': True,
        'category': 'FAULT',
        'definition': '电力+方言复合表达，表示线路故障严重，"老火"为玉溪方言程度副词',
        'confidence': 0.85,
    },
    {
        'word': '咋个整',
        'has_dialect_features': True,
        'power_domain_relevance': False,
        'category': 'NOT_POWER',
        'definition': '玉溪方言疑问句式"咋个"+动词"整"，表示"怎么办"，为方言疑问结构',
        'confidence': 0.90,
    },
    {
        'word': '好呢',
        'has_dialect_features': True,
        'power_domain_relevance': False,
        'category': 'NOT_POWER',
        'definition': '形容词"好"+语气词"呢"，表示"好的/行"，带有方言语气特征',
        'confidence': 0.85,
    },
    {
        'word': '拿着',
        'has_dialect_features': False,
        'power_domain_relevance': False,
        'category': 'NOT_POWER',
        'definition': '标准汉语动词"拿"+体标记"着"，没有方言特征',
        'confidence': 0.95,
    },
]

# ============================================================
# Rule-based Oracle Configuration
# ============================================================
# Dialect morphological patterns for the rule-based oracle.
# Replaces the old single-character DIALECT_FULL_WORDS which matched standard
# Chinese words (供电, 电网) but missed dialect terms.
DIALECT_MORPHOLOGICAL_CUES = {
    # Complete dialect words — candidate contains any of these as substring
    'full_words': [
        '扎实', '鬼火', '老火', '冇', '木', '得行', '给有', '咋个',
        '不来电', '木电电', '黑古隆冬', '黑咕隆咚', '黑古笼冬', '鬼火绿', '咋整',
        '给有电', '不鬼火', '停电哈', '着电', '老火了', '鬼火呢',  
    ],
    # Yuxi dialect suffixes — candidate ends with one of these
    'suffixes': ['呢', '噶', '嘛', '哈', '着'],
    # Yuxi dialect prefixes — candidate starts with one of these
    'prefixes': ['咋个', '给有', '挨', '冇', '木'],
}

# Flat list of known dialect words (used by RuleBasedOracle fallback and oracle.py imports)
DIALECT_FULL_WORDS = DIALECT_MORPHOLOGICAL_CUES['full_words']

# ============================================================
# Fragment Filter Configuration
# ============================================================
# Pure function-word combos — these character sequences are always
# grammatical particles chained together, never lexical items.
DIALECT_PURE_PARTICLE_COMBOS = {
    '呢嘛', '了嘛', '得嘛', '了噶', '呢么', '呢呀', '呢安', '呢嗨',
    '达哈',
}

# Characters that serve as dialect particles / aspect markers.
# Used by the fragment filter and wordhood scorer to detect
# grammatical constructions vs. fixed expressions.
DIALECT_PARTICLE_CHARS = set('着呢嘛噶哈了得过')

# ============================================================
# Wordhood Scoring Configuration (for SBERT combined score)
# ============================================================
WORDHOOD_ALPHA = 0.60  # Weight for SBERT similarity in combined score
WORDHOOD_BETA = 0.40   # Weight for wordhood signal in combined score

# ============================================================
# Independent LLM Evaluator Configuration (Phase 6)
# ============================================================
# Higher-capability model for independent evaluation, separate from the
# selection oracle (which uses LLM_MODEL).  This decouples evaluation from
# the selection pipeline so that metrics are not self-fulfilling.
EVAL_LLM_MODEL = os.environ.get('EVAL_LLM_MODEL', 'deepseek-v4-pro')  # Evaluator; override via env
# For cross-model evaluation, set via environment variables:
#   EVAL_LLM_MODEL='glm-5' EVAL_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
#   EVAL_LLM_MODEL='gpt-4o' EVAL_LLM_BASE_URL=https://api.openai.com/v1
EVAL_LLM_API_KEY = ""   # Set via environment variable OPENAI_API_KEY
EVAL_LLM_BASE_URL = os.environ.get('EVAL_LLM_BASE_URL', 'https://api.deepseek.com')

EVAL_LLM_TEMPERATURE = 0.0
EVAL_LLM_MAX_TOKENS = 800      # Increased from 400: eliminates truncation warnings
EVAL_LLM_SEED = 42
EVAL_LLM_JSON_MODE = True

# Human annotation sampling
EVAL_BOUNDARY_SAMPLE = 50   # Number of boundary terms to sample
EVAL_RANDOM_SAMPLE = 50     # Number of random terms to sample
EVAL_BOUNDARY_MARGIN = 0.10 # Margin threshold for boundary detection

# ============================================================
# Experiment Configurations (5 main + 1 ablation)
# ============================================================
EXPERIMENTS = {
    # All configurations use PMI (not NPMI): the bidirectional anchoring
    # filter already removes power-irrelevant bigrams, and PMI avoids the
    # jieba-tokenization and typo-filter artifacts that affect NPMI mode.
    'baseline_margin_rule': {
        'name': 'Margin + Rule (Soft-Label baseline)',
        'al_strategy': 'margin',
        'oracle_type': 'rule',
        'use_npmi': False,
        'description': 'Baseline: PMI + SBERT + Margin + Rule (soft-label, default)',
    },
    'ablation_margin_llm': {
        'name': 'Margin + RAG-LLM Oracle',
        'al_strategy': 'margin',
        'oracle_type': 'rag_llm',
        'use_npmi': False,
        'description': 'Ablation: isolates RAG-LLM Oracle (vs baseline rule oracle)',
    },
    'ablation_tdcal_rule': {
        'name': 'TD-CAL + Rule Oracle',
        'al_strategy': 'tdcal',
        'oracle_type': 'rule',
        'use_npmi': False,
        'description': 'Ablation: isolates TD-CAL (vs baseline margin)',
    },
    'random_llm': {
        'name': 'Random + RAG-LLM Oracle',
        'al_strategy': 'random',
        'oracle_type': 'rag_llm',
        'use_npmi': False,
        'description': 'Lower bound for active learning (random selection)',
    },
    'proposed_tdcal_llm': {
        'name': 'TD-CAL + RAG-LLM Oracle (Proposed)',
        'al_strategy': 'tdcal',
        'oracle_type': 'rag_llm',
        'use_npmi': False,
        'description': 'Full proposed: PMI + SBERT+wordhood + TD-CAL + RAG-LLM',
    },

    # ── Ablation: soft-label vs hard power gate ──
    # Soft-label (baseline_margin_rule): all dialect-anchored bigrams kept,
    #   power_related tag stored as metadata, relevance deferred downstream.
    # Hard gate (ablation_hard_gate_rule): candidates without power-domain
    #   characters/terms are discarded at extraction (power_related only).
    'ablation_hard_gate_rule': {
        'name': 'Margin + Rule (Hard Gate)',
        'al_strategy': 'margin',
        'oracle_type': 'rule',
        'use_npmi': False,
        'hard_power_gate': True,
        'description': 'Ablation: hard power-gate (power_related only; vs soft-label baseline)',
    },
}
