# Experiment Comparison (Phase 6 LLM Evaluation)

| Model | DP | PDP | L_d | L_pd | AvgConf | Ref(Exact) | Ref(Token) | Lexicon | Queries |
|-------|----|-----|-------|--------|---------|------------|------------|---------|---------|
| Margin + Rule (Hard Gate) | 1.0000 | 0.6250 | 8 | 5 | 0.9188 | 0 | 0 | 8 | 5 |
| Margin + Rule (Soft-Label baseline) | 1.0000 | 0.1600 | 25 | 4 | 0.9060 | 0 | 0 | 25 | 15 |
| Margin + RAG-LLM Oracle | 1.0000 | 0.1250 | 32 | 4 | 0.8953 | 0 | 0 | 32 | 15 |
| TD-CAL + Rule Oracle | 1.0000 | 0.1481 | 27 | 4 | 0.9056 | 0 | 0 | 27 | 15 |
| Random + RAG-LLM Oracle | 1.0000 | 0.1212 | 33 | 4 | 0.8976 | 0 | 0 | 33 | 15 |
| TD-CAL + RAG-LLM Oracle (Proposed) | 1.0000 | 0.1250 | 32 | 4 | 0.8984 | 0 | 0 | 32 | 15 |
