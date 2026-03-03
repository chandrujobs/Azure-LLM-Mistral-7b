# Production Accuracy Runbook

## Objective
Raise end-to-end production accuracy using a repeatable evaluation + review loop.

## Current Baseline
- source_hit_rate_at_k: 0.95
- mrr_at_k: 0.807
- avg_keyword_coverage_at_k: 0.5675
- avg_required_fact_coverage_at_k: 0.2166

## Files
- Eval set: `eval_questions.jsonl`
- Eval summary: `output/production_eval_summary.json`
- Eval details: `output/production_eval_details.jsonl`
- Human review pack (high-risk 150): `output/production_review_pack_150.jsonl`

## Production Gates (recommended)
- source_hit_rate_at_k >= 0.94
- mrr_at_k >= 0.80
- avg_keyword_coverage_at_k >= 0.55
- avg_required_fact_coverage_at_k >= 0.20
- forbidden_fact_hit_rate_at_k <= 0.02

## Weekly Loop
1. Run eval:
   `python evaluate_rag.py --questions-file eval_questions.jsonl --out-json output/production_eval_summary.json --out-details-jsonl output/production_eval_details.jsonl`
2. Run gated eval for CI:
   `python evaluate_rag.py --questions-file eval_questions.jsonl --min-source-hit-rate 0.94 --min-mrr 0.80 --min-keyword-coverage 0.55 --min-required-fact-coverage 0.20 --max-forbidden-fact-hit-rate 0.01`
3. Review 150 high-risk rows from `output/production_review_pack_150.jsonl`.
4. For each reviewed row, fix one or more:
   - question wording (clear intent)
   - expected_keywords
   - required_facts
   - forbidden_facts
   - expected_sources
5. Re-run eval and compare metrics.
6. Promote only if all production gates pass.

## Labeling Rules
- required_facts: 3-5 atomic facts, concise and doc-grounded.
- forbidden_facts: 1-2 common hallucination/incorrect claims.
- expected_keywords: 4-6 discriminative terms from source docs.
- comparison questions must include explicit alternatives.
- troubleshooting questions must include concrete failure context.

## Notes
- Prioritize fixing rows where source rank > 2 and required_fact_coverage < 0.2.
- Avoid overly generic prompts (they pull unrelated docs).
- Keep one canonical eval file (`eval_questions.jsonl`).
