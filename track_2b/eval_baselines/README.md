# Baselines

Not part of Clausus and not in its dependencies. Kept so anyone can recompute the comparison.

## Microsoft Presidio

```
python -m venv .venv-presidio && .venv-presidio/Scripts/pip install presidio-analyzer==2.2.364 \
  "de_core_news_lg @ https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl"
.venv-presidio/Scripts/python eval_baselines/presidio_baseline.py data/eval/holdout.json
```

Setup: largest German spaCy model, all predefined recognizers loaded for German, default score
threshold, every detected span masked. Scored with the same `score_case` as Clausus.
