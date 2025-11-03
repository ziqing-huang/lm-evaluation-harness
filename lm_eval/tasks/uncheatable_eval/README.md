# Uncheatable Eval Perplexity Tasks

These tasks evaluate language models on the public **Uncheatable Eval** corpora.
Each task measures rolling log-likelihood (word and byte perplexity, plus bits per byte)
over documents scraped after mid-2024 across Wikipedia, GitHub, BBC, arXiv, and AO3.

## Data Location

By default the loader looks for normalized dumps that match `<dataset>_*.jsonl.gz`
under Marin's cache path `raw/uncheatable-eval/latest/` (and the historical
`uncheatable_eval/data/` layout). Plain `.jsonl` and `.json` files are accepted as well.
If your checkout stores the data elsewhere, point the loader at the parent directory with either:

```bash
export UNCHEATABLE_EVAL_DATA_ROOT=/path/to/uncheatable_eval/data
```

or by setting `dataset_kwargs.data_root` in a custom YAML that includes these tasks. You can target any directory containing the gzip-compressed JSONL exports produced by `marin.download.uncheatable_eval`.

## Task Groups

* `uncheatable_eval` – focuses on the seven English/code domains used in the Marin exp1600 sweep.
* `uncheatable_eval_full` – aggregates all currently available Uncheatable Eval dumps.
