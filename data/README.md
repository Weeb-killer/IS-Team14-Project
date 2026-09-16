# Dataset directory

Place the local SR-BH 2020 CSV in this directory. The default configuration first looks for:

```text
data/data_capec_multilabel.csv
```

If that exact path does not exist, it searches for a unique CSV anywhere below `data/`.

Dataset files in this directory are intentionally ignored by Git. Use `source.mode: remote` in `configs/dataset.srbh2020.yaml` to download and cache the dataset automatically, or use `source.mode: local` to read a manually downloaded CSV.
