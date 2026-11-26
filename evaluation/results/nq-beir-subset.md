## Retrieval benchmark: nq-beir-subset

- Corpus files: 2361 (indexed 2361, errors 0)
- Indexing time: 101.2 s (23.34 files/s)
- Peak resident memory: 555 MB
- Queries per mode set: 300

| Modes | Recall@1 | Recall@5 | Recall@10 | MRR | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| keyword | 0.759 | 0.936 | 0.964 | 0.890 | 6.9 | 9.8 |
| text_semantic | 0.869 | 0.997 | 1.000 | 0.978 | 14.7 | 16.3 |
| keyword + text_semantic | 0.876 | 1.000 | 1.000 | 0.984 | 22.0 | 24.7 |
