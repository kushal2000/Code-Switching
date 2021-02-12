# Code-Switching

Implementation of the paper "Code-switching patterns can be an effective route to improve performance of downstream NLP applications: A case study of humour, sarcasm and hate speech detection ~ [Bansal et al.](https://arxiv.org/abs/2005.02295)"

This repository also adds multilingual BERT based models and a Fusion Net based on metric losses to improve upon the baselines.

## Results (Macro-F1 scores)

| Model                     | Humour | Sarcasm | Hate  | Sentiment |
|---------------------------|--------|---------|-------|-----------|
| Baseline                  | 69.34  | 78.40   | 33.60 | 58.24     |
| HAN+features              | 70.91  | 79.05   | 42.89 | 47.91     |
| mBERT+features            | 72.62  | 87.88   | 62.76 | 59.08     |
| mBERT+HAN+features        | 73.28  | 87.62   | 62.85 | 58.24     |
| mBERT+Fusion Net+features | 73.19  | 88.18   | 63.11 | 59.30     |
