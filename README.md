# NERNETS

Several intriguing networks specialized in named entity recognition. This repository is designed for doing named
entity recognition quickly with a bunch of models. An example dataset which is MIT Movies
(https://groups.csail.mit.edu/sls/downloads/movie/) with currently implemented models shown in the notebooks
section.

## Requirements

Python 3.6+
PyTorch 1.5.0
TorchText 0.6.0

``` ├── LICENSE
    │
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, in this case only best BERT and with same name their
    │                         and their tensorboard logs with same names
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention, modelname_dataset and tensorboard also
    │                         tensorboard logs of some of the models which need careful examination.
    │
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- TODO: makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── base <- Base classes for implementing a design pattern
    │   │
    │   ├── data
    │   │   └── make_dataset.py <- Scripts to download or generate data
    │   │   └── build_features.py <- Scripts to turn raw data into features for modeling
    │   │   └── analyze_features.py <- For analyzing when building
    │   │
    │   ├── bilstm_trainer.py <- trainer script for bilstm
    │   │
    │   ├── char_lstm_trainer.py <- trainer script for char bi lstm TODO: will be in the same file with bilstm
    │   │
    │   ├── models         <- Scripts to the architecture of models
    │   │   │
    │   │   ├── bertner.py
    │   │   └── lstm.py
    │   │   └── char_lstm.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- TODO: tox file with settings for running tox; see tox.readthedocs.io
```

Several Named entity recognition models for training and exprerimenting with NER task.
- [x] BERT Large model architecture and training loop.
- [x] BiLSTM model architecture and training loop.
- [x] Char BiLSTM model architecture and training loop.
- [ ] Write inference for all models.
- [ ] Detailed instructions about how to use the repository.
- [ ] Wrap everything up in a pypi package.
- [ ] Refactor everything and have one more generalized training loop and inference.
- [ ] Roberta and Deberta model architecture and training loop.

## Quick Inference

TODO: Will write inference for whole models.

