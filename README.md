## MOVIE NER
├── LICENSE
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

BERT_MIT_MOVIES_PRETRAINED_DOWNLOAD (pretrained model and logs)
https://drive.google.com/drive/folders/174xu6RdQfaHlPC5YYp6Gk9UOYAnE7TTT?usp=sharing
BERT_MIT_TRIVIA_PRETRAINED_DOWNLOAD (pretrained model and logs)
https://drive.google.com/drive/folders/1L2x_xiEcOWv7kC7-MTcborWHw5EkJNri?usp=sharing