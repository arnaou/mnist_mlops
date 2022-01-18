MNIST Project
==============================
Repository for MNIST project conducted as part of the course ML-OPS (02476) Jan 2022
by: arnaou@kt.dtu.dk

Installation
------------
to install dependencies run the following command
```
pip install -r requirements.txt
```
Alternatively, dockers can be used as follows:
```
docker build -f trainer.dockerfile . -t trainer:latest
```

Usage
------------
**MAKE SURE YOU ARE IN THE MAIN DIRECTORY BEFORE RUNNING ANY OF THE COMMANDS**

To construct mnist data
```
python src/data/make_dataset.py --input_filepath data/raw --output_filepath data/processed
```
Alternatively
```
make data
```
To train the model
```
python src/models/train_model.py experiment_model=exp_mod1 experiment_train=exp_tr1 
```
where *experiment_model* and *experiment_train* refers to the config files containing the parameters for the model and the training procedure respectively.

To perform the prediction
```
python src/models/predict_model.py experiment_predict: exp_pred1
```
Where *experiment_train* refers to the config file containing the parameters for the prediction.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── exercices          <- Contains folder and exercices done as part of the course.
    |
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

Usage
------------
**MAKE SURE YOU ARE IN THE MAIN DIRECTORY BEFORE RUNNING ANY OF THE COMMANDS**

To construct mnist data
```
python src/data/make_dataset.py --input_filepath data/raw --output_filepath data/processed
```
Alternatively
```
make data
```
To train the model
```
python src/models/train_model.py --data_path data/processed  --input_size 784 --output_size 10 --hidden_layers 512 256 128 --dropout_rate 0.2 
```
To perform the prediction
```
python src/models/predict_model.py --model_path models/fcModel.pt --data_path data/raw/example_images.npy
```



check list
------------
- [x] get cookiecutter










--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
