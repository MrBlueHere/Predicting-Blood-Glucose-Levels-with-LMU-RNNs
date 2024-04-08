# Predicting-Blood-Glucose-Levels-with-LMU-RNNs

## Introduction
This is the implementation of the paper "Predicting Blood Glucose Levels with LMU Recurrent Neural Networks: A Novel Computational Model".

### Abstract
Type 1 diabetes disrupts normal blood glucose regulation
due to the destruction of insulin-producing cells, necessitating insulin
therapy through injections or insulin pumps. Consumer devices can forecast blood glucose levels by leveraging data from blood glucose sensors
and other sources. Such predictions are valuable for informing patients
about their blood glucose trajectory and supporting various downstream
applications. Numerous machine-learning models have been explored for
blood glucose prediction.

This study introduces a novel application of Legendre Memory Units for
blood glucose prediction. Employing a multivariate time series, predictions are made with 30-minute and 60-minute horizons. The proposed
model is comparable with state-of-the-art models on the OhioT1DM
dataset, encompassing eight weeks of data from 12 distinct patients.

## Requirements
Run with >= Python 3.10, requirements are available in `requirements.txt`. Install them with `pip install -r requirements.txt`. 

## Data
The data used in this project is the OhioT1DM dataset, which is available at http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html. The data should have the following structure:
```
`-- data
    `-- OhioT1DM
        |-- 2018
        |   |-- train
        |   |   |-- 559-ws-training.xml
        |   |   |-- ...
        |   `-- test
        |       |-- 559-ws-testing.xml
        |       |-- ...
        `-- 2020
            |-- train
            |   |-- 540-ws-training.xml
            |   |-- ...
            `-- test
                |-- 540-ws-testing.xml
                |-- ...
```


## Usage
Notebooks which can be re-run to reproduce the results are available in the `code` directory as well as the script for hyperparameter tuning.

## Authors
- Ladislav Floriš
- Daniel Vašata
