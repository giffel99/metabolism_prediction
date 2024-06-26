# Information of all files in the /src/ folder

## config.py

This file controls the behavior of the entiro repo. It is possible to:

- Specify all hyperparameters. See the report for full description.
  -- Specify structure hyperparameters for the FoT.
  -- Specify hyperparameters related to learning rate.
  -- Specify other parameters such as batch size and epochs
- Train a new FoT from scratch.
- Load a trained model and further train it.
- Specify what training & evaluation dataset to use.
- Evaluate a model using our evaluation framework
- Able to conduct pretraining on the MMP set
- Able to train an ensemble of models.
- Able to conduct Optuna experiments.
- Able to change generational parameters such as beam size and n_best
- Able to postprocess predictions made by model.
- Specify whether additional print statements should be made

## dataset.py

Contains the logic for loading a curated dataset file into desired formats the framework can use.
It tokenizes, sets dataloaders, and can split sets.

## ensemble_model.py

Contains code related to the ensemble model strategy

## eval.py

Contains the neccesary logic for evaluating a saved model. It is made to be executed when evaluating a model. It uses predefined methods in the "evaluation.py", which contains the evaluation methods.

## evaluation.py

Methods for evaluation is stored here. One can use independent metrics here or checkout the eval.py which is meant to run all metrics on a saved model.

## generation_strategy.py

Logic for the generation strategies are stored here. The main one is beam_search but it also contains a greedy_decode.

## model.py

Contains methods for fetching a Focused transformer model. This file's responebility is either loading/saving a model or the construction of new ones.

## postprocessing.py

Contains methods for postprocessing predictions made by the model. Can be toggled on/off in config.py

## preprocessing.py

Contains the methods used to pre-process the data. Is meant to run once in the beginning to get hold of curated datasets from uncured datasets. Files preprocessed here are stored in "train", "test" "processed_data" folders.

## tokenizer.py

Contains code for configuring and buildning a tokenizer. This needs only to be executed once to get a tokenizer and a vocabulary.

## train.py

Contains the training framework. This file is executed for training models. See the guide for some examples.

## utils.py

Contains util functions not bound to specific parts of the frameworks for re-use.

# Folders in /src/

## curate_databases/

This folder contains preprocessing files for specific dataset used. These have been: Drugbank and Matched molecular pairs.

## data_analysis/

Contains notebooks and python scripts that gives insights to the data and model performances.

## dataset/

This folder contains the datasets used throught the project.

## focused_transformer/

Contains a copy of source files of the focused transformer.

## smiles-tokenizer

Contains the tokenizer used for this project.
