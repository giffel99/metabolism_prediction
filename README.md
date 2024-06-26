# Master thesis: AI for prediction of metabolites

## Prediction of Metabolites

This is the code from the master thesis AI for prediction of metabolites written by Amanda Dehlén and Pär Aronsson 2024.

## Contents

This repository contains the necessary files for training the Focused Transformer (FoT) from scratch with various features one can add.

Separate README files are provided for the datasets in respective directories and are located in the /dataset/ folder. The train/, test/, and processed_data/ contain the datasets used. A detailed description of the files and folders in /src/ can be found in the README in /src/.

## Features

In this repository it is possible to:

- Get access to publicly curated metabolic reaction data sourced Drugbank and MetXBioDB with a size of ~3k pairs.
- Get access to publicly curated matched molecular pairs data with a size of ~1M pairs.
- Train a Focused Transformer from scratch or further train one.
- Conduct two types of strategies for improving performance.
  -- Pre-training on a large set of curated matched molecular pairs. This dataset contains ~ 1M datapoints of SMILES pairs.
  -- Ensemble modeling. One can specify how many models to use.
- Possible to augment more datapairs.

## Guide

This section will provide some easy guides on what one can do in this repository.

### Prerequisites

- This project used python 3.9.7.
- Regarding pytorch, we used pytorch with cuda 11.8
- Run: pip install -r requirements.txt.

### Switching what dataset to use

1. Checkout the folders /train/, /test/ /processed_data/ for available curated datasets.
2. Checkout config.py where you can configure what dataset to use.
   2.1. For finetuning change: "train_dataset"
   2.2. For pretraining change: "pretrain_dataset"
   2.3. For evaluation change: "eval_dataset"

### Training a model

1. For training you want to start with checking out config.py to specify the desired hyperparameters for your setup.
   1.1 If you want to train a new model you set "preload" to false, to start a new model. Else set it as a string "latest"
   1.2 You want to set the name for you model which is "model_basename". If you preload a model you need to match the name of this name to the one you have saved. You can view saved models per default in "/fotmodel_weights/"
   1.3 specify what dataset you want to train and eval on.
2. For a finetuned model on metabolic set.
   2.1 Set "pretrain" and "ensemble_model" to false.
3. For pretraining and finetuning.
   3.1 set "pretrain" to True
4. For ensemble training. Set "ensemble model" to true
5. Lastly you want to run the command "python train.py" in the terminal

### Loading a pretrained model and training it

1. go to config.py
   1.1. set "preload" to true
   1.2 You want to match the pretrained model name with "pretrain_model_basename".
   1.3. You want to set "pretrain_preload" to "latest" and pretrain true
   1.4. Name the finetuned model you want to train with "model_basename"
2. Run train.py in the terminal.

### Evaluating a trained model.

1. Go to config.py
   1.1. Checkout "eval_dataset" and set the dataset you want to evaluate on.
   1.2. If you want to postprocess you predictions set "post_process_active" to true.
   1.3 You can modify the beam-search parameters "beam_size" for how many predictions per input and "n_best" with how many of the best to save.
2. Specify what model to eval
   2.1. If its a finetuned model match the "model_basename"
   2.2. If its a pretrained ONLY model then match "pretrain_model_basename"
3. Run eval.py
4. Checkout data/evaluation/evaluation_runs.csv for your results.

## Requirements

In the file requirements.txt you can see the dependencies required for this repo
The Python version used was version 3.9.7

## Acknowledgements

We would like to thank our academic supervisor and company advisor Rocío Mercado Oropeza and Filip Miljovi\'c for their invaluable feedback, insightful discussions, and for keeping us on the right track. A thanks to AstraZeneca is also given, for the opportunity to perform this master thesis. Ola Engkvist also deserves our thanks for his dedicated service as our examiner.

We want to also acknowledge the work of the creat
ors of the model Focused Transformer, which have made this project possible.

## Disclaimer
