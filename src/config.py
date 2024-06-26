from pathlib import Path

def get_config():
    return {
        "vocab_size": 32, # The Tokenizer has 29 tokens + 3 special tokens thus the model needs to have a size of atleast 32 
        "hidden_size":64,
        "num_hidden_layers":2,
        "num_attention_heads":2,
        "max_position_embeddings":1024,
        "intermediate_size":64,

        # Train loop specific
        "batch_size": 16,
        "num_epochs": 200,
        "seq_len": 200,
        "train_val_split":0.8999,
        "save_interval": 10, 
        "num_warmup_epochs":9,
        # Specify which folder it lies in. "train", "test", "processed_data"
        "train_dataset": "train/metabolic_smiles", #train/metabolic_smiles"metxbiodb_smiles", "metxbiodb_unique_parents", "metxbiodb_smiles_small", "metabolic_smiles"
        
        # Loss and optimization specifics
        "lr": 0.000048,
        "top_lr": 0.000132,
        "loss_threshold":1e-20,
        "optimizer":'adam', # Available: adam, sgd
        'scheduler':'exponential', # Available: exponential, linear
        "scheduler_param":0.95, # ExponentialLR: 0.9, Cosine: 0.5, Linear: num_epochs   

        # Model and tokenizer specific
        "datasource": 'trained_fot',
        "model_folder": "weights",
        "model_basename": "fotmodel_", 
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json", 
        "experiment_name": "runs/fotmodel_",

        # Pretraining loop specific
        "pretrain": False, 
        "pretrain_preload": None,
        "pretrain_batch_size": 128, 
        "pretrain_num_epochs": 200, 
        "pretrain_num_warmup_epochs":30,
        "pretrain_model_basename": "fot_pretrain_fst_may_twenty_",
        # Specify which folder it lies in. "train", "test", "processed_data" 
        "pretrain_dataset": "train/matched_molecular_pairs_small", #matched_molecular_pairs

        # Ensemble model
        "ensemble_model": False,
        "ensemble_nr": 3,
        "ensemble_model_folder": "ensemble",
        "ensemble_model_basename": "fotmodel_ensemble", 
        "ensemble_dataset": ["train/drugbank_smiles", "train/metxbiodb_smiles", "train/augmented_smiles"], 

        # Beam search specific
        "beam_size": 10, 
        "n_best": 10, 

        # Eval loop specific 
        # Specify which folder it lies in. "train", "test", "processed_data"
        "eval_dataset": "test/metabolic_smiles", #"gloryx_smiles_first_generation", "metxbiodb_smiles_small", "metxbiodb_smiles", "metxbiodb_unique_parents", "metabolic_smiles"
        "post_process_active": False, 

        # Optuna
        "trials": 300, 
        "is_optuna_session":False,

        # printing result
        "val_print_output": False, 
        "train_print_output": True, 
        "eval_print_output": True,
        "save_loss":True,

    }
def file_path(model_folder, filename, epoch : str): 
    model_filename = f"{filename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def get_pretraining_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    return file_path(model_folder, config['pretrain_model_basename'], epoch)

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    return file_path(model_folder, config['model_basename'], epoch)

def get_ensemble_weights_file_path(config, epoch: str, ensemble_nr): 
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{ensemble_nr}_{config['ensemble_model_basename']}*.pt"
    return file_path(model_folder, model_filename, epoch)   

############# LATEST FILEPATHS ##################################
def latest_pretraining_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['pretrain_model_basename']}*"
    return latest_file_path(model_folder, model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config, model_filename):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    return latest_file_path(model_folder, f"{model_filename}*")
    
def latest_weights_file_path_ensemble(config, ensemble_nr): 
    model_folder = f"{config['datasource']}_{config['ensemble_model_folder']}"
    model_filename = f"{ensemble_nr}_{config['ensemble_model_basename']}*.pt"
    return latest_file_path(model_folder, model_filename)

def latest_file_path(model_folder, model_filename): 
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files = [str(file) for file in weights_files]
    
    weights_files_with_number = [(int("".join([i for i in file if i.isdigit()])), file) for file in weights_files]
    weights_files_with_number.sort(reverse=True)
    (_, weights_file) = weights_files_with_number[0]

    return str(weights_file)
