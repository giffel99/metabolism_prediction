import torch
import warnings
from pathlib import Path
from evaluation import perform_evaluation_of_result
from utils import set_up_device
from dataset import arrange_molecules_with_all_metabolites, get_eval_dataloader
from config import get_config, latest_weights_file_path, latest_weights_file_path_ensemble
from model import build_tokenizer, get_model, preload_model
from generation_strategy import beam_search, beam_search_ensemble
from postprocessing import postprocess_data

##################### TEST METHOD #############################
def create_source_and_predictions_based_on_dict(parent_child_dict):
    source_smiles_list = list(parent_child_dict.keys())
    predicted_list = [parent_child_dict[parent] for parent in source_smiles_list]
    return source_smiles_list, predicted_list

### EVALUATION LOOP ####
def get_ds(data_set_name, seq_len):
    tokenizer = build_tokenizer() # This method will call the model.py file. The tokenizer is configured there

    data, eval_dataloader = get_eval_dataloader(tokenizer, data_set_name, seq_len) # This method will call a function in dataset.py which will load the data in the desired format

    return data, eval_dataloader, tokenizer

### EVALUATION LOOP ####
def run_evaluation(config, beam_size, n_best, name_of_run): 
    seq_len = config['seq_len']
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    max_position_embeddings = config['max_position_embeddings']
    intermediate_size = config['intermediate_size']
    optimizer_name = config['optimizer']
    top_lr = config['top_lr']
    
    torch.cuda.empty_cache()
    device = set_up_device()

    # We make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    data_set_name = config['eval_dataset']
    data, eval_dataloader, tokenizer = get_ds(data_set_name, seq_len)
    
    model = get_model(tokenizer, vocab_size, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, max_position_embeddings) # This will need to be updated when we know what parameters our model will use.
    
    # The scheduler_param is the hyperparameter the scheduler use. Check config file. Change this this parameter if changing the scheduler
    optimizer_constructors = {
        "adam": lambda: torch.optim.Adam(model.parameters(), lr=top_lr, eps=1e-8),
        "sgd": lambda: torch.optim.SGD(model.parameters(), lr=top_lr, momentum=0.9)
    }
    optimizer = optimizer_constructors[optimizer_name]()
    
    model_filename = latest_weights_file_path(config, config['model_basename'])

    if model_filename: 
        model, _, _, _ = preload_model(model_filename, model, optimizer, device)
    else:
        raise Exception("No model to load, train the model first")
    model.eval()

    return evaluation_loop(model, beam_search, data, eval_dataloader, tokenizer, device, seq_len, beam_size, n_best, name_of_run, config["eval_print_output"], config["post_process_active"])

def run_evaluation_ensemble(config, beam_size, n_best, name_of_run): 
    seq_len = config['seq_len']
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    max_position_embeddings = config['max_position_embeddings']
    intermediate_size = config['intermediate_size']
    optimizer_name = config['optimizer']
    top_lr = config['top_lr']

    torch.cuda.empty_cache()
    device = set_up_device()

    # We make sure the weights folder exists
    Path(f"{config['datasource']}_{config['ensemble_model_folder']}").mkdir(parents=True, exist_ok=True)
    
    data_set_name = config['eval_dataset']
    data, eval_dataloader, tokenizer = get_ds(data_set_name, seq_len)
    model = get_model(tokenizer, vocab_size, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, max_position_embeddings) # This will need to be updated when we know what parameters our model will use.

    # The scheduler_param is the hyperparameter the scheduler use. Check config file. Change this this parameter if changing the scheduler
    optimizer_constructors = {
        "adam": lambda: torch.optim.Adam(model.parameters(), lr=top_lr, eps=1e-8),
        "sgd": lambda: torch.optim.SGD(model.parameters(), lr=top_lr, momentum=0.9)
    }
    optimizer = optimizer_constructors[optimizer_name]()
    
    models = []
    for nr in range(config["ensemble_nr"]): 
        model_filename = latest_weights_file_path_ensemble(config, nr)

        if model_filename: 
            model, _, _, _ = preload_model(model_filename, model, optimizer, device)
            models.append(model)
            model.eval()
        else:
            raise Exception("No model to load, train the model first")

    return evaluation_loop(models, beam_search_ensemble, data, eval_dataloader, tokenizer, device, seq_len, beam_size, n_best, name_of_run, config["eval_print_output"], config["post_process_active"])
    
def evaluation_loop(models, decode_strategy, data, evaluation_ds, tokenizer, device, max_len, beam_size, n_best, name_of_run, print_output, post_process_active):
    count = 0
    parent_child_dict = arrange_molecules_with_all_metabolites(data)
    source_smiles_list, predicted = [], []

    with torch.no_grad():
        for batch in evaluation_ds:
            count += 1
            src_input = batch['src_input'].to(device) # (B, seq_len)

            # check if batch size is 1
            assert src_input.size(0) == 1, "Batch size must be 1 for evaluation"

            model_out_smiles, _ = decode_strategy(models, tokenizer, beam_size, n_best, src_input, device, max_len, print_output)
            if post_process_active: 
                model_out_smiles = postprocess_data(model_out_smiles, print_output)
            
            source_smiles = batch["src_smiles"][0] # This format will probabily be changed. It should be one entry of the smiles "CCOCCOO" of the parent
            target_smiles = batch["tgt_smiles"][0]

            if print_output: 
                print(f'For {source_smiles}')
                print(f'True one:')
                print(target_smiles)
                print(f'The predictions was: ')
                for pred in model_out_smiles: 
                    print(pred)
                print('-'*80)
            source_smiles_list.append(source_smiles)
            predicted.append(model_out_smiles)
    print("Metabolites generated")
    return perform_evaluation_of_result(n_best, parent_child_dict, source_smiles_list, predicted,  name_of_run, beam_size)

def eval_model(config):
    if config["ensemble_model"]:
        print("Check ensemble model")
        precision, recall, sim = run_evaluation_ensemble(config, config['beam_size'], config['n_best'], "Ensemble_standard")
    else: 
        precision, recall, sim = run_evaluation(config, config['beam_size'], config['n_best'], config["model_basename"]) 

## evaluation with top n. Tries out 3 different beam sizes and runs the evaluation loop for them all.
def eval_model_with_top_n(config): 
    n_bests = [5, 10, 15]
    beam_sizes = [5, 10, 15]
    for n_best in n_bests: 
        for beam_size in beam_sizes: 
            if beam_size > n_best: 
                print(f"Run with n_best: {n_best} and beam_size: {beam_size}")
                if config["ensemble_model"]:
                    precision, recall, sim = run_evaluation_ensemble(config, config['beam_size'], config['n_best'], "Ensemble_top_n")
                else: 
                    precision, recall, sim = run_evaluation(config, beam_size, n_best, "Basemodel_top_n") 

def main():
    warnings.filterwarnings('ignore')
    config = get_config()
    eval_model(config)
    #eval_model_with_top_n(config)
    
if __name__ == "__main__":    
   main()


