import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
import numpy as np
from functools import partial
from pathlib import Path
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
import optuna
from transformers.modeling_outputs import CausalLMOutputWithPast
from model import get_model, build_tokenizer, preload_model
from dataset import  get_train_val_dataloaders
from config import get_config, file_path, get_weights_file_path, latest_weights_file_path, latest_pretraining_weights_file_path, get_pretraining_weights_file_path, latest_weights_file_path_ensemble, get_ensemble_weights_file_path
from utils import  save_output, set_up_device
from generation_strategy import greedy_decode, beam_search
from evaluation import control_predictions_per_molecule, get_metabolite_probability, get_validity, get_topn_accuracy, fingerprint_similarity_valid_smiles
from difflib import SequenceMatcher


#### EVALUATION DONE DURING TRAINING FOR ANALLYSIS #### 
def run_eval(model, val_eval_dataloader, val_eval_unique_parents, train_eval_dataloader, train_eval_unique_parents, tokenizer, epoch, max_len, device, print_msg, print_output = True):
    perform_eval(model, val_eval_dataloader, val_eval_unique_parents, tokenizer, epoch, max_len, "val", device, print_msg, print_output)
    perform_eval(model, train_eval_dataloader, train_eval_unique_parents, tokenizer, epoch, max_len, "train", device, print_msg, print_output)

def perform_eval(model, dataloader, unique_parents, tokenizer, epoch, max_len, eval_version, device, print_msg, print_output = True):
    model.eval()
    count = 0

    top_10_preds, top_5_preds, top_1_preds, total_source = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            count += 1

            source_input = batch['src_input'].to(device) # (B, seq_len)
            labels = batch['label'].to(device) # (B, seq_len)

            # check if batch size is 1
            assert source_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out: CausalLMOutputWithPast = model(input_ids = source_input, labels = labels, return_dict = True) # (batch, seq_len, vocab)            


            # generated output sequence. Used for evaluation
            # TODO: REPLACE WITH BEAM SEARCH
            
            output_top10, _ = beam_search(model, tokenizer, 10, 10, source_input, device, max_len, print_output)
            output_top5, _ = beam_search(model, tokenizer, 5, 5, source_input, device, max_len, print_output)
            output_top1, _ = beam_search(model, tokenizer, 1, 1, source_input, device, max_len, print_output)
            
            top_10_preds.append(output_top10)
            top_5_preds.append(output_top5)
            top_1_preds.append(output_top1)
            total_source.append(batch["src_smiles"][0])

    
    metabolite_probs = get_metabolite_probability(model, unique_parents, tokenizer, device)
    true_positives_per_molecule, _, nr_children_per_molecule, total_nr_predictions  = control_predictions_per_molecule(unique_parents, total_source, top_10_preds)

    max_sims_valid, max_sims_invalid = fingerprint_similarity_valid_smiles(top_10_preds, unique_parents)

    top10_accuracy = get_topn_accuracy(true_positives_per_molecule, nr_children_per_molecule)

    top_5_true_positives, _, _, _ = control_predictions_per_molecule(unique_parents, total_source, top_5_preds)
    top5_accuracy = get_topn_accuracy(top_5_true_positives, nr_children_per_molecule)

    top_1_true_positives, _, _, _ = control_predictions_per_molecule(unique_parents, total_source, top_1_preds)
    top1_accuracy = get_topn_accuracy(top_1_true_positives, nr_children_per_molecule)
    validity, valid_smiles_per_molecule, total_nr_valid_smiles = get_validity(top_10_preds)


    save_output([metabolite_probs], 'prob_of_metabolite', config['model_basename'], epoch, eval_version)    
    save_output([max_sims_valid], 'fingerprints_top_10_valid', config['model_basename'], epoch, eval_version)    
    save_output([max_sims_invalid], 'fingerprints_top_10_invalid', config['model_basename'], epoch, eval_version)    
    
    save_output([sum(true_positives_per_molecule)], 'true_positives', config['model_basename'], epoch, eval_version)
    save_output([top1_accuracy], 'Top_1_accuracy', config['model_basename'], epoch, eval_version)
    save_output([top5_accuracy], 'Top_5_accuracy', config['model_basename'], epoch, eval_version)
    save_output([top10_accuracy], 'Top_10_accuracy', config['model_basename'], epoch, eval_version)
    save_output([sum(total_nr_predictions)], 'total_nr_predictions', config['model_basename'], epoch, eval_version)
    save_output([total_nr_valid_smiles], 'valid_smiles', config['model_basename'], epoch, eval_version)
    save_output([validity], 'validity', config['model_basename'], epoch, eval_version)
    save_output([valid_smiles_per_molecule], 'valid_smiles_per_molecule', config['model_basename'], epoch, eval_version)
    print_msg(f"{f'Valid_smiles: ':>12}{total_nr_valid_smiles}")
    return total_nr_valid_smiles


### VALIDATION LOOP ####
def run_validation(model, validation_ds, tokenizer, loss_fn, max_len, device, print_msg, print_output = True, num_examples=1):
    model.eval()
    val_losses, similarity_scores = [], []
    count = 0

    with torch.no_grad():
        for batch in validation_ds:
            count += 1

            source_input = batch['src_input'].to(device) # (B, seq_len)
            labels = batch['label'].to(device) # (B, seq_len)

            # check if batch size is 1
            assert source_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out: CausalLMOutputWithPast = model(input_ids = source_input, labels = labels, return_dict = True) # (batch, seq_len, vocab)            

            # Calculate validation loss
            out_batch, out_seq_len, out_vocab_len = model_out.logits.shape # (batch, seq_len, vocab)  
            label_batch, label_seq_len = labels.shape # (batch, seq_len)
            
            loss_out = model_out.logits.view(out_batch*out_seq_len, out_vocab_len)  # (batch * seq_len, vocabulary)
            
            loss_labels = labels.view(label_batch*label_seq_len) # (batch * seq_len)
            calculated_loss = loss_fn(loss_out,loss_labels).cpu()
            val_losses.append(calculated_loss)
            
            # generated output sequence. Used for evaluation
            # TODO: REPLACE WITH BEAM SEARCH
            model_out_smiles = greedy_decode(model, source_input, tokenizer, max_len, device)[0]

            # Source and target used for comparison / logging
            source_smiles = batch["src_smiles"][0] # This format will probabily be changed. It should be one entry of the smiles "CCOCCOO" of the parent
            target_smiles = batch["tgt_smiles"][0] # This format will probabily be changed. It should be one entry of the smiles "CCOCCOO" of the child
            
            character_similarity = SequenceMatcher(None,target_smiles, model_out_smiles).ratio()
            similarity_scores.append(character_similarity)
            if print_output and count <= num_examples:
                # Print the source, target and model output
                print_msg('-'*80)
                print_msg(f"{f'SOURCE: ':>12}{source_smiles}")
                print_msg(f"{f'TARGET: ':>12}{target_smiles}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_smiles}")
                print_msg(f"{f'SIMILARITY SCORE: ':>12}{character_similarity}")
                print_msg('-'*80)

    print_msg(f"{f'Average similarity score: ':>12}{np.average(similarity_scores)}")
    print_msg('-'*80)

    # Return the mean loss and mean similarity score. 
    return np.mean(val_losses), np.average(similarity_scores)


##############################################################################################
'''
Call necessary functions that loads and builds the tokenizers, datasets, data_loaders.
Functions used can be found in Datasets.py, models.py
'''
def get_ds(train_val_split, seq_len, batch_size, dataset):
    # This method will call the model.py file. The tokeinzer is configured there
    tokenizer = build_tokenizer() 

    train_dataloader, val_dataloader, val_eval_dataloader, val_eval_unique_parents, train_eval_dataloader, train_eval_unique_parents = get_train_val_dataloaders(tokenizer, train_val_split, seq_len, batch_size, dataset) 

    return  train_dataloader, val_dataloader, tokenizer, val_eval_dataloader, val_eval_unique_parents, train_eval_dataloader, train_eval_unique_parents

def save_model(config, epoch, model, optimizer, global_step, num_epochs, model_folder, filename): 
    print(f"Save model {filename}")
        # Save the model
    if(epoch % config["save_interval"] == 0 or num_epochs == epoch):    
        model_filename = file_path(model_folder, filename, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step':global_step
        }, model_filename)

################### TRAIN FOR ENSEMBLE MODEL #################
def train_models_for_ensemble_model(
        config, 
        lr,
        top_lr,
        num_warmup_epochs,
        scheduler_param,
        batch_size, 
        num_epochs,
        pretrain_batch_size, 
        pretrain_num_epochs, 
        pretrain_num_warmup_epochs,
        hidden_size, 
        num_hidden_layers, 
        intermediate_size,
        num_attention_heads,
        optimizer_name,
        scheduler_name
        ):
    # We make sure the weights folder exists
    Path(f"{config['datasource']}_{config['ensemble_model_folder']}").mkdir(parents=True, exist_ok=True)
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    device = set_up_device()

    print("always pretrains!")
    train_losses, val_losses, val_similarity_scores, learning_rates, lr  = pretrain_model(config, lr, top_lr, pretrain_num_warmup_epochs, scheduler_param, pretrain_batch_size, pretrain_num_epochs, hidden_size, 
                                                                   num_hidden_layers, intermediate_size, num_attention_heads, optimizer_name, scheduler_name, device)    
    print(f"Result from pretraining: {train_losses}, {val_losses}, {val_similarity_scores}")

    datasets = config["ensemble_dataset"]
    for nr_model, dataset in enumerate(datasets): 
        print(f"Train model with {dataset}")
        train_losses, val_losses, val_similarity_scores, learning_rates, opz_lr  = finetune_model_ensemble(config, lr, top_lr, num_warmup_epochs, scheduler_param, dataset, 
                                                                            nr_model, batch_size, pretrain_num_epochs + num_epochs, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, optimizer_name, scheduler_name, device, True)
        print(f"Ensemble Model nr {nr_model} result: {train_losses}, {val_losses}, {val_similarity_scores}")
        print("-"*50)
    return train_losses, val_losses, val_similarity_scores, learning_rates, opz_lr

def finetune_model_ensemble(
        config,     
        lr, 
        top_lr,
        num_warmup_epochs,
        scheduler_param,
        dataset,
        ensemble_nr, 
        batch_size, 
        num_epochs,
        hidden_size, 
        num_hidden_layers, 
        intermediate_size,
        num_attention_heads,
        optimizer_name,
        scheduler_name, 
        device,
        pretrain
        ):
    print('Finetuning started')
    
    # If we specify to use a preload model before training we load it here
    preload = config['preload']
    model_filename = latest_weights_file_path_ensemble(config, ensemble_nr) if preload == 'latest' else get_ensemble_weights_file_path(config, preload, ensemble_nr) if preload else None
    save_model_filename =  str(ensemble_nr) + "_" + config["ensemble_model_basename"]
    model_folder = model_folder = f"{config['datasource']}_{config['ensemble_model_folder']}"
    if model_filename is None and pretrain: 
        model_filename = latest_pretraining_weights_file_path(config)
    print(model_filename)

    return training_loop(config, lr, top_lr, num_warmup_epochs, scheduler_param, batch_size, num_epochs, dataset, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, 
                         optimizer_name, scheduler_name, model_filename, model_folder, save_model_filename, device)

######################## TRAIN LOOP ###########################
def train_model(
        config, 
        lr,
        top_lr,
        num_warmup_epochs,
        scheduler_param,
        batch_size, 
        num_epochs,
        pretrain_batch_size, 
        pretrain_num_epochs, 
        pretrain_num_warmup_epochs,
        hidden_size, 
        num_hidden_layers, 
        intermediate_size,
        num_attention_heads,
        optimizer_name,
        scheduler_name
        ): 
    
    # We make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    device = set_up_device()

    if config["pretrain"]: 
        pretrain_train_losses, pretrain_val_losses, pretrain_val_similarity_scores, pretrain_learning_rates, lr, pretrain_nr_valid_smiles  = pretrain_model(config, lr, top_lr, pretrain_num_warmup_epochs, scheduler_param, pretrain_batch_size, pretrain_num_epochs, hidden_size, 
                                                                   num_hidden_layers, intermediate_size, num_attention_heads, optimizer_name, scheduler_name, device)
        
        #print(f"Result from pretraining: {pretrain_train_losses[-1]}, {pretrain_val_losses[-1]}, {pretrain_val_similarity_scores[-1]}")
        
        train_losses, val_losses, val_similarity_scores, learning_rates, lr, nr_valid_smiles = finetune_model(config, lr, top_lr, num_warmup_epochs, scheduler_param, batch_size, pretrain_num_epochs + num_epochs, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, optimizer_name, scheduler_name, device, True)
        
        pretrain_train_losses += train_losses
        pretrain_val_losses += val_losses
        pretrain_val_similarity_scores += val_similarity_scores
        pretrain_learning_rates += learning_rates
        pretrain_lr += lr
        pretrain_nr_valid_smiles +=nr_valid_smiles
        return pretrain_train_losses, pretrain_val_losses, pretrain_val_similarity_scores, pretrain_learning_rates, lr, pretrain_nr_valid_smiles
    return finetune_model(config, lr, top_lr, num_warmup_epochs, scheduler_param, batch_size, num_epochs, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, optimizer_name, scheduler_name, device, False)

def finetune_model(
        config,     
        lr, 
        top_lr,
        num_warmup_epochs,
        scheduler_param,
        batch_size, 
        num_epochs,
        hidden_size, 
        num_hidden_layers, 
        intermediate_size,
        num_attention_heads,
        optimizer_name,
        scheduler_name, 
        device,
        pretrain
        ):
    print('Finetuning started')
    dataset = config['train_dataset']
    
    # If we specify to use a preload model before training we load it here
    preload = config['preload']
    model_filename = latest_weights_file_path(config, config['model_basename']) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    save_model_filename = config["model_basename"]
    model_folder = model_folder = f"{config['datasource']}_{config['model_folder']}"
    if model_filename is None and pretrain: 
        model_filename = latest_pretraining_weights_file_path(config)

    return training_loop(config, lr, top_lr, num_warmup_epochs, scheduler_param, batch_size, num_epochs, dataset, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, 
                         optimizer_name, scheduler_name, model_filename, model_folder, save_model_filename, device)

def pretrain_model(
    config,     
    lr,
    top_lr,
    num_warmup_epochs,
    scheduler_param,
    batch_size, 
    num_epochs,
    hidden_size, 
    num_hidden_layers, 
    intermediate_size,
    num_attention_heads,
    optimizer_name,
    scheduler_name, 
    device):
    
    print('Pretraining started')
    dataset = config['pretrain_dataset'] 

    preload = config['pretrain_preload']
    model_filename = latest_pretraining_weights_file_path(config) if preload == 'latest' else get_pretraining_weights_file_path(config, preload) if preload else None

    save_model_filename = config["pretrain_model_basename"]
    model_folder = f"{config['datasource']}_{config['model_folder']}"

    return training_loop(config, lr, top_lr, num_warmup_epochs, scheduler_param, batch_size, num_epochs, dataset, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, 
                         optimizer_name, scheduler_name, model_filename, model_folder, save_model_filename, device)

def training_loop(
    config, 
    lr,
    top_lr,
    num_warmup_epochs,
    scheduler_param,
    batch_size, 
    num_epochs,
    dataset, 
    hidden_size, 
    num_hidden_layers,
    intermediate_size,
    num_attention_heads,
    optimizer_name,
    scheduler_name,
    model_filename, 
    model_folder, 
    model_file_start, 
    device
    ):

    seq_len = config['seq_len']
    train_val_split = config['train_val_split']
    vocab_size = config['vocab_size']
    max_position_embeddings = config['max_position_embeddings']
    train_dataloader, val_dataloader, tokenizer, val_eval_dataloader, val_eval_unique_parents, train_eval_dataloader, train_eval_unique_parents = get_ds(train_val_split, seq_len, batch_size, dataset)
    model = get_model(tokenizer, vocab_size, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, max_position_embeddings)
    model.to(device)
     
    initial_epoch = 0
    global_step = 0
    
    # Used for Optuna. To find what optimizer is used, check the config file
    # Fetches a configured optimizer
    optimizer_constructors = {
        "adam": lambda: torch.optim.Adam(model.parameters(), lr=top_lr, eps=1e-8),
        "sgd": lambda: torch.optim.SGD(model.parameters(), lr=top_lr, momentum=0.9)
    }
    optimizer = optimizer_constructors[optimizer_name]()
    if model_filename:
        print(f"Load model: {model_filename}")
        model, initial_epoch, optimizer, global_step = preload_model(model_filename, model, optimizer, device)
    else:
        print('No model to preload, starting from scratch')

    # Used for optuna. To find what scheduler is used, check the config file
    # The scheduler_param is the hyperparameter the scheduler use. Check config file. Change this this parameter if changing the scheduler
    scheduler_constructors = {
        "exponential": lambda: ExponentialLR(optimizer, gamma=scheduler_param),
        "linear": lambda: LinearLR(optimizer, start_factor=0.33, end_factor=scheduler_param[0], total_iters=scheduler_param[1])
    }
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=lr/top_lr, 
        end_factor=1, 
        total_iters=num_warmup_epochs,
        last_epoch=-1
        )
    
    scheduler = scheduler_constructors[scheduler_name]()
    combined_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[num_warmup_epochs])
    
    # Our defined loss function. this can be swapped for others.
    # ignore_index make sure we dont compare the padding token because. We dont want that. Label smoothing turned off per default.
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(device)

    # the variables below are the result of the train_loop. The purpose is to give insight to performance
    train_losses, val_losses, val_similarity_scores, learning_rates, nr_valid_smiles = [], [], [], [], []

    for epoch in range(initial_epoch, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        average_train_losses = []

        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d} with lr {optimizer.param_groups[0]["lr"]}')
        for batch in batch_iterator:            
            source_input = batch['src_input'].to(device) # (batch, seq_len, vocab)
            
            # Masking all successing elements
            #attention_mask = batch['attention_mask']
            #attention_mask = torch.squeeze(attention_mask, dim=1)

            # compare the output with the label
            labels = batch["label"].to(device) # (batch, seq_len)

            # Inputs the src, labels, and (optinally) attention_maks. Outputs an object containing logits, attentions
            model_out: CausalLMOutputWithPast = model(input_ids = source_input, labels = labels, return_dict = True) # (batch, seq_len, vocab)            
            
            # Compute the loss
            out_batch, out_seq_len, out_vocab_len = model_out.logits.shape # (batch, seq_len, vocab)  
            label_batch, label_seq_len = labels.shape # (batch, seq_len)
            
            loss_out = model_out.logits.view(out_batch*out_seq_len, out_vocab_len)  # (batch * seq_len, vocabulary)
            
            loss_labels = labels.view(label_batch*label_seq_len) # (batch * seq_len)
            loss = loss_fn(loss_out,loss_labels)
          
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            average_train_losses.append(loss.item())
 
            # Backpropagate the loss
            loss.backward()
                       
            # Update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
            global_step+=1 
        train_losses.append(np.average(average_train_losses))        

        # step the scheduler and log the new lr
        combined_scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # After all batches in epoch is finished
        validation_loss, val_similarity_score = run_validation(model, val_dataloader, tokenizer, loss_fn, seq_len, device, lambda msg: batch_iterator.write(msg), config['val_print_output'])
       
        save_output([np.average(average_train_losses)], 'train_losses', config['model_basename'], epoch)
        save_output([optimizer.param_groups[0]["lr"]],'learning_rates', config['model_basename'], epoch)       
        save_output([validation_loss], 'val_losses', config['model_basename'] , epoch)
        save_output([val_similarity_score],'val_similarities', config['model_basename'], epoch)

        if (epoch % 5 == 0): 
            valid_smiles = run_eval(model, val_eval_dataloader, val_eval_unique_parents, train_eval_dataloader, train_eval_unique_parents, tokenizer, epoch, seq_len, device, lambda msg: batch_iterator.write(msg), config['val_print_output'])
            nr_valid_smiles.append(valid_smiles)

        # Print progress
        if(config['train_print_output']):
            print(f'Epoch {epoch}.')
            print(f"Average loss was:             {train_losses[-1]}")
            print(f"Average validation loss was:  {validation_loss}")
            print(f"Average similarity score was: {val_similarity_score}")

        # Append the validation scores
        val_losses.append(validation_loss)
        val_similarity_scores.append(val_similarity_score)
        save_model(config, epoch, model, optimizer, global_step, num_epochs, model_folder, model_file_start)
        
        
        # If the loss average loss for the twenty recent runs has not decreased we decide to stop the training.
        if( len(val_losses) >= 30 and min(val_losses)*1.2 < val_losses[-1] ):
            print(f"Training ended earlier due to no decrease of loss. Model trained for: {epoch} epochs.")
            return train_losses, val_losses, val_similarity_scores, learning_rates, optimizer.param_groups[0]["lr"], nr_valid_smiles
        

    return train_losses, val_losses, val_similarity_scores, learning_rates, optimizer.param_groups[0]["lr"], nr_valid_smiles
#### OPTUNA #### 
##############################################################################################

def objective(trial, config): 
    batch_size = trial.suggest_int('batch_size', 8, 32)
    num_epochs = 80 #With early stopping
    pretrain_batch_size = config['pretrain_batch_size']
    pretrain_num_epochs = config['pretrain_num_epochs']
    pretrain_num_warmup_epochs = config['pretrain_num_warmup_epochs']
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    # Ensure top_lr is always greater than lr by multiplying lr by a small factor > 1
    min_top_lr = lr * (1 + 1e-9)  # Adjust the 1e-9 factor based on required precision
    top_lr = trial.suggest_float('top_lr', min_top_lr, 1e-1, log=True)
    # If top lr still is smaller than lr than we set them to equal as a safety measure
    if(top_lr < lr):
        print("top_lr still managed to be smaller than lr. Setting them to equal value.")
        top_lr = lr
    
    num_warmup_epochs = trial.suggest_int('num_warmup_epochs', 1, 20)
    hidden_size = trial.suggest_categorical('hidden_size',[64, 128, 256, 512, 1024]) 
    num_attention_heads = trial.suggest_categorical('num_attention_heads', [2, 4, 8, 16])
    num_hidden_layers = trial.suggest_categorical('num_hidden_layers', [2, 3 ,4])
    intermediate_size = trial.suggest_categorical('intermediate_size',[64, 128, 256, 512, 1024]) 
    scheduler_name = trial.suggest_categorical('scheduler', ['exponential', 'linear'])
    optimizer_name = "adam"
    # Different schedulers use different parameters thus must be checked
    
    if(scheduler_name == "linear"):
        linear_end_factor = trial.suggest_categorical('linear_end_factor', [1e-5, 1e-4, 1e-3, 1e-2])
        linear_num_epochs = trial.suggest_int('linear_num_epochs', 1, (num_epochs - num_warmup_epochs))
        scheduler_param = [linear_end_factor, linear_num_epochs]
    elif(scheduler_name == "exponential"):
        scheduler_param = trial.suggest_categorical('exponential_param', [0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
    
    
    print(f"Trial {trial}: with batchsize {batch_size}, epochs {num_epochs}, and learning rate {lr}, and hidden size {hidden_size}, and num_layers {num_hidden_layers}, and intermediate_size: {intermediate_size}, and attention heads {num_attention_heads}")
    print(f'Using optimizer: {optimizer_name} and scheduler: {scheduler_name}, top_lr: {top_lr}, num_warmup_epochs: {num_warmup_epochs}, and schedule params: {scheduler_param}')
    
    train_losses, val_losses, val_similarity_scores, learning_rates, _, nr_valid_smiles = train_model(
        config,
        lr,
        top_lr, 
        num_warmup_epochs, 
        scheduler_param, 
        batch_size, 
        num_epochs, 
        pretrain_batch_size, 
        pretrain_num_epochs,
        pretrain_num_warmup_epochs,
        hidden_size, 
        num_hidden_layers,
        intermediate_size,
        num_attention_heads, 
        optimizer_name, 
        scheduler_name
        )
    return min(val_losses)

def train_with_optuna(config): 
    study = optuna.create_study(
        storage="sqlite:///db.fot",  
        study_name="fot-schedule_optimizer_20",
        directions=["minimize"]
        )
    study.optimize(partial(objective, config=config), n_trials=config["trials"]) #100
    print(f"Best value: {study.best_value} (params: {study.best_params})")

#### TRAIN NORMAL ####

def train(config): 
    lr = config['lr']
    top_lr = config['top_lr']
    batch_size = config['batch_size']
    pretrain_batch_size = config['pretrain_batch_size']
    num_epochs = config['num_epochs']
    pretrain_num_epochs = config['pretrain_num_epochs']
    hidden_size = config['hidden_size'] 
    num_hidden_layers = config['num_hidden_layers']
    intermediate_size = config['intermediate_size'] 
    num_attention_heads = config['num_attention_heads']
    num_warmup_epochs = config['num_warmup_epochs']
    pretrain_num_warmup_epochs = config['pretrain_num_warmup_epochs']
    scheduler_param = config['scheduler_param']
    optimizer_name = config['optimizer']
    scheduler_name = config['scheduler']

    if config["ensemble_model"]: 
        print("Ensemble model")
        train_losses, val_losses, val_similarity_scores, learning_rates, _ = train_models_for_ensemble_model(
            config,
            lr,
            top_lr, 
            num_warmup_epochs, 
            scheduler_param, 
            batch_size, 
            num_epochs, 
            pretrain_batch_size, 
            pretrain_num_epochs,
            pretrain_num_warmup_epochs,
            hidden_size, 
            num_hidden_layers,
            intermediate_size,
            num_attention_heads, 
            optimizer_name, 
            scheduler_name)    
    else: 
        print("Model, not ensemble")
        train_losses, val_losses, val_similarity_scores, learning_rates, _, valid_smiles = train_model(
            config,
            lr,
            top_lr, 
            num_warmup_epochs, 
            scheduler_param, 
            batch_size, 
            num_epochs, 
            pretrain_batch_size, 
            pretrain_num_epochs,
            pretrain_num_warmup_epochs,
            hidden_size, 
            num_hidden_layers,
            intermediate_size,
            num_attention_heads, 
            optimizer_name, 
            scheduler_name
            )
    # Print results from the train session
    if config['train_print_output']: 
        print('-'*80)
        print(train_losses)
        print(val_losses)
        print(val_similarity_scores)
        print('-'*80)

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    if(config['is_optuna_session']):
        train_with_optuna(config)
    else:
        train(config)
    
