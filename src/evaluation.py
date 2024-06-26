from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import os
import torch
from utils import fingerprint_similarity_single, validate_molecule_match, fingerprint_similarities_multiple, valid_smiles
import torch.nn.functional as F

################# HANDLE SAVE OF EVALUATION ###################
def save_evaluation(n_best, nr_valid_smiles, tp, fp, fn, validity, valid_smiles_per_molecule, precision, recall, topn_accuracy, p_one_m, p_half_m, p_all_m, out_size_avg, total_avg_sim, avg_sims, beam_size = None, name_of_run = "standard", directory = "data/evaluation", filename = "evaluation_runs.csv"): 
    eval_df = pd.DataFrame()
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = "/".join([directory, filename])

    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        eval_df = pd.DataFrame(data)
    else: 
        eval_df = pd.DataFrame(columns=['Name_of_run', 'Beam_size', 'N_best', 'Valid_smiles', 'True_positives','False_positives', 'False_negatives', 'validity', 'topn_accuracy', 'Precision','Recall','Per_one_metabolite','Per_half_metabolite','Per_all_metabolite', 'valid_smiles_per_molecule', 'Output_size_avg', 'Total_avg_sim', 'Avg_sims'])
    new_datapoint = {"Name_of_run": name_of_run, 'Beam_size': beam_size,'N_best': n_best, 'Valid_smiles': nr_valid_smiles,  "True_positives": tp, "False_positives": fp, 'False_negatives': fn, 'validity':validity, 'topn_accuracy': topn_accuracy,"Precision" : precision, "Recall": recall, "Per_one_metabolite" : p_one_m, "Per_half_metabolite": p_half_m, "Per_all_metabolite" : p_all_m, 'valid_smiles_per_molecule': valid_smiles_per_molecule, 'Output_size_avg': out_size_avg, 'Total_avg_sim':total_avg_sim, 'Avg_sims': avg_sims}
    eval_df.loc[len(eval_df)] = new_datapoint
    eval_df.to_csv(filepath,index=False)

def save_predictions(parents, parent_dict, predictions, directory = "data/evaluation", filename = "predictions.csv"): 
    pred_df = pd.DataFrame()
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = "/".join([directory, filename])
    metabolites = []
    for parent in parents: 
        metabolites.append(parent_dict.get(parent))

    pred_df["parents_smiles"] = parents
    pred_df["metabolites"] = metabolites
    pred_df["predictions"] = predictions
    
    pred_df.to_csv(filepath,index=False)

def save_true_predictions_only(parents, parent_dict, predictions, positives_per_mol, directory = "data/evaluation", filename = "predictions_tp.csv"): 
    pred_df = pd.DataFrame()
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = "/".join([directory, filename])    
    pars, mets, preds, trues = [], [], [], []
    for i in list(range(0, len(parents))): 
        if positives_per_mol[i] > 0: 
            parent = parents[i]
            pars.append(parent)
            mets.append(parent_dict.get(parent))
            preds.append(predictions[i])
            trues.append(positives_per_mol[i])

    pred_df["parents_smiles"] = pars
    pred_df["true_positives"] = trues
    pred_df["metabolites"] = mets
    pred_df["predictions"] = preds
    
    pred_df.to_csv(filepath,index=False)

################# CONTROL RESULT METHOD ##################
# parent_child_dict: a dictionary where the key is a molecule and the value is a list of their metabolites
# parents: a list of the smiles given the model
# prediction_list: a list of list with prediction for each parent
# check number of correct predictions per parent
def control_predictions_per_molecule(parent_child_dict, parents, prediction_list): 
    correct_predictions, false_negatives, total_nr_predictions, nr_children_per_parent = [], [], [], []
    for parent, predictions in zip(parents, prediction_list): 
        children = parent_child_dict[parent]   
        nr_children_per_parent.append(len(children))
        total_nr_predictions.append(len(predictions))
        if predictions != []: 
            # validate for each prediction for a molecule, check if it matches any of the valid metabolites
            prediction_status = [any(map(lambda expec: validate_molecule_match(expec, pred), children))                 
                                for pred in predictions] 
            true_positives = prediction_status.count(True)
            correct_predictions.append(true_positives)
            false_negatives.append(len(children)-true_positives)
            
            
        else: 
            correct_predictions.append(0)
            false_negatives.append(len(children))
            print("No predictions")
    return correct_predictions, false_negatives, nr_children_per_parent, total_nr_predictions

################# CONTROL RESULT METHOD ##################
def precision(tp, fp): 
    if (tp + fp) == 0: 
        return 0
    return tp / (tp + fp)

#Recall is the fraction of the positive examples that were correctly labeled by the model as positive.
def recall(tp, fn): 
    if fn == 0: 
        return 0
    return tp / (tp + fn)

# Out of N guesses how many true positives where found
def get_topn_accuracy(correct, nr_children_per_parent): 
    return np.mean([cor/preds for (cor, preds) in zip(correct, nr_children_per_parent)])

# How many cases where atleast one metabolite was found 
def percentage_at_least_one_metabolite_per_parent(correct_predictions):
    at_least_one_correct = [pred >= 1 for pred in correct_predictions]
    if len(at_least_one_correct) == 0: 
        return 0
    return at_least_one_correct.count(True) / len(at_least_one_correct)

# How many cases where atleast half metabolite was found
def percentage_at_least_half_metabolites_per_parent(true_positives, possible_metabolites): 
    at_least_half = [pred / total > 0.5 for (pred, total) in zip(true_positives, possible_metabolites)]
    if len(at_least_half) == 0: 
        return 0
    return at_least_half.count(True) / len(at_least_half)

# How many cases where all metabolite was found
def percentage_all_metabolites_per_parent(correct_predictions, possible_metabolites): 
    all = [pred >= total for (pred, total) in zip(correct_predictions, possible_metabolites)]
    if len(all) == 0: 
        return 0
    return all.count(True) / len(all)

# Gets the number of true and false positives for predictions on all the parents
def get_true_and_false_positives(correct_predictions, false_negatives_list, total_predictions): 
    true_positives = sum(correct_predictions)
    false_negatives = sum(false_negatives_list)
    false_positives = sum(total_predictions) - true_positives
    return true_positives, false_positives, false_negatives

# The fingerprint similarity between valid predictions and metabolites 
def fingerprint_similarity_valid_smiles(top_n_preds, unique_parents):
    max_sims_valid = []
    max_sims_invalid = []
    for i, (_, childs) in enumerate(unique_parents.items()):
        valids = [ smiles for smiles in  top_n_preds[i] if (valid_smiles(smiles))]
        invalids = [ smiles for smiles in  top_n_preds[i] if not (valid_smiles(smiles))]
        if(valids != []):
            similarities = fingerprint_similarity_single(childs, valids)
            max_valid = max(similarities)
            max_sims_valid.append(max_valid)
        if(invalids != []):
            character_similarity = [SequenceMatcher(None ,invalid, child).ratio() for invalid, child in zip(childs, invalids)]
            max_invalid = max(character_similarity)
            max_sims_invalid.append(max_invalid)
    return max_sims_valid, max_sims_invalid

# Return the average fingerprint for a valid SMILES
def fingerprint_similarity_average(predicted, correct):  
    similarities = fingerprint_similarities_multiple(predicted, correct)
    avg_sims = []
    for sims in similarities: 
        if len(sims) != 0: 
            avg_sim = sum(sims) / len(sims)
            avg_sims.append(avg_sim)
    if len(avg_sims) == 0: 
        return 0, avg_sims
    total_avg_sim = sum(avg_sims) / len(avg_sims)
    return total_avg_sim, avg_sims

def average_number_of_predictions(predicted_lists): 
    if len(predicted_lists) == 0: 
        return 0
    return sum([len(preds) for preds in predicted_lists]) / len(predicted_lists)

# Returns the probability weight that a metabolite will be generated given a model 
def get_metabolite_probability(model, unique_parents, tokenizer, device):
    prob_of_metabolites = []
    for parent, childs in unique_parents.items():
        parent_tokens = torch.LongTensor([[1] + tokenizer.encode(parent) + [2] + [3]*120]).to(device)
        model_out_logits = model(input_ids=parent_tokens).logits[0]
        for child in childs:
            child_tokens = [1] + tokenizer.encode(child) + [2]
            log_likelihood = 0.0
            for i, token_id in enumerate(child_tokens):
                logits = model_out_logits[i]
                log_probs = F.log_softmax(logits, dim=-1)  
                log_prob = log_probs[token_id]
                log_likelihood += log_prob.item() 
            prob_of_metabolites.append([log_likelihood, child])
    return prob_of_metabolites

# Calculates validity given a prediction list of each parent. How many of the metabolites was valid vs the number of predictions 
def get_validity(preds_list): 
    valid_smiles_per_molecule = [[valid_smiles(pred) for pred in predictions].count(True) for predictions in preds_list]
    exist_valid_per_molecule = [valid_smiles >= 1 for valid_smiles in valid_smiles_per_molecule]
    validity = sum(exist_valid_per_molecule) / len(exist_valid_per_molecule)
    return validity, valid_smiles_per_molecule, sum(valid_smiles_per_molecule)

def get_average_fps_for_closest_prediction(predicted_list, parent_child_dict):
    best_average_sims = []
    for i, (parent, childs) in enumerate(parent_child_dict.items()):
        predictions = predicted_list[i]
        best_average_sims_for_preds = []
        for pred in predictions:
            character_similarities = [SequenceMatcher(None,child, pred).ratio() for child in childs]
            best_average_sims_for_preds.append(max(character_similarities))
        best_average_sims.append(max(best_average_sims_for_preds))
    return best_average_sims


# Runs a set of evaluations for a given prediction list. Calculates and stores the evalatiuon scores in data/evaluations/evaluations_runs.csv and is labeled by the "name_of_run"
def perform_evaluation_of_result(n_best, parent_child_dict, parents, predicted_list, name_of_run, beam_size = None):
    
    true_positives_per_molecule, false_positives_per_molecule, nr_children_per_molecule, total_nr_predictions = control_predictions_per_molecule(parent_child_dict, parents, predicted_list)

    topn_accuracy = get_topn_accuracy(true_positives_per_molecule, nr_children_per_molecule)

    validity, valid_smiles_per_molecule, total_nr_valid_smiles = get_validity(predicted_list)

    true_positives, false_positives, false_negatives = get_true_and_false_positives(true_positives_per_molecule, false_positives_per_molecule, total_nr_predictions)

    prec = precision(true_positives, false_positives)

    rec = recall(true_positives, false_negatives)

    per_one_metabolite = percentage_at_least_one_metabolite_per_parent(true_positives_per_molecule)

    per_half_metabolites = percentage_at_least_half_metabolites_per_parent(true_positives_per_molecule, nr_children_per_molecule)

    per_all_metabolites = percentage_all_metabolites_per_parent(true_positives_per_molecule, nr_children_per_molecule)

    total_avg_sim, avg_sims = fingerprint_similarity_average(parents, predicted_list)
   
    avg_out_size = average_number_of_predictions(predicted_list)

    average_fps_for_closest_predictions = get_average_fps_for_closest_prediction(predicted_list, parent_child_dict)
    #print(true_positives_per_molecule[:88])
    #print(nr_children_per_molecule[:88])
    #print(valid_smiles_per_molecule[:88])
    #print("_"*20)
    #print(parents[89])
    #print(true_positives_per_molecule[89:200])
    #print(nr_children_per_molecule[89:200])
    #print(valid_smiles_per_molecule[89:200])

    print("-"*80)
    print(average_fps_for_closest_predictions)
    print("average:", np.average(average_fps_for_closest_predictions))

    save_predictions(parents, parent_child_dict, predicted_list)
    save_true_predictions_only(parents, parent_child_dict, predicted_list, true_positives_per_molecule)
    save_evaluation(n_best, total_nr_valid_smiles, true_positives, false_positives, false_negatives, validity, valid_smiles_per_molecule, prec, rec, topn_accuracy, per_one_metabolite, per_half_metabolites, per_all_metabolites, avg_out_size, total_avg_sim, avg_sims, beam_size, name_of_run)
    return prec, rec, total_avg_sim