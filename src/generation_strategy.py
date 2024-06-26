from functools import partial
import torch 
from torch import nn

#### GREEDY ####
def greedy_decode(model, source,tokenizer, max_len, device):
    encoder_output = model(source)
    _, predicted_indices = torch.max(encoder_output.logits, dim=-1)
    generated_tokens = tokenizer.decode( predicted_indices.flatten(), skip_special_tokens=True)  
    generated_string = ''.join(generated_tokens)
    return [generated_string]


#### BEAM SEARCH ####

def step_in_beam_search(input_ids, log_prob, prediction, stopping_criterion, tokenizer, beam_size, device): 
    _, vocab_size = prediction.shape
    _, n_tokens = input_ids.shape 

    log_prob = log_prob.unsqueeze(-1)

    # check if beam is finished
    is_finished = stopping_criterion(input_ids)
    is_not_finished = ~is_finished

    # Select the beams that are finished and unfinished, respectively.
    finished_ids = input_ids[is_finished].to(device)
    unfinished_ids = input_ids[is_not_finished].to(device)
    n_unfinished = unfinished_ids.shape[0] # the amount of unfinished beams

    # current probabilites for finished and unfinished beams
    finished_log_probs = log_prob[is_finished]
    unfinished_log_probs = log_prob[is_not_finished]

    finished_log_probs = finished_log_probs.to(device)
    unfinished_log_probs = unfinished_log_probs.to(device)
    prediction = prediction.to(device)

    log_probs_beams_expanded = torch.add(unfinished_log_probs, prediction[n_tokens, :])
    log_probs_beams_expanded = log_probs_beams_expanded.to(device)

    assert(log_probs_beams_expanded.shape == torch.Size([n_unfinished, vocab_size]))
    
    expanded_sorted = log_probs_beams_expanded.flatten().sort(descending=True)
    assert(expanded_sorted.values.shape == torch.Size([n_unfinished*vocab_size]))
    
    next_unfinished_idx = 0
    next_finished_idx = 0
    
    beams = []
    new_cum_log_probs = []

    # If we select the finished beams, we will have to add some padding.
    padding = torch.tensor([tokenizer.pad_token_id]).to(device)

    for i in range(beam_size): 
        if next_finished_idx >= finished_log_probs.shape[0] \
            or expanded_sorted.values[next_unfinished_idx] > finished_log_probs[next_finished_idx]:
            # Select the next best unfinished beam:

            # Index among the unfinished beams of the highest-scoring candidate.
            seq_idx = torch.div(expanded_sorted.indices[next_unfinished_idx], prediction.shape[-1], rounding_mode="floor")

            # Compute the index in the vocabulary of the highest-scoring candidate.
            next_token = expanded_sorted.indices[next_unfinished_idx] % prediction.shape[-1]
            next_token = next_token.to(device)
            # Tensor next_beam where the next token id is added to 
            # to the corresponding beam from the previous step.
            next_beam = torch.cat((unfinished_ids[seq_idx],next_token[None]))
            assert(next_beam.shape == torch.Size([n_tokens+1]))

            log_probs_for_beam = expanded_sorted.values[next_unfinished_idx]

            next_unfinished_idx += 1
        else:
            # We select the next best previously finished beam:

            # Tensor next_beam where paddings is added to the
            # beam from the previous step.
            next_beam = torch.cat((finished_ids[next_finished_idx],padding))
            assert(next_beam.shape == torch.Size([n_tokens+1]))
            log_probs_for_beam = finished_log_probs[next_finished_idx]
            next_finished_idx += 1

        # Add the current beam to the list of selected beams.
        beams.append(next_beam)  
        new_cum_log_probs.append(log_probs_for_beam)

    input_ids = torch.stack(beams)
    log_prob = torch.tensor(new_cum_log_probs)

    return input_ids, log_prob

def sequence_ends_with_eos(input_ids, tokenizer):
    output_tensor = torch.zeros(len(input_ids), dtype=torch.bool)
    for i, input_id in enumerate(input_ids): 
        last_token = input_id[-1]
        # Should pad be an stopping criterion?
        if last_token == tokenizer.eos_token_id or last_token == tokenizer.pad_token_id:
            output_tensor[i] = True
    return output_tensor

# TODO: implementera i batches
def beam_search(model, tokenizer, beam_size, n_best, source, device, max_length=100, print_output=True):
    
    stopping_criterion = partial(sequence_ends_with_eos, tokenizer=tokenizer)
    softmax = nn.Softmax(dim=1)
    model_output = model(source) #.to(device)
    prediction = model_output.logits

    # batch_size will be 1, remove batch size generation
    batch_size, _, _ = prediction.shape
    assert(batch_size == 1)
    prediction = prediction[0, :, :]
    prediction = softmax(prediction)

    # predict first row
    log_prob, input_ids = prediction[0, :].topk(beam_size, sorted=True) 
    input_ids = input_ids.unsqueeze(-1)

    # take steps in beam search as long as all beams are not done
    while not torch.all(stopping_criterion(input_ids)) and input_ids.shape[1] < max_length: 
        input_ids, log_prob = step_in_beam_search(input_ids, log_prob, prediction, stopping_criterion, tokenizer, beam_size, device)
    # pick the n_best ones
    input_ids = input_ids[:n_best]
    log_prob = log_prob[:n_best]
    # generate SMILES from tokens
    result = []
    for input_id in input_ids: 
        generated_tokens = tokenizer.decode(input_id, skip_special_tokens=True)
        generated_string = ''.join(generated_tokens)
        result.append(generated_string)

    list(set(result))
    return result, log_prob.tolist()


def beam_search_ensemble(models, tokenizer, beam_size, n_best, src_input, device, max_len, print_output): 
    smiles_out = []
    for model in models: 
        model_out_smiles, log_prob = beam_search(model, tokenizer, beam_size, n_best, src_input, device, max_len, print_output)

        smiles_out = smiles_out + list(zip(log_prob, model_out_smiles))
    
    smiles_out.sort(reverse=True) 
    probs, smiles = list(zip(*smiles_out))  
    return list(set(smiles))[:n_best], probs[:n_best]