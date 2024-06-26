#The output of the ensemble is the union of the predictions from each model.
#More specically, we ne-tuned multiple models varying the model hyperparameters and the SMILES augmentation process (more details in ESI: S1.1†). The
#selection process of the models to form the ensemble was based on nding a trade-off between maximizing the number of correctly identied metabolites while keeping the output size,
#which is an indication of false positives, low. The resulting ensemble model consists of 6 ne-tuned models. The output of the ensemble is the union of the 
# sets of predicted metabolites from each individual model

# remake eval so that how the result is gathered is given 
from transformers.modeling_outputs import CausalLMOutputWithPast

def ensemble_model(models, source): 
    models_logits = []
    for model in models: 
        logits = model(source)
        print(logits)
        models_logits = models_logits + logits

    return CausalLMOutputWithPast(
        loss=None,
        logits=models_logits,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
    )