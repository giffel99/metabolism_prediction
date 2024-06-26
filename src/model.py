
import torch
from focused_transformer.configuration_longllama import LongLlamaConfig
from focused_transformer.modeling_longllama import LongLlamaForCausalLM
from transformers import LlamaTokenizerFast


# Inputs the config file
# Outputs a new Transformer model
# This method configures the start of a new FoT.
def build_focused_transformer(tokenizer, vocab_size, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, max_position_embeddings):
    configuration = LongLlamaConfig(
        vocab_size=vocab_size, 
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id
        )
    model = LongLlamaForCausalLM(configuration)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model total params:", pytorch_total_params)
    return model


# Builds tokenizer
def build_tokenizer():
    tokenizer:LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained('smiles_tokenizer/')
    tokenizer.add_bos_token = False
    return tokenizer

def get_model(tokenizer, vocab_size, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, max_position_embeddings):
    
    model: LongLlamaForCausalLM = build_focused_transformer(tokenizer, vocab_size, hidden_size, num_hidden_layers, intermediate_size,
                                                            num_attention_heads, max_position_embeddings) # add all newcessary configurations we need to build and get the model
    return model

def preload_model(model_filename, model, optimizer, device): 
    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename, map_location = device)
    model.load_state_dict(state['model_state_dict'])
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
    return model, initial_epoch, optimizer, global_step

if __name__=='__main__':
    build_tokenizer()
    