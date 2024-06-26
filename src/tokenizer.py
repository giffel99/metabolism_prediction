import pandas as pd
from transformers import LlamaTokenizerFast
from tokenizers import SentencePieceBPETokenizer


# Creates a new tokinzer and trains it on a an input dataset.
# it saves the trained tokenizer in a json format and is saved in a specified folder. 
# Run this method anywhere to build a new tokenizer. 
def train_and_save_tokenizer():
    
    data_smiles = pd.read_csv(f'smiles_tokenizer/vocab.csv')
    data_smiles = data_smiles.to_numpy()        # Create a SentencePieceBPETokenizer
    tokenizer = SentencePieceBPETokenizer(add_prefix_space = False)
    
    # Train the SentencePieceBPETokenizer on the dataset
    
    tokenizer.train_from_iterator(
        iterator=data_smiles,
        vocab_size=40,        
        min_frequency = 100,
        show_progress=True,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    )
    
    # Save the tokenizer
    tokenizer.save("smiles_tokenizer/sentencepiece_tokenizer.json", pretty=True)
    # Load the new tokenizer as a LlamaTokenizerFast
    new_llama_tokenizer = LlamaTokenizerFast(
        tokenizer_file="smiles_tokenizer/sentencepiece_tokenizer.json",
        unk_token="<unk>",
        unk_token_id=0,
        bos_token="<s>",
        bos_token_id=1,
        eos_token="</s>",
        eos_token_id=2,
        pad_token="<pad>",
        pad_token_id=3,
        padding_side="right")
    
    new_llama_tokenizer.add_tokens(['Br','Cl','Si'])
  
    # Save the new tokenizer
    new_llama_tokenizer.save_pretrained("smiles_tokenizer")

if __name__ == "__main__":
    train_and_save_tokenizer()
