import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import GroupShuffleSplit 

############# Help functions ########
def arrange_molecules_with_all_metabolites(data): 
    parent_child_smiles_dict = {}

    for parent in data["parent_smiles"].unique(): 
        children = data.loc[data['parent_smiles'] == parent]["metabolite_smiles"]
        parent_child_smiles_dict[parent] = list(children)
    return parent_child_smiles_dict

def get_unique_parent_dataset(data): 
    unique_parent_dataset = []
    
    for parent in data["parent_smiles"].unique(): 
        children = data.loc[data['parent_smiles'] == parent]["metabolite_smiles"]
        unique_parent_dataset.append((parent, list(children)[0]))

    unique_parent_df = pd.DataFrame(unique_parent_dataset)
    unique_parent_df.columns = ["parent_smiles", "metabolite_smiles"]
    return unique_parent_df

################# Split functions ##########################      
# Split data based on parents, with identical parents in the same split
def data_split_parent(data, train_val_split): 
    val_size = 1 - train_val_split
    
    splitter = GroupShuffleSplit(test_size=val_size, n_splits=2)
    split = splitter.split(data, groups=data['parent_smiles'])
    train_inds, test_inds = next(split)

    train_subset = Subset(data.to_numpy(), train_inds)
    val_subset = Subset(data.to_numpy(), test_inds)
    
    return train_subset, val_subset, train_inds, test_inds

def random_data_split(data_smiles, train_val_split):
    # Here we can change the train/val split in the dataset. We can also add the split ratio to the config.
    train_ds_size = int(train_val_split * len(data_smiles))
    val_ds_size = len(data_smiles) - train_ds_size
    return random_split(data_smiles,[train_ds_size, val_ds_size])

def create_eval_dataset(df, inds, size = 100): 
    if inds.shape[0] < size: 
        size = inds.shape[0]
    eval_inds = random.sample(list(inds), size)
    eval_dataset = df.iloc[eval_inds]
    eval_ds_raw = Subset(df.to_numpy(), eval_inds)

    eval_unique_parents = arrange_molecules_with_all_metabolites(eval_dataset)
    return eval_ds_raw, eval_unique_parents

##############################################################
#Exposed function the rest of the modules. fetches the data, creates new Dataset objects for train and val, creates dataloaders for val and train.
def get_train_val_dataloaders(tokenizer, train_val_split, seq_len, batch_size, data_set = "processed_data/metxbiodb_smiles_small", pretraining = False): 
    
    # load dataset
    data_smiles_df = pd.read_csv(f'dataset/{data_set}.csv')
    data_smiles_df = data_smiles_df[["parent_smiles", "metabolite_smiles", ]] 
    data_smiles = data_smiles_df.to_numpy()[1:]    

    train_ds_raw, val_ds_raw, train_inds, test_inds = data_split_parent(data_smiles_df, train_val_split) 

    val_eval_ds_raw, val_eval_unique_parents = create_eval_dataset(data_smiles_df, test_inds)
    train_eval_ds_raw, train_eval_unique_parents = create_eval_dataset(data_smiles_df, test_inds)


    # Here we create the Dataset objects. The input should be a matrix/df of our data. 
    train_ds = MetaboliteDataset(train_ds_raw, tokenizer, seq_len) 
    val_ds = MetaboliteDataset(val_ds_raw, tokenizer, seq_len)
    val_eval_ds = MetaboliteDataset(val_eval_ds_raw, tokenizer, seq_len)
    train_eval_ds = MetaboliteDataset(train_eval_ds_raw, tokenizer, seq_len)

    max_len_src = 0
    max_len_tgt = 0
    
    for item in data_smiles:
        src_tokens = tokenizer.tokenize(item[0])
        tgt_tokens = tokenizer.tokenize(item[1])
        max_len_src = max(max_len_src, len(src_tokens))
        max_len_tgt = max(max_len_tgt, len(tgt_tokens))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Dataloaders. this handles the batches and is what gets input to the training loop.  
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    val_eval_dataloader = DataLoader(val_eval_ds, batch_size=1, shuffle=False)
    train_eval_dataloader = DataLoader(train_eval_ds, batch_size=1, shuffle=False)
    return train_dataloader, val_dataloader, val_eval_dataloader, val_eval_unique_parents, train_eval_dataloader, train_eval_unique_parents

def get_eval_dataloader(tokenizer, data_set, seq_len): 
    # load dataset
    data_smiles_df = pd.read_csv(f'dataset/{data_set}.csv')
    data_smiles_df = data_smiles_df[["parent_smiles", "metabolite_smiles", ]]
    unique_df = get_unique_parent_dataset(data_smiles_df)
    data_smiles = unique_df.to_numpy() # parent_smiles must be first column in matrix
    # Here we create the Dataset objects. The input should be a matrix/df of our data. 
    eval_ds = MetaboliteDataset(data_smiles, tokenizer, seq_len) 

    max_len_src = 0
    max_len_tgt = 0

    for item in data_smiles:
        src_tokens = tokenizer.tokenize(item[0]) 
        tgt_tokens = tokenizer.tokenize(item[1])
        max_len_src = max(max_len_src, len(src_tokens))
        max_len_tgt = max(max_len_tgt, len(tgt_tokens))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Dataloaders. this handles the batches
    # should evaluation be done in another way?
    eval_dataloader = DataLoader(eval_ds, batch_size=1, shuffle=False)
    return data_smiles_df, eval_dataloader


########### Dataset class ###########
# Used to create the dataset class that the Dataloaders use. Uses the tokenizer and raw dataset file to create an iterable dataset object. 
class MetaboliteDataset(Dataset):
    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer = tokenizer
        self.bos_token = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.eos_token_id], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.pad_token_id], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,idx):
        
        src_target_pair = self.ds[idx]
        src_smiles = src_target_pair[0]
        tgt_smiles = src_target_pair[1]

        # Transform smiles into token ids        
        src_input_tokens = self.tokenizer.encode(src_smiles)
        tgt_input_tokens = self.tokenizer.encode(tgt_smiles)
        
        # Add bos, eos, padding
        src_num_padding_tokens = self.seq_len - len(src_input_tokens) - 2
        tgt_num_padding_tokens = self.seq_len - len(tgt_input_tokens) - 2

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if src_num_padding_tokens < 0 or tgt_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # input of the deccoder. Starts with a bos token, then input tokens, padding tokens, and lastly eos token.
        src_input = torch.cat(
            [
                self.bos_token,
                torch.tensor(src_input_tokens, dtype=torch.int64),  
                self.eos_token,
                torch.tensor([self.pad_token] * src_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Labels, starts with input, followed by eos, lastly the padding.
        label = torch.cat(
            [   
                self.bos_token,
                torch.tensor(tgt_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * tgt_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        attention_mask = (label != self.tokenizer.pad_token_id).long() # Masks padding tokens to ensure model do not take those into consideration

        # Double check the size of the tensors to make sure they are all seq_len long
        assert src_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # returns a dictionary of the following items.
        return {
        "src_input": src_input, # (seq_len) 
        "attention_mask": attention_mask, # (seq_len) 
        "label": label,  # (seq_len)
        "src_smiles": src_smiles,
        "tgt_smiles": tgt_smiles,
    }
