from utils import valid_smiles, molecule_allowed_based_on_weight, atoms_allowed_in_molecules

def filtering_of_data(data, filter_method): 
    valid = list(filter(filter_method, data))
    removed = len(data) - len(valid)
    return valid, removed

def postprocess_data(data, print_output): 
    data, invalid_smiles = filtering_of_data(data, valid_smiles)
    data, invalid_weight = filtering_of_data(data, molecule_allowed_based_on_weight)
    data, invalid_atoms = filtering_of_data(data, atoms_allowed_in_molecules)
    if print_output: 
        total_removed = invalid_smiles + invalid_atoms + invalid_weight 
        print("Number of removed ones: " + str(total_removed))
        print(f"Division is (Smiles, atoms, weight): ({invalid_smiles}, {invalid_atoms}, {invalid_weight})")
    return data

