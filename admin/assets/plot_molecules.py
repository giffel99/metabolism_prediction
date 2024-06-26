import rdkit as rd
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd

def get_image_of_molecule(smiles, file_name): 
    file_name = "".join([file_name, ".png"])
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol, file_name)

def get_non_canonical_smiles(smiles): 
    mol = Chem.MolFromSmiles(smiles)
    k_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
    print(k_smiles)
    return k_smiles

def plot_dataset(): 
    data_set = "dataset"
    data_smiles = pd.read_csv(f'{data_set}.csv')
    parents = data_smiles["parent_smiles"].unique()
    metabolites = data_smiles["metabolite_smiles"].unique()
    for molecule in metabolites: 
        get_image_of_molecule(molecule, molecule)
    for molecule in parents: 
        get_image_of_molecule(molecule, molecule)

def plot_molecule(): 
    molecules = ["CC[C@H](C)[C@H](N)C(=O)O", "B(C(C)NC(=O)C(C(C)CC)N)(O)O"]
    for molecule in molecules: 
        get_image_of_molecule(molecule, molecule)

def main():
    #plot_dataset()
    plot_molecule()

if __name__ == "__main__":    
   main()
