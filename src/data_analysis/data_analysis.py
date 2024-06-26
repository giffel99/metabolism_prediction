import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
import numpy as np
#from preprocessing import compare_dataset_remove_duplicates

colors = ['#b7ded2', '#f6a6b2', '#f7c297', '#90d2d8', '#ffecb8', '#30afba', '#c6b0cc', '#90b8ab']
colors = ['#4a2377', '#f55f74', '#8cc5e3', '#0d7d87', '#ffecb8', '#30afba', '#c6b0cc', '#90b8ab']

def get_dataset(data_set): 
    return pd.read_csv(f'dataset/{data_set}.csv')    

#---------------PLOTTING------------------------------
def create_pychart(data, labels, name): 
    
     # Creating plot
    fig = plt.figure(figsize=(10,5))
    wedges2, texts, autotexts = plt.pie(data, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 14})
    plt.legend(wedges2, labels,
        title="Mediating enzymes",
        loc="center left",
        bbox_to_anchor=(0.8, 0, 0, 0))
    plt.savefig(f"../admin/plots/{name}.png")

def create_pychart_double(data1, label1, data2, label2, name1, name2, title): 
    
     # Creating plot
    fig, axs = plt.subplots(1, 2, figsize=(14,7))

    wedges, texts, autotexts = axs[0].pie(data1, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 14})
    axs[0].set_title(name1, fontdict={'fontsize': 18, 'fontweight': 'medium'})
    axs[0].legend(wedges, label1,
        title="Mediating enzymes",
        loc="center left",
        bbox_to_anchor=(0.8, 0, 0, 0))

    wedges2, texts, autotexts = axs[1].pie(data2, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 14})
    axs[1].set_title(name2, fontdict={'fontsize': 18, 'fontweight': 'medium'})
    axs[1].legend(wedges2, label2,
        title="Mediating enzymes",
        loc="center left",
        bbox_to_anchor=(0.8, 0, 0, 0))

    plt.savefig(f"../admin/plots/{title}.png")

def create_pychart_hist(dist, n_bins, name, xlabel, ylabel): 
    fig = plt.figure()

    plt.hist(dist, bins=n_bins, color='#90d2d8')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(f"../admin/plots/{name}.png")

def create_pychart_double_hist(dist1, label1, dist2, label2, n_bins, title, xlabel, ylabel): 
    fig, axs = plt.subplots(1, 2, sharey=True)

    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)

    axs[0].hist(dist1, bins=n_bins, color='#90d2d8')
    axs[0].set_title(label1, fontdict={'fontsize': 14, 'fontweight': 'medium'})

    axs[1].hist(dist2, bins=n_bins, color='#90d2d8')
    axs[1].set_title(label2, fontdict={'fontsize': 14, 'fontweight': 'medium'})

    plt.savefig(f"../admin/plots/{title}.png")

#----------------------DATA ANALYSIS---------------------------------------------------

#---------------------ENZYMES---------------------------------------------------------

# Compares datasets for duplicate data points
def get_duplicates_from_datasets(dataset_first, dataset_second):
    equal_count = 0
    for i, row_db in dataset_first.iterrows():
        parent = row_db['parent_smiles']
        child = row_db['metabolite_smiles']

        duplicated = dataset_second[(dataset_second["parent_smiles"] == parent) & (dataset_second["metabolite_smiles"] == child)]

        if not duplicated.empty: 
            equal_count += 1 
            
    return equal_count

def data_source_analysis(dataset_first, dataset_first_name, dataset_second, dataset_second_name): 
    dataset_first = dataset_first[['parent_smiles', 'metabolite_smiles']]
    dataset_second = dataset_second[['parent_smiles', 'metabolite_smiles']]
    duplicate_count = get_duplicates_from_datasets(dataset_first, dataset_second)
    print(duplicate_count)

    first_dataset_length = len(dataset_first) - duplicate_count
    second_dataset_length = len(dataset_second) - duplicate_count

    labels = [dataset_first_name, dataset_second_name, f"{dataset_first_name} and {dataset_second_name}"]
    data = [first_dataset_length, second_dataset_length, duplicate_count]
    
    create_pychart(data, labels, "data_sources")

def curate_metabolic_for_enzyme_analysis(dataset): 
    before = dataset.shape[0]
    after = dataset["enzymes"].isna().sum() 
    #print(before)
    ##print(after)
    #print(after/before)
    enzymes = dataset.fillna(value="None")["enzymes"]

    enzymes = [list(map(lambda x: x.strip(),  row.split(";"))) for row in enzymes]
    enzymes = list(chain(*enzymes))
    before = len(enzymes)
    #print(before)
    enzymes = [enzyme for enzyme in enzymes if enzyme != "None"]
    after = len(enzymes)
    #print(len(enzymes))
    
    enzymes = [enzyme.lower() for enzyme in enzymes]
    enzymes = curate_metxbiodb_for_enzyme_analysis(enzymes)
    enzymes = curate_drugbank_for_enzyme_analysis(enzymes)
    return enzymes
    
def curate_metxbiodb_for_enzyme_analysis(enzymes): 
    enzymes = ["CYP" if ("cyp" in enzyme) else enzyme for enzyme in enzymes]
    enzymes = ["NAT" if ("nat" in enzyme) else enzyme for enzyme in enzymes]
    enzymes = ["UGT" if ("ugt" in enzyme) else enzyme for enzyme in enzymes]
    enzymes = ["GST" if ("gst" in enzyme) else enzyme for enzyme in enzymes]
    enzymes = ["UDP" if ("udp" in enzyme) else enzyme for enzyme in enzymes]
    enzymes = ["SULT" if ("sult" in enzyme) else enzyme for enzyme in enzymes]
    enzymes = ["COMT" if ("comt" in enzyme) else enzyme for enzyme in enzymes]   
    return enzymes 

def curate_drugbank_for_enzyme_analysis(enzymes):
    enzymes = ["CYP" if ("cytochrome" in enzyme) else enzyme for enzyme in enzymes]
    enzymes = ["Other oxidoreductases" if (("prostaglandin g/h synthase")  in enzyme or "oxidoreductases" in enzyme or ("nadp" in enzyme) or ("nad(p)h" in enzyme)) else enzyme for enzyme in enzymes] 
    enzymes = ["SULT" if (("sulfotransferase")  in enzyme) else enzyme for enzyme in enzymes] 
    enzymes = ["GST" if (("glutathione s-transferase")  in enzyme) else enzyme for enzyme in enzymes] 
    enzymes = ["MPO" if (("myeloperoxidase")  in enzyme) else enzyme for enzyme in enzymes] 
    enzymes = ["Other hydrolases" if (("esterase"  in enzyme) or "acetyltransferase" in enzyme or "carboxylic ester hydrolase" in enzyme) else enzyme for enzyme in enzymes]   

    enzymes = ["UDP" if enzyme.startswith("udp") else enzyme for enzyme in enzymes]
    enzymes = ["NADPH" if enzyme.startswith("carbonyl reductase") else enzyme for enzyme in enzymes]
    enzymes = ["NADPH" if (("nadp" in enzyme) or ("nad(p)h" in enzyme)) else enzyme for enzyme in enzymes]

    enzymes = ["COMT" if (("catechol o-methyltransferase")  in enzyme) else enzyme for enzyme in enzymes] 
  
    enzymes = ["XDH" if (("xanthine dehydrogenase")  in enzyme) else enzyme for enzyme in enzymes]
    return enzymes

def count_enzymes(enzymes): 
    family_count = {}
    for enzyme in set(enzymes): #set gives unique data automatically 
        nr_of_enzyme = [em in enzyme for em in enzymes].count(True)
        family_count.update({enzyme: nr_of_enzyme})
    return family_count

def divide_enzymes_into_families(family_count, limit): 
    result = {"Others": 0}
    rest = {}

    for (name, count) in family_count.items(): 
        if count > limit: 
            result.update({name: count})
        else: 
            rest.update({name: count})
            result.update({"Others": (result["Others"] + count)})
    return result, rest


def enzyme_analysis(enzymes, name, limit): 
    # create list of lists 
    family_count, rest = divide_enzymes_into_families(count_enzymes(enzymes), limit)
    labels1 = family_count.keys()
    data1 = family_count.values()
    
    
    #create_pychart(data1, labels1, "enzymes_" + name)
    print(rest)
    print("Next step")
    fam_count, _ = divide_enzymes_into_families(rest, 20)
   # print(rest)
    print(fam_count)

    labels2 = fam_count.keys()
    data2 = fam_count.values()
    
   # create_pychart(data2, labels2, "enzymes_rest_" + name)

    create_pychart_double(data1, labels1, data2, labels2, "", "", "enzymes_split_" + name)

def enzyme_analysis_double(enzymes1, enzymes2, name1, name2, limit1, limit2, title): 
    print("first_step")
    family_count, _ = divide_enzymes_into_families(count_enzymes(enzymes1), limit1)
    print(family_count)
    family_count = dict(sorted(family_count.items(), key=lambda x:x[1], reverse=True))
    label1 = list(family_count.keys())
    data1 = list(family_count.values())

    family_count, _ = divide_enzymes_into_families(count_enzymes(enzymes2), limit2)
    family_count = dict(sorted(family_count.items(), key=lambda x:x[1], reverse=True))

    labels2 = family_count.keys()
    data2 = family_count.values()

    create_pychart_double(data1, label1, data2, labels2, name1, name2, "enzymes_" + title)

#------------------------MOLECULAR WEIGHT----------------------------

def molecular_weight_analysis(dataset, name): 
    metab_weights, diff_weights = [], []

    parents = dataset["parent_smiles"]
    metabolites = dataset["metabolite_smiles"]
    for parent, metabolite in zip(parents, metabolites): 
        metabolite_mol_weight = Descriptors.ExactMolWt(Chem.MolFromSmiles(metabolite))
        metab_weights.append(metabolite_mol_weight)

        parent_mol_weight = Descriptors.ExactMolWt(Chem.MolFromSmiles(parent))
        diff_mol_weight = abs(parent_mol_weight - metabolite_mol_weight)
        diff_weights.append(diff_mol_weight)
    
    unique_parents = get_unique_parent_dataset(dataset)["parent_smiles"]
    parent_weights = [Descriptors.ExactMolWt(Chem.MolFromSmiles(parent)) for parent in unique_parents]
   
    create_pychart_double_hist(parent_weights, "Parents", metab_weights, "Metabolites", 50, "molecular_weights_" + name, "Molecular weights (Da)", "Counts")
    create_pychart_hist(diff_weights, 40, "molecular_weights_diff_" + name, "Molecular weight difference (Da)",  "Counts")
    
#---------------------------------Similarity----------------------------------------
def parent_child_similarity(dataset, name): 
    par = dataset["parent_smiles"]
    met = dataset["metabolite_smiles"]
    parents = [Chem.MolFromSmiles(smiles) for smiles in dataset["parent_smiles"]]
    metabolites = [Chem.MolFromSmiles(smiles) for smiles in dataset["metabolite_smiles"]]

    fpgen = AllChem.GetRDKitFPGenerator()
    fps_parents = [fpgen.GetFingerprint(x) for x in parents]
    fps_metabolites = [fpgen.GetFingerprint(x) for x in metabolites]
    similarities = []
    for i, (fpp, fpm) in enumerate(zip(fps_parents, fps_metabolites)): 
        sim = DataStructs.TanimotoSimilarity(fpp, fpm)
        
        similarities.append(sim)
    avg = sum(similarities) / len(similarities)
    print(f"Fingerprint similarity average: {avg}")
    print("similarities" + name)

    create_pychart_hist(similarities, 50, "similarities" + name, "Fingerprint similarity", "Counts")
    
def get_nr_unique_parent_dataset(data): 
    unique_parent_dataset = get_unique_parent_dataset(data)
    return len(unique_parent_dataset)

def get_unique_parent_dataset(data): 
    unique_parent_dataset = []
    
    for parent in data["parent_smiles"].unique(): 
        children = data.loc[data['parent_smiles'] == parent]["metabolite_smiles"]
        unique_parent_dataset.append((parent, list(children)[0]))

    unique_parent_df = pd.DataFrame(unique_parent_dataset)
    unique_parent_df.columns = ["parent_smiles", "metabolite_smiles"]
    return unique_parent_df

def main():
    drugbank = get_dataset("train/drugbank_smiles")
    metxbiodb = get_dataset("train/metxbiodb_smiles")
    metabolic = get_dataset("train/metabolic_smiles")
    #mmp = get_dataset("train/matched_molecular_pairs")
    #nr_of_datapoints = len(mmp)
    #print(f"Datapoints: {nr_of_datapoints}")

    gloryx = get_dataset("test/gloryx_smiles_first_generation")
    test_metabolic = get_dataset("test/metabolic_smiles")
    #drugbank_no_duplicates = compare_dataset_remove_duplicates(drugbank, metxbiodb)
    
    #UNRELEVANT RIGHT NOW
    #unique_parents_nr = get_nr_unique_parent_dataset(gloryx)
    #print(f'Number of unique parents: {unique_parents_nr}')
    
    #data_source_analysis(drugbank, "DrugBank", metxbiodb, "MetXBioDB")
    #nr_of_datapoints = len(metabolic)
    #print(f"Datapoints: {nr_of_datapoints}")
    all_metabolic = pd.concat([metabolic, test_metabolic])
    #molecular_weight_analysis(metabolic, "metabolic")
    #molecular_weight_analysis(test_metabolic, "test_metabolic")    
    molecular_weight_analysis(all_metabolic, "all_metabolic")    
    parent_child_similarity(all_metabolic,"all_metabolic")

    # enzyme analysis
    #print(metabolic)
    enzymes_metabolic = curate_metabolic_for_enzyme_analysis(metabolic)
    enzymes_test_metabolic = curate_metabolic_for_enzyme_analysis(test_metabolic)
    enzyme_analysis_double(enzymes_metabolic, enzymes_test_metabolic, "Metabolic train/val set", "Metabolic test set", 44,  10, "metabolic_train_test")

    

##################### ALL BELOW IS NOT USED ######################


    #print(enzymes_metabolic)
    # #enzymes_metxbiodb = curate_metxbiodb_for_enzyme_analysis(metxbiodb)
    # #enzymes_drugbank = curate_drugbank_for_enzyme_analysis(drugbank)
    # #enzyme_analysis_double(enzymes_metxbiodb, enzymes_drugbank, "MetXBioDB", "DrugBank", 25, 10)
    # #enzyme_analysis(enzymes_metxbiodb, "metxbiodb", 25)
    # #enzyme_analysis(enzymes_drugbank, "drugbank", 10)
    # #enzyme_analysis(enzymes_metxbiodb + enzymes_drugbank, "dataset", 30)

    # similarities

    # #parent_child_similarity(metxbiodb,"metxbiodb")
    # #parent_child_similarity(drugbank,"drugbank")


if __name__ == "__main__":    
    main()


