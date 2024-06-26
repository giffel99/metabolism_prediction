## README FOR DRUGBANK CURATION

Hi and welcome to this readme file. :)

## The curate_drugbank.py will run all relevant drugbank datasources, parse them, and finally map them together into one CSV file.

### To run the script you must be in the folder /src and run "py curate_databases/curate_drugbank.py"

# Datasources used and where to find them

drugbank_filepath = https://go.drugbank.com/releases/latest#full
drugbank_metabolite_structures_filepath = https://go.drugbank.com/releases/latest#structures (Full Metabolite Structures in SDF Format)
drugbank_drug_structures = https://go.drugbank.com/releases/latest#structures (Full Drug Structures in SDF Format)
drugbank_external_drugs = https://go.drugbank.com/releases/latest#structures (Structure External Links)
hmdb_metabolites = https://hmdb.ca/downloads (All Metabolites)

---

# Conditionals

## These are flags one can set allowing one to specify what functions to run

    get_reaction_pairs = True
    get_drug_metabolite_info = True
    parse_and_remove_bad_data = True
    get_external_drugs = True
    get_hdmb_drugs = True
    combine_the_data = True

# Methods

---

# These can be run all together but easier if run one at a time.

drugbank_filepath = "dataset/raw_data/drugbank_full_database.xml"
parsed_pair_name_output = "dataset/raw_data/drugbank_pairs_names.csv"

# Get the reaction pairs "parent and child" along with enzymes, name, etc...

parse_and_save_metabolite(drugbank_filepath,parsed_pair_name_output,-1 )

# get the drug & metabolite information

---

## Metabolites information

drugbank_metabolite_structures_filepath = "dataset/raw_data/drugbank_metabolite_structures.sdf"
parsed_metabolite_structures_output = "dataset/raw_data/drugbank_metabolite_structures.csv"

## Function that will read the SDF file and parse the metabolite information

read_sdf_file(drugbank_metabolite_structures_filepath, parsed_metabolite_structures_output, -1)

## Function that removes bad rows from the parsed list

## remove_bad_drug_metabolite_rows(parsed_metabolite_structures_output)

## Drug (Parent) information

drugbank_drug_structures = "dataset/raw_data/drugbank_drug_structures.sdf"
parsed_drug_structures_output = "dataset/raw_data/drugbank_drug_structures.csv"

## Function that will read the SDF file and parse the drug (parent) information

read_sdf_file(drugbank_drug_structures, parsed_drug_structures_output, -1, True, 2057177 )

## Function that removes bad rows from the parsed list

## remove_bad_drug_metabolite_rows(parsed_drug_structures_output)

## external drugs information not present in the other drug/metabolite files

drugbank_external_drugs = "dataset/raw_data/drugbank_external_structures.csv"
drugbank_external_drugs_cleaned = "dataset/raw_data/drugbank_external_structures_cleaned.csv"

## This data is easy extracted so only these steps were needed

db_external_df = pd.read_csv(drugbank_external_drugs)
db_external_df = db_external_df[["dbid","smiles","inchi_key",'name']]
db_external_df.to_csv(drugbank_external_drugs_cleaned,index=False)

## Function that removes bad rows from the parsed list

## remove_bad_drug_metabolite_rows(drugbank_external_drugs_cleaned)

## HMDB drug and metabolites information

hmdb_metabolites = "dataset/raw_data/hmdb_metabolites.xml"
hmdb_cleaned = "dataset/raw_data/hmdb_cleaned.csv"

## Maps the MHDB data into a csv file

hmdb_metabolite_map(hmdb_metabolites,hmdb_cleaned, -1 )

## Function that removes bad rows from the parsed list

## remove_bad_drug_metabolite_rows(hmdb_cleaned)

## Combine the drug metabolite into one file

## The resulting information files we have are:

### Drug, metabolite, hmdb, external

drugbank_full_structures = "dataset/raw_data/drugbank_full_structures.csv"
combine_datasets(parsed_metabolite_structures_output, parsed_drug_structures_output, drugbank_full_structures)
combine_datasets(drugbank_full_structures, drugbank_external_drugs_cleaned, drugbank_full_structures)
combine_datasets(drugbank_full_structures, hmdb_cleaned, drugbank_full_structures)

---

# map reaction pairs to smiles and inchi

## This will go over the reaction (parent child) and map their id or names to get their smiles and inchi

parent_child_smiles_output = "dataset/raw_data/drugbank_pairs_smiles.csv"
parent_child_error_map = "dataset/raw_data/drugbank_error_map.csv"
combine_id_with_smiles_inchi(parsed_pair_name_output, drugbank_full_structures, parent_child_smiles_output,parent_child_error_map)

## Check hmdb for those pairs who could not be mapped

parent_child_smiles_output_hmdb = "dataset/raw_data/drugbank_pairs_smiles_hmdb.csv"
parent_child_error_map_hmdb = "dataset/raw_data/drugbank_error_map_hmdb.csv"
parent_child_smiles_finalized = "dataset/raw_data/drugbank_pairs_smiles_finalized.csv"
combine_id_with_smiles_inchi(parent_child_error_map, hmdb_cleaned, parent_child_smiles_output_hmdb,parent_child_error_map_hmdb)

## Finally the first parent/child smiles gets combined with those the HMDB could map

combine_datasets(parent_child_smiles_output, parent_child_smiles_output_hmdb, parent_child_smiles_finalized)

## Now your done. The result will be a raw csv file with all reaction pairs containing smiles and inchi.

## This data can now go through the preprocessing pipeline to remove duplicates, make them cannonicalised, etc...
