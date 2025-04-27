# Clean dataset
import os
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


file_path = "C:/Users/Asus/Desktop"
file_name = "8irs.xlsx"
input_file_path = os.path.join(file_path, file_name)
dataframe = pd.read_excel(input_file_path)

def validate_smiles (smiles_list):
    """
    Validate SMILES strings and returns a list of valid SMILES

    """
    valid_smiles = []
    for smiles in tqdm(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
        except Exception as e:
            print(f"Error validationg SMILES {smiles}:{e}")
    return valid_smiles

def clean_dataset (dataframe):
    """
    Cleans the dataframe by validating SMILES strings and creating graphs.

    """
    valid_smiles = validate_smiles(dataframe["SMILES"])
    cleaned_df = dataframe[dataframe["SMILES"].isin(valid_smiles)].reset_index(drop=True)
    cleaned_df.to_excel("Cleaned_"+file_name, index = False)
    return cleaned_df

cleaned_df = clean_dataset(dataframe)

