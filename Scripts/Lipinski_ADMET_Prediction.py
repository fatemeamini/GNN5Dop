import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen

# Function to calculate ADMET and Lipinski properties
def calculate_admet_lipinski(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ['Invalid SMILES'] * 12

        # Calculate ADMET properties
        ext_ppb = Descriptors.MolWt(mol) / 500  # Example Ext PPB (arbitrary normalization)
        alogp98 = Crippen.MolLogP(mol)  # AlogP98 using RDKit's Crippen module
        psa_2d = Descriptors.TPSA(mol)  # PSA-2D (Topological Polar Surface Area)
        bbb = Crippen.MolLogP(mol) - (psa_2d / 100)  # Simplified BBB penetration score
        absorption = 100 - psa_2d  # Simplified absorption score (arbitrary calculation)

        # Calculate Lipinski's Rule of Five Properties
        mw = Descriptors.MolWt(mol)  # Molecular Weight
        logp = Crippen.MolLogP(mol)  # LogP
        hbd = Descriptors.NumHDonors(mol)  # Hydrogen Bond Donors
        hba = Descriptors.NumHAcceptors(mol)  # Hydrogen Bond Acceptors

        # Lipinski criteria checks
        lipinski_mw_pass = mw <= 500
        lipinski_logp_pass = logp <= 5
        lipinski_hbd_pass = hbd <= 5
        lipinski_hba_pass = hba <= 10
        lipinski_pass = lipinski_mw_pass and lipinski_logp_pass and lipinski_hbd_pass and lipinski_hba_pass

        # ADMET Pass Criteria
        admet_pass = (ext_ppb >= 0.8) and (alogp98 <= 5) and (psa_2d <= 140) and (bbb >= 0.1) and (absorption >= 70)

    

        return [
            round(ext_ppb, 2), round(alogp98, 2), round(psa_2d, 2), round(bbb, 2), round(absorption, 2),  # ADMET
            round(mw, 2), round(logp, 2), hbd, hba,  # Lipinski properties
            
        ]

    except Exception as e:
        return ['Error'] * 12

# Load input data
input_file = "C:/Users/Asus/Desktop/75.ligs.xlsx"   # Replace with your input file name
output_file = 'admet_lipinski_full_results.another.code5.xlsx'  # Replace with your desired output file name

# Load DataFrame with headers: LIGAND, SMILES, Ki, pKi
data = pd.read_excel(input_file)

# Apply ADMET and Lipinski calculations to each row
data[['Ext PPB', 'AlogP98', 'PSA-2D', 'BBB', 'Absorption',
      'Molecular Weight', 'LogP', 'HBD', 'HBA']] = data.apply(
    lambda row: calculate_admet_lipinski(row['SMILES']), axis=1, result_type='expand'
)


# Save the styled DataFrame to an Excel file
data.to_excel(output_file, index=False, engine='openpyxl')
print(f"ADMET and Lipinski analysis saved to {output_file}")