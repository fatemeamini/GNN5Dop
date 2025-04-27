#!/usr/bin/env python3
"""
Protein Ligand Binding Affinity Downloader

This GUI tool fetches binding affinity and structural information from several online
sources, including BindingDB, RCSB, UniProt, NCBI, and ZINC (via ChEMBL). The BindingDB routine
has been updated to first retrieve UniProt IDs via PDBe and then query the new
getLindsByUniprots endpoint. If data are not found, the method falls back to additional endpoints.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import requests
import logging
import tkinter.font as tkFont
import json
import urllib.parse
import time
import os
from datetime import datetime
import re
from io import StringIO

# Set up logging to file and console
log_folder = "logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_filename = os.path.join(log_folder, f"protein_data_fetcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class ProteinDataFetcher:
    def __init__(self):
        self.user_agent = "ProteinDataFetcher/1.0 (academic research tool)"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json"
        })

    def fetch_binding_db_data(self, pdb_code):
        logging.info(f"Fetching BindingDB data for {pdb_code}")
        all_ligand_data = []

        # METHOD A: Try using UniProt mapping from PDBe first
        uniprot_ids = []
        pdbe_mapping_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code.lower()}"
        try:
            logging.debug(f"Requesting PDBe UniProt mapping: {pdbe_mapping_url}")
            mapping_response = self.session.get(pdbe_mapping_url, timeout=30)
            mapping_response.raise_for_status()
            mapping_data = mapping_response.json()
            if pdb_code.lower() in mapping_data:
                uniprot_mappings = mapping_data[pdb_code.lower()].get("UniProt", {})
                for uid in uniprot_mappings.keys():
                    if uid not in uniprot_ids:
                        uniprot_ids.append(uid)
                logging.debug(f"Found UniProt IDs for {pdb_code}: {uniprot_ids}")
        except Exception as e:
            logging.error(f"Error retrieving UniProt mapping for {pdb_code}: {e}")

        # If we found UniProt IDs, try the new BindingDB endpoint
        if uniprot_ids:
            uniprot_str = ",".join(uniprot_ids)
            bindingdb_url = f"https://bindingdb.org/rest/getLigandsByUniprots?uniprot={uniprot_str}&cutoff=10000&response=application/json"
            logging.debug(f"Using BindingDB UniProt endpoint: {bindingdb_url}")
            try:
                response = self.session.get(bindingdb_url, timeout=300)
                logging.debug(f"Response status (uniProt query): {response.status_code}")
                if response.status_code == 200:
                    try:
                        json_data = response.json()
                        # Note the endpoint returns key "getLindsByUniprotsResponse" and nested "affinities"
                        if ("getLindsByUniprotsResponse" in json_data and
                            "affinities" in json_data["getLindsByUniprotsResponse"]):
                            affinities = json_data["getLindsByUniprotsResponse"]["affinities"]
                            logging.debug(f"Found {len(affinities)} affinity records from BindingDB (uniProt)")
                            for item in affinities:
                                ligand_info = {
                                    "Database": "BindingDB",
                                    "PDB ID": pdb_code,
                                    "UniProt": item.get("query", ""),  # UniProt ID from which the affinity was obtained
                                    "Ligand ID": item.get("monomerid", ""),
                                    "SMILES": item.get("smile", ""),
                                    "Affinity Type": item.get("affinity_type", ""),
                                    "Affinity Value": item.get("affinity", ""),
                                    "PubMed ID": item.get("pmid", ""),
                                    "DOI": item.get("doi", ""),
                                    "BindingDB URL": f"https://www.bindingdb.org/bind/chemsearch/marvin/MolStructure.jsp?monomerid={item.get('monomerid', '')}"
                                }
                                all_ligand_data.append(ligand_info)
                        else:
                            logging.warning("The uniProt endpoint did not return the expected 'getLindsByUniprotsResponse' structure.")
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to decode JSON from uniProt endpoint: {e}")
                        logging.debug(f"Response text (first 500 chars): {response.text[:500]}")
            except Exception as e:
                logging.error(f"Error fetching BindingDB data via uniProt endpoint: {e}")

        # If the new endpoint did not return any records, fall back to alternative methods.
        if not all_ligand_data:
            fallback_urls = [
                f"https://www.bindingdb.org/rwd/rest/target/pdbids/{pdb_code.lower()}/ligands",
                f"https://www.bindingdb.org/rwd/rest/search/pdbcode/{pdb_code.lower()}",
                f"https://www.bindingdb.org/rwd/rest/target_search/PDB%20ID/{pdb_code.upper()}"
            ]
            for url in fallback_urls:
                logging.debug(f"Trying fallback BindingDB URL: {url}")
                try:
                    response = self.session.get(url, timeout=30)
                    logging.debug(f"Response status: {response.status_code}")
                    if response.status_code == 200:
                        try:
                            json_data = response.json()
                            # Case where JSON returns a list of ligands
                            if isinstance(json_data, list):
                                for item in json_data:
                                    if isinstance(item, dict):
                                        ligand_info = {
                                            "Database": "BindingDB",
                                            "PDB ID": pdb_code,
                                            "Ligand ID": item.get("ligandId", ""),
                                            "SMILES": item.get("smiles", ""),
                                            "Affinity Type": item.get("affinityType", ""),
                                            "Affinity Value": item.get("affinityValue", ""),
                                            "Affinity Units": item.get("affinityUnits", ""),
                                            "PubMed ID": item.get("pubmedId", ""),
                                            "BindingDB URL": f"https://www.bindingdb.org/bind/browsecitation.jsp?citation_id={item.get('citationId', '')}"
                                        }
                                        all_ligand_data.append(ligand_info)
                            # Case where JSON has a dict with a "result" key containing ligands
                            elif isinstance(json_data, dict) and "result" in json_data and isinstance(json_data["result"], list):
                                for result in json_data["result"]:
                                    if "ligands" in result and isinstance(result["ligands"], list):
                                        for ligand in result["ligands"]:
                                            ligand_info = {
                                                "Database": "BindingDB",
                                                "PDB ID": pdb_code,
                                                "Ligand ID": ligand.get("ligandId", ""),
                                                "SMILES": ligand.get("smiles", ""),
                                                "Affinity Type": ligand.get("affinityType", ""),
                                                "Affinity Value": ligand.get("affinityValue", ""),
                                                "Affinity Units": ligand.get("affinityUnits", ""),
                                                "PubMed ID": result.get("pubmedId", ""),
                                                "BindingDB URL": f"https://www.bindingdb.org/bind/browsecitation.jsp?citation_id={result.get('citationId', '')}"
                                            }
                                            all_ligand_data.append(ligand_info)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to decode JSON from fallback URL: {e}")
                            logging.debug(f"Response text (first 500 chars): {response.text[:500]}")
                    if all_ligand_data:
                        logging.info(f"Found {len(all_ligand_data)} ligand records via fallback endpoints for {pdb_code}")
                        break
                except Exception as e:
                    logging.error(f"Error fetching from fallback BindingDB URL {url}: {e}")

        # Next: If still no data, try scraping an HTML page
        if not all_ligand_data:
            try:
                html_url = f"https://www.bindingdb.org/jsp/dbsearch/PrimarySearch_ki.jsp?energyterm=kJ/mole&tag=r21&pdbids_all=on&pdbids={pdb_code}&submit=Search"
                logging.debug(f"Trying BindingDB HTML scraping: {html_url}")
                response = self.session.get(html_url, timeout=30)
                if response.status_code == 200:
                    monomer_pattern = r"viewMonomer\.jsp\?monomerid=(\d+)"
                    monomer_ids = re.findall(monomer_pattern, response.text)
                    if monomer_ids:
                        logging.debug(f"Found {len(monomer_ids)} monomer IDs via HTML scraping")
                        for monomer_id in monomer_ids[:10]:
                            ligand_info = {
                                "Database": "BindingDB",
                                "PDB ID": pdb_code,
                                "Ligand ID": monomer_id,
                                "SMILES": "HTML extraction – see URL",
                                "Affinity Type": "See URL",
                                "Affinity Value": "See URL",
                                "BindingDB URL": f"https://www.bindingdb.org/bind/chemsearch/marvin/MolStructure.jsp?monomerid={monomer_id}"
                            }
                            all_ligand_data.append(ligand_info)
            except Exception as e:
                logging.error(f"Error during BindingDB HTML scraping: {e}")

        # Finally: If no data so far, try a protein name lookup as a last resort
        if not all_ligand_data:
            protein_mapping = {
                "PCSK9": "3P5B",
            }
            if pdb_code in protein_mapping:
                protein_name = protein_mapping[pdb_code]
                encoded_name = urllib.parse.quote(protein_name)
                protein_url = f"https://www.bindingdb.org/rwd/rest/target/name/{encoded_name}/ligands"
                try:
                    logging.debug(f"Trying BindingDB protein name lookup: {protein_name}")
                    response = self.session.get(protein_url, timeout=30)
                    if response.status_code == 200:
                        try:
                            json_data = response.json()
                            if isinstance(json_data, list):
                                for item in json_data:
                                    ligand_info = {
                                        "Database": "BindingDB",
                                        "PDB ID": pdb_code,
                                        "Protein": protein_name,
                                        "Ligand ID": item.get("ligandId", ""),
                                        "SMILES": item.get("smiles", ""),
                                        "Affinity Type": item.get("affinityType", ""),
                                        "Affinity Value": item.get("affinityValue", ""),
                                        "Affinity Units": item.get("affinityUnits", ""),
                                        "BindingDB URL": f"https://www.bindingdb.org/bind/chemsearch/marvin/MolStructure.jsp?monomerid={item.get('ligandId', '')}"
                                    }
                                    all_ligand_data.append(ligand_info)
                        except json.JSONDecodeError as e:
                            logging.warning("Failed to decode JSON from protein name lookup")
                except Exception as e:
                    logging.error(f"Error fetching BindingDB data by protein name: {e}")

        if not all_ligand_data:
            logging.warning(f"No BindingDB data found for {pdb_code}")
            return [{
                "Database": "BindingDB",
                "PDB ID": pdb_code,
                "Result": "No binding data found",
                "Note": "Check https://www.bindingdb.org/rwd/bind/search.jsp for manual search"
            }]

        return all_ligand_data

    def fetch_rcsb_data(self, pdb_code):
        logging.info(f"Fetching RCSB data for {pdb_code}")
        try:
            entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_code}"
            logging.debug(f"Requesting RCSB entry data: {entry_url}")
            entry_response = self.session.get(entry_url, timeout=30)
            entry_response.raise_for_status()
            entry_data = entry_response.json()

            ligand_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_code}"
            logging.debug(f"Requesting RCSB ligand data: {ligand_url}")
            try:
                ligand_response = self.session.get(ligand_url, timeout=30)
                ligand_response.raise_for_status()
                ligand_data = ligand_response.json()
                has_ligand_data = True
            except Exception as e:
                logging.warning(f"No specific ligand data found for {pdb_code} at RCSB: {e}")
                ligand_data = {}
                has_ligand_data = False

            assembly_url = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_code}/1"
            logging.debug(f"Requesting RCSB assembly data: {assembly_url}")
            try:
                assembly_response = self.session.get(assembly_url, timeout=30)
                assembly_response.raise_for_status()
                assembly_data = assembly_response.json()
            except Exception as e:
                logging.warning(f"No assembly data found for {pdb_code} at RCSB: {e}")
                assembly_data = {}

            result = {
                "Database": "RCSB",
                "PDB ID": pdb_code,
                "Title": entry_data.get("struct", {}).get("title", ""),
                "Method": entry_data.get("exptl", [{}])[0].get("method", "") if entry_data.get("exptl") else "",
                "Resolution": entry_data.get("rcsb_entry_info", {}).get("resolution_combined", [0])[0]
                              if isinstance(entry_data.get("rcsb_entry_info", {}).get("resolution_combined", []), list) else 0,
                "Deposition Date": entry_data.get("rcsb_accession_info", {}).get("deposit_date", ""),
                "RCSB URL": f"https://www.rcsb.org/structure/{pdb_code}"
            }
            results = []
            if has_ligand_data:
                nonpolymer_entities = ligand_data.get("nonpolymer_entities", [])
                if nonpolymer_entities:
                    for entity in nonpolymer_entities:
                        entity_id = entity.get("rcsb_nonpolymer_entity", {}).get("pdbx_entity_id")
                        entity_info = result.copy()
                        entity_info.update({
                            "Ligand ID": entity_id,
                            "Ligand Name": entity.get("rcsb_nonpolymer_entity", {}).get("pdbx_description", ""),
                            "Ligand Formula": entity.get("rcsb_nonpolymer_entity", {}).get("formula_weight", ""),
                            "Ligand SMILES": entity.get("rcsb_nonpolymer_entity", {}).get("pdbx_ccd_compnd_info", {}).get("smiles", "")
                        })
                        results.append(entity_info)
                else:
                    result["Ligand Info"] = "No specific ligand entities found"
                    results.append(result)
            else:
                result["Ligand Info"] = "No ligand data available from RCSB"
                results.append(result)
            return results
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching RCSB data for {pdb_code}: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error with RCSB data for {pdb_code}: {e}")
        except Exception as e:
            logging.error(f"Error processing RCSB data for {pdb_code}: {e}")
        return [{
            "Database": "RCSB",
            "PDB ID": pdb_code,
            "Result": "Error retrieving data",
            "RCSB URL": f"https://www.rcsb.org/structure/{pdb_code}"
        }]

    def fetch_uniprot_data(self, pdb_code):
        logging.info(f"Fetching UniProt data for {pdb_code}")
        try:
            mapping_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code.lower()}"
            logging.debug(f"Requesting UniProt mapping: {mapping_url}")
            mapping_response = self.session.get(mapping_url, timeout=30)
            mapping_response.raise_for_status()
            mapping_data = mapping_response.json()
            results = []
            if pdb_code.lower() in mapping_data:
                uniprot_mappings = mapping_data[pdb_code.lower()].get("UniProt", {})
                if uniprot_mappings:
                    for uniprot_id in uniprot_mappings:
                        try:
                            uniprot_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
                            logging.debug(f"Requesting detailed UniProt data for: {uniprot_id}")
                            uniprot_response = self.session.get(uniprot_url, timeout=30)
                            uniprot_response.raise_for_status()
                            uniprot_data = uniprot_response.json()
                            protein_name = ""
                            if "proteinDescription" in uniprot_data:
                                if "recommendedName" in uniprot_data["proteinDescription"]:
                                    protein_name = uniprot_data["proteinDescription"]["recommendedName"].get("fullName", {}).get("value", "")
                            gene_names = []
                            if "genes" in uniprot_data:
                                for gene in uniprot_data["genes"]:
                                    if "geneName" in gene:
                                        gene_names.append(gene["geneName"].get("value", ""))
                            organism = ""
                            if "organism" in uniprot_data:
                                organism = uniprot_data["organism"].get("scientificName", "")
                            function = ""
                            if "comments" in uniprot_data:
                                for comment in uniprot_data["comments"]:
                                    if comment.get("commentType") == "FUNCTION":
                                        if "texts" in comment:
                                            function = comment["texts"][0].get("value", "")
                                            break
                            result = {
                                "Database": "UniProt",
                                "PDB ID": pdb_code,
                                "UniProt ID": uniprot_id,
                                "Protein Name": protein_name,
                                "Gene Names": ", ".join(gene_names),
                                "Organism": organism,
                                "Function": (function[:500] + "..." if len(function) > 500 else function),
                                "UniProt URL": f"https://www.uniprot.org/uniprotkb/{uniprot_id}/entry"
                            }
                            results.append(result)
                        except Exception as e:
                            logging.error(f"Error fetching detailed UniProt data for {uniprot_id}: {e}")
                            results.append({
                                "Database": "UniProt",
                                "PDB ID": pdb_code,
                                "UniProt ID": uniprot_id,
                                "Result": f"Error retrieving detailed data: {e}",
                                "UniProt URL": f"https://www.uniprot.org/uniprotkb/{uniprot_id}/entry"
                            })
                else:
                    logging.warning(f"No UniProt mappings found for {pdb_code}")
                    results.append({
                        "Database": "UniProt",
                        "PDB ID": pdb_code,
                        "Result": "No UniProt mappings found"
                    })
            else:
                logging.warning(f"PDB ID {pdb_code} not found in UniProt mappings")
                results.append({
                    "Database": "UniProt",
                    "PDB ID": pdb_code,
                    "Result": "PDB ID not found in UniProt mappings"
                })
            return results
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching UniProt data for {pdb_code}: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error with UniProt data for {pdb_code}: {e}")
        except Exception as e:
            logging.error(f"Error processing UniProt data for {pdb_code}: {e}")
        return [{
            "Database": "UniProt",
            "PDB ID": pdb_code,
            "Result": "Error retrieving data"
        }]

    def fetch_ncbi_data(self, pdb_code):
        logging.info(f"Fetching NCBI data for {pdb_code}")
        try:
            citation_url = f"https://data.rcsb.org/rest/v1/core/citation/{pdb_code}"
            logging.debug(f"Requesting RCSB citation data: {citation_url}")
            citation_response = self.session.get(citation_url, timeout=30)
            citation_response.raise_for_status()
            citation_data = citation_response.json()
            results = []
            if "citations" in citation_data:
                for citation in citation_data["citations"]:
                    if ("rcsb_citation" in citation and
                        "pdbx_database_id_PubMed" in citation["rcsb_citation"]):
                        pmid = citation["rcsb_citation"]["pdbx_database_id_PubMed"]
                        try:
                            pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
                            logging.debug(f"Requesting PubMed data for: {pmid}")
                            pubmed_response = self.session.get(pubmed_url, timeout=30)
                            pubmed_response.raise_for_status()
                            pubmed_data = pubmed_response.json()
                            if "result" in pubmed_data and pmid in pubmed_data["result"]:
                                article_data = pubmed_data["result"][pmid]
                                result = {
                                    "Database": "NCBI",
                                    "PDB ID": pdb_code,
                                    "PubMed ID": pmid,
                                    "Title": article_data.get("title", ""),
                                    "Journal": article_data.get("fulljournalname", ""),
                                    "Publication Date": article_data.get("pubdate", ""),
                                    "Authors": ", ".join(article_data.get("authors", [])),
                                    "DOI": citation["rcsb_citation"].get("pdbx_database_id_DOI", ""),
                                    "PubMed URL": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                }
                                results.append(result)
                            else:
                                logging.warning(f"Could not find PubMed data for ID {pmid}")
                                results.append({
                                    "Database": "NCBI",
                                    "PDB ID": pdb_code,
                                    "PubMed ID": pmid,
                                    "Result": "PubMed data not available",
                                    "PubMed URL": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                })
                        except Exception as e:
                            logging.error(f"Error fetching PubMed data for {pmid}: {e}")
                            results.append({
                                "Database": "NCBI",
                                "PDB ID": pdb_code,
                                "PubMed ID": pmid,
                                "Result": f"Error retrieving PubMed data: {e}",
                                "PubMed URL": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            })
                    else:
                        results.append({
                            "Database": "NCBI",
                            "PDB ID": pdb_code,
                            "Title": citation.get("rcsb_citation", {}).get("title", ""),
                            "Journal": citation.get("rcsb_citation", {}).get("journal_abbrev", ""),
                            "Year": citation.get("rcsb_citation", {}).get("year", ""),
                            "Result": "No PubMed ID available"
                        })
            if not results:
                logging.warning(f"No citation data found for {pdb_code}")
                results.append({
                    "Database": "NCBI",
                    "PDB ID": pdb_code,
                    "Result": "No citation data found"
                })
            return results
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching NCBI data for {pdb_code}: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error with NCBI data for {pdb_code}: {e}")
        except Exception as e:
            logging.error(f"Error processing NCBI data for {pdb_code}: {e}")
        return [{
            "Database": "NCBI",
            "PDB ID": pdb_code,
            "Result": "Error retrieving data"
        }]

    def fetch_zinc_data(self, pdb_code):
        logging.info(f"Fetching ZINC data for {pdb_code}")
        try:
            ligand_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_code}"
            logging.debug(f"Requesting RCSB data for ZINC lookup: {ligand_url}")
            ligand_response = self.session.get(ligand_url, timeout=30)
            ligand_response.raise_for_status()
            ligand_data = ligand_response.json()

            ligand_ids = []
            if "rcsb_binding_sites" in ligand_data:
                for site in ligand_data["rcsb_binding_sites"]:
                    if "comp_id" in site:
                        ligand_ids.append(site["comp_id"])
            if "rcsb_nonpolymer_entity_container_identifiers" in ligand_data:
                for entity in ligand_data["rcsb_nonpolymer_entity_container_identifiers"]:
                    if "auth_asym_ids" in entity:
                        for auth_id in entity["auth_asym_ids"]:
                            if auth_id not in ligand_ids:
                                ligand_ids.append(auth_id)
            results = []
            if ligand_ids:
                for ligand_id in ligand_ids:
                    try:
                        chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_structures__canonical_smiles__flexmatch={ligand_id}"
                        logging.debug(f"Searching for ligand {ligand_id} in ChEMBL: {chembl_url}")
                        chembl_response = self.session.get(chembl_url, timeout=30)
                        if chembl_response.status_code == 200:
                            chembl_data = chembl_response.json()
                            if "molecules" in chembl_data and len(chembl_data["molecules"]) > 0:
                                molecule = chembl_data["molecules"][0]
                                result = {
                                    "Database": "ZINC (via ChEMBL)",
                                    "PDB ID": pdb_code,
                                    "Ligand ID": ligand_id,
                                    "ChEMBL ID": molecule.get("molecule_chembl_id", ""),
                                    "Name": molecule.get("pref_name", ""),
                                    "SMILES": molecule.get("molecule_structures", {}).get("canonical_smiles", ""),
                                    "Molecular Formula": molecule.get("molecule_properties", {}).get("full_molformula", ""),
                                    "Molecular Weight": molecule.get("molecule_properties", {}).get("full_mwt", ""),
                                    "ChEMBL URL": f"https://www.ebi.ac.uk/chembl/compound_report_card/{molecule.get('molecule_chembl_id', '')}"
                                }
                                results.append(result)
                            else:
                                result = {
                                    "Database": "ZINC (Not Found)",
                                    "PDB ID": pdb_code,
                                    "Ligand ID": ligand_id,
                                    "Result": "No matching data found in ChEMBL/ZINC",
                                    "Note": "Check manually at https://zinc.docking.org/"
                                }
                                results.append(result)
                        else:
                            logging.warning(f"ChEMBL search failed for ligand {ligand_id} with status {chembl_response.status_code}")
                            result = {
                                "Database": "ZINC",
                                "PDB ID": pdb_code,
                                "Ligand ID": ligand_id,
                                "Result": f"ChEMBL search failed with status {chembl_response.status_code}"
                            }
                            results.append(result)
                    except Exception as e:
                        logging.error(f"Error searching ZINC/ChEMBL for ligand {ligand_id}: {e}")
                        result = {
                            "Database": "ZINC",
                            "PDB ID": pdb_code,
                            "Ligand ID": ligand_id,
                            "Result": f"Error during search: {e}"
                        }
                        results.append(result)
            if not results:
                try:
                    ccd_url = f"https://data.rcsb.org/graphql?query={{chemical_components(comp_id:\"{pdb_code}\"){{chem_comp{{id,type,name,formula,inchi,inchikey}}}}}}"
                    ccd_response = self.session.get(ccd_url, timeout=30)
                    if ccd_response.status_code == 200:
                        ccd_data = ccd_response.json()
                        if ("data" in ccd_data and "chemical_components" in ccd_data["data"] and
                            ccd_data["data"]["chemical_components"]):
                            for comp in ccd_data["data"]["chemical_components"]:
                                if "chem_comp" in comp:
                                    result = {
                                        "Database": "ZINC (via RCSB CCD)",
                                        "PDB ID": pdb_code,
                                        "Ligand ID": comp["chem_comp"].get("id", ""),
                                        "Name": comp["chem_comp"].get("name", ""),
                                        "Formula": comp["chem_comp"].get("formula", ""),
                                        "InChI": comp["chem_comp"].get("inchi", ""),
                                        "InChIKey": comp["chem_comp"].get("inchikey", ""),
                                        "RCSB URL": f"https://www.rcsb.org/ligand/{comp['chem_comp'].get('id', '')}"
                                    }
                                    results.append(result)
                except Exception as e:
                        logging.error(f"Error checking CCD for {pdb_code}: {e}")
            if not results:
                logging.warning(f"No ligand information found for {pdb_code}")
                results.append({
                    "Database": "ZINC",
                    "PDB ID": pdb_code,
                    "Result": "No ligand information found"
                })
            return results
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching ZINC data for {pdb_code}: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error with ZINC data for {pdb_code}: {e}")
        except Exception as e:
            logging.error(f"Error processing ZINC data for {pdb_code}: {e}")
        return [{
            "Database": "ZINC",
            "PDB ID": pdb_code,
            "Result": "Error retrieving data"
        }]

def collect_data(data_fetcher, protein, db_selections, progress_callback=None):
    all_data = []
    pdb_codes = {
        "PCSK9": ["3P5B"]
    }
    api_functions = {
        "BindingDB": data_fetcher.fetch_binding_db_data,
        "RCSB": data_fetcher.fetch_rcsb_data,
        "UniProt": data_fetcher.fetch_uniprot_data,
        "NCBI": data_fetcher.fetch_ncbi_data,
        "ZINC": data_fetcher.fetch_zinc_data,
    }
    total_operations = len(pdb_codes.get(protein, [])) * len(db_selections)
    completed = 0
    for pdb_code in pdb_codes.get(protein, []):
        protein_data = {"Protein": protein, "PDB ID": pdb_code}
        for database in db_selections:
            logging.info(f"Collecting data for protein {protein} (PDB: {pdb_code}) from database {database}")
            if progress_callback:
                progress_callback(completed / total_operations * 100, f"Fetching {database} data for {protein} (PDB: {pdb_code})")
            try:
                results = api_functions[database](pdb_code)
                for result in results:
                    data_entry = protein_data.copy()
                    data_entry.update(result)
                    all_data.append(data_entry)
            except Exception as e:
                logging.error(f"Error collecting data from {database} for protein {protein}: {e}")
                data_entry = protein_data.copy()
                data_entry.update({
                    "Database": database,
                    "Result": f"Error: {e}"
                })
                all_data.append(data_entry)
            completed += 1
            if progress_callback:
                progress_callback(completed / total_operations * 100)
    return all_data

# GUI Application
class ProteinDataApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Protein Ligand Binding Affinity Downloader")
        self.root.geometry("600x500")
        self.data_fetcher = ProteinDataFetcher()
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=12)
        self.root.option_add("*TCombobox*Listbox*Font", default_font)
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.db_frame = ttk.LabelFrame(self.main_frame, text="Databases", padding="10")
        self.db_frame.pack(fill=tk.X, pady=5)
        self.db_options = ["BindingDB", "RCSB", "UniProt", "NCBI", "ZINC"]
        self.db_vars = {db: tk.BooleanVar(value=True) for db in self.db_options}
        for i, db in enumerate(self.db_options):
            check = ttk.Checkbutton(self.db_frame, text=db, variable=self.db_vars[db])
            check.grid(row=0, column=i, padx=10)
        self.protein_frame = ttk.LabelFrame(self.main_frame, text="Protein Classes", padding="10")
        self.protein_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.protein_options = ["PCSK9"]
        self.protein_vars = {protein: tk.BooleanVar() for protein in self.protein_options}
        self.canvas = tk.Canvas(self.protein_frame)
        self.scrollbar = ttk.Scrollbar(self.protein_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        for i, protein in enumerate(self.protein_options):
            row, col = divmod(i, 2)
            check = ttk.Checkbutton(self.scrollable_frame, text=protein, variable=self.protein_vars[protein])
            check.grid(row=row, column=col, sticky="w", padx=10, pady=5)
        self.button_frame = ttk.Frame(self.main_frame, padding="10")
        self.button_frame.pack(fill=tk.X, pady=10)
        self.select_all_button = ttk.Button(self.button_frame, text="Select All Proteins",
                                              command=lambda: self.select_all_proteins(True))
        self.select_all_button.pack(side=tk.LEFT, padx=5)
        self.deselect_all_button = ttk.Button(self.button_frame, text="Deselect All Proteins",
                                                command=lambda: self.select_all_proteins(False))
        self.deselect_all_button.pack(side=tk.LEFT, padx=5)
        self.download_button = ttk.Button(self.button_frame, text="Download Data", command=self.download_data)
        self.download_button.pack(side=tk.RIGHT, padx=5)
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.pack(fill=tk.X, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=100,
                                        mode="determinate", variable=self.progress_var)
        self.progress.pack(fill=tk.X)
    
    def select_all_proteins(self, select=True):
        for var in self.protein_vars.values():
            var.set(select)
    
    def update_progress(self, value, message=None):
        self.progress_var.set(value)
        if message:
            self.status_var.set(message)
        self.root.update_idletasks()
    
    def download_data(self):
        selected_proteins = [protein for protein, var in self.protein_vars.items() if var.get()]
        selected_dbs = [db for db, var in self.db_vars.items() if var.get()]
        if not selected_proteins:
            messagebox.showwarning("Selection Error", "Please select at least one protein class.")
            return
        if not selected_dbs:
            messagebox.showwarning("Selection Error", "Please select at least one database.")
            return
        self.download_button.config(state=tk.DISABLED)
        self.status_var.set("Downloading data...")
        all_results = []
        try:
            for protein in selected_proteins:
                data = collect_data(self.data_fetcher, protein, selected_dbs, self.update_progress)
                all_results.extend(data)
            if all_results:
                df = pd.DataFrame(all_results)
                filetypes = [("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
                file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=filetypes)
                if file_path:
                    if file_path.endswith(".csv"):
                        df.to_csv(file_path, index=False)
                    else:
                        df.to_excel(file_path, index=False)
                    if messagebox.askyesno("Open File", "Data saved successfully. Do you want to open it?"):
                        os.startfile(file_path)
            else:
                messagebox.showinfo("No Data", "No data was retrieved.")
        except Exception as e:
            logging.error(f"Error during data download: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.download_button.config(state=tk.NORMAL)
            self.status_var.set("Ready")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProteinDataApp(root)
    root.mainloop()
