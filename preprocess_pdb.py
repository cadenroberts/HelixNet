#!/usr/bin/env python3

import openmm
from openmm import *
import openmm.app as app
import numpy as np
import pdbfixer
import requests
import json
import os
import shutil
from rdkit import Chem
from rdkit.Chem import AllChem

from openff.toolkit import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator

import requests
import urllib.parse
import sys

def get_non_water_atom_indexes(topology):
    return np.array([a.index for a in topology.atoms() if a.residue.name != 'HOH'])

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            os.makedirs(f"{folder_path}/raw")
            os.makedirs(f"{folder_path}/processed")
            print(f"Folder created: {folder_path}")
        except OSError as e:
            print(f"Error: Unable to create folder {folder_path}. {e}")
    else:
        print(f"Folder already exists: {folder_path}")

def get_atomSubset(pdb_path=str):
    proteinResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR', 'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL']
    
    pdb = PDBFile(pdb_path)

    atomSubset = []
    topology = pdb.getTopology()
    for atom in topology.atoms():
        if atom.residue.name in proteinResidues:
            atomSubset.append(atom.index)
    
    return atomSubset

def get_rcsb_ligand_smiles(comp_id):
    try:
        return get_rcsb_ligand_smiles_exc(comp_id)
    except Exception as e:
        print("Error in RCSB query")
        print(e)
    return None

def get_rcsb_ligand_smiles_exc(comp_id):
    comp_id = str(comp_id)
    if len(comp_id) != 3:
        raise RuntimeError("Invalid comp_id {comp_id}, must be a 3 letter string.")
    
    query_string = "{chem_comp(comp_id:\"" + comp_id + "\"){chem_comp{id,name,formula},rcsb_chem_comp_descriptor{SMILES,SMILES_stereo}}}"
    query_url = "https://data.rcsb.org/graphql?query=" + urllib.parse.quote(query_string)

    response = requests.get(query_url)
    response.raise_for_status()

    return response.json()["data"]["chem_comp"]["rcsb_chem_comp_descriptor"]["SMILES_stereo"]

def replace_ligands(pdb_filename, modeller, smiles_templates=True):
    pdb_mol = Chem.rdmolfiles.MolFromPDBFile(pdb_filename, removeHs=False, proximityBonding=True)

    standardResidues = {"ALA", "ARG", "ASN", "ASP", "ASX", "CYS", "GLU", "GLN", "GLX", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
    standardResidues.add("HOH")

    fragments = dict()
    small_molecules_seen = dict()

    for frag_idx, frag in enumerate(Chem.rdmolops.GetMolFrags(pdb_mol, asMols=True)):
        a = frag.GetAtomWithIdx(0)
        r_name = a.GetPDBResidueInfo().GetResidueName()

        if frag.GetNumAtoms() == 1:
            continue
        if r_name in standardResidues:
            continue
        r_id = a.GetPDBResidueInfo().GetResidueNumber()
        r_chain = a.GetPDBResidueInfo().GetChainId()

        is_alone = True
        for a in frag.GetAtoms():
            if r_id != a.GetPDBResidueInfo().GetResidueNumber():
                is_alone = False
                break
        if is_alone:
            rcsb_smiles = get_rcsb_ligand_smiles(r_name)
            if rcsb_smiles is None:
                print(f"Could not find template for {r_name}")
                continue
            template = Chem.MolFromSmiles(rcsb_smiles)
            if smiles_templates:
                small_molecules_seen[r_name] = rcsb_smiles
            else:
                small_molecules_seen[r_name] = template
            frag = AllChem.AssignBondOrdersFromTemplate(template, frag)
            frag = Chem.AddHs(frag, addCoords=True)
            fragments[f"{r_chain}-{r_name}-{r_id}"] = frag

    if fragments:
        print(f"Found {len(fragments)} small molecules:", ", ".join(fragments.keys()))

    to_delete = []
    for residue in modeller.topology.residues():
        if residue.name not in standardResidues:
            query_key = f"{residue.chain.id}-{residue.name}-{residue.id}"
            if query_key in fragments:
                print("Removing", query_key)
                to_delete.append(residue)
    modeller.delete(to_delete) 

    if smiles_templates:
        small_molecules = list(small_molecules_seen.values())
    else:
        small_molecules = []
        for k, template in small_molecules_seen.items():
            print(f"Added {k} to small molecule templates.")
            template_mol = Molecule.from_rdkit(template, allow_undefined_stereo=True)
            small_molecules.append(template_mol)

    for k, frag in fragments.items():
        frag_mol = Chem.MolToMolBlock(frag)
        frag_mol = Chem.MolFromMolBlock(frag_mol)
        frag_mol = Molecule.from_rdkit(frag_mol, allow_undefined_stereo=True)
        frag_top = frag_mol.to_topology()
        modeller.add(frag_top.to_openmm(), frag_top.get_positions().to_openmm())
        print(f"Added {k} to structure")

    return small_molecules

def add_ff_template_generator_from_json(forcefield, small_molecules_path, cache_path=None):
    with open(small_molecules_path, "r") as f:
        small_molecules_smiles = json.load(f)

    add_ff_template_generator_from_smiles(forcefield, small_molecules_smiles, cache_path)

def add_ff_template_generator_from_smiles(forcefield, small_molecules_smiles, cache_path=None):
    small_molecules = []
    for smiles in small_molecules_smiles:
        template = Chem.MolFromSmiles(smiles)
        template_mol = Molecule.from_rdkit(template, allow_undefined_stereo=True)
        small_molecules.append(template_mol)

    gaff = GAFFTemplateGenerator(molecules=small_molecules, cache=cache_path)
    forcefield.registerTemplateGenerator(gaff.generator)

    print(f"Added {len(small_molecules)} small molecule templates to forcefield")

def prepare_protein(pdbid=str):
    print(f"Preprocess of {pdbid}")
    create_folder(f"{pdbid}_WP")
    pdb_path = f"{pdbid}_WP/raw/{pdbid}.pdb"

    pdb_url = f"https://files.rcsb.org/download/{pdbid}.pdb"

    if not os.path.exists(pdb_path):
        r = requests.get(pdb_url)
        r.raise_for_status()
        with open(pdb_path, "wb") as f:
            f.write(r.content)
    else:
        print(f"{pdbid}.pdb already downloaded")

    fixer = pdbfixer.PDBFixer(pdb_path)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    print(f"Missing residues: {fixer.missingResidues}")
    print(f"Missing terminals: {fixer.missingTerminals}")
    print(f"Missing atoms: {fixer.missingAtoms}")

    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    for key in list(keys):
        chain = chains[key[0]]
        assert key[1] != 0 or key[1] != len(list(chain.residues())), "Terminal residues are not removed."

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    ph = 7.0
    fixer.addMissingHydrogens(ph)

    modeller = app.Modeller(fixer.topology, fixer.positions)

    small_molecules = replace_ligands(pdb_path, modeller)

    print("\nAfter the process")
    print(f"Missing residues: {fixer.missingResidues}")
    print(f"Missing terminals: {fixer.missingTerminals}")
    print(f"Missing atoms: {fixer.missingAtoms}")

    forcefield_configs = ["amber14-all.xml", "amber14/tip3pfb.xml"]
    json.dump(forcefield_configs, open(f"{pdbid}_WP/processed/forcefield.json", 'w', encoding='utf-8'))

    forcefield = app.ForceField(*forcefield_configs)

    if small_molecules:
        json.dump(small_molecules, open(f"{pdbid}_WP/processed/{pdbid}_processed_ligands_smiles.json", 'w'))
        template_cache_path = f"{pdbid}_WP/processed/{pdbid}_processed_ligands_cache.json"
        add_ff_template_generator_from_smiles(forcefield, small_molecules, cache_path=template_cache_path)

    unmatched_residues = [r for r in forcefield.getUnmatchedResidues(modeller.topology) if r.name != "UNK"]
    if unmatched_residues:
        raise RuntimeError("Structure still contains unmatched residues after fixup: " + str(unmatched_residues))

    modeller.addSolvent(forcefield, padding=1.0 * unit.nanometers, ionicStrength=0.15 * unit.molar)

    top = modeller.getTopology()
    pos = modeller.getPositions()
    app.PDBFile.writeFile(top, pos, open(f"{pdbid}_WP/processed/{pdbid}_processed.pdb", 'w'))

    pdb = app.PDBFile(f"{pdbid}_WP/processed/{pdbid}_processed.pdb")
    assert pdb.topology.getNumResidues() == top.getNumResidues()
    assert pdb.topology.getNumAtoms() == top.getNumAtoms()
    assert pdb.topology.getNumBonds() == top.getNumBonds()

    top_bonds = [len([*i.bonds()]) for i in top.residues() if i.name == 'UNK']
    pdb_bonds = [len([*i.bonds()]) for i in pdb.topology.residues() if i.name == 'UNK']
    assert pdb_bonds == top_bonds

if __name__ == "__main__":
   if len(sys.argv) != 2:
       print("Error. usage: ./preprocess_pdb.py PDB_ID")
       sys.exit(1)
   else:
       prepare_protein(sys.argv[1])
