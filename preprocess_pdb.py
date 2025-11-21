#!/usr/bin/env python3

import os
import sys
import json
import shutil
import requests
import numpy as np
import pdbfixer
import openmm as mm
import openmm.app as app
from openmm import unit
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator


# ============================================================
#  SIMPLE LOCAL PATH HANDLER
# ============================================================

def wp_path(pdbid: str, subpath: str = ""):
    """
    Return ./PDBID_WP/subpath relative to where script is run.
    """
    root = os.path.join(os.getcwd(), f"{pdbid}_WP")
    return os.path.join(root, subpath)


def create_dirs(pdbid: str):
    base = wp_path(pdbid)
    subdirs = ["raw", "processed"]

    if not os.path.exists(base):
        os.makedirs(base)

    for s in subdirs:
        os.makedirs(os.path.join(base, s), exist_ok=True)

    print(f"[dir] Created directory tree under {base}")


# ============================================================
#  LIGAND UTILS (minimal version of ligands.py)
# ============================================================

def get_rcsb_ligand_smiles(resname):
    """Query RCSB for ligand template."""
    try:
        url = f"https://files.rcsb.org/ligands/view/{resname}_ideal.sdf"
        rsp = requests.get(url)
        if rsp.status_code != 200:
            return None
        mol = Chem.MolFromMolBlock(rsp.text, sanitize=False)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def replace_ligands(pdb_filename, modeller):
    pdb_mol = Chem.rdmolfiles.MolFromPDBFile(pdb_filename, removeHs=False, proximityBonding=True)

    standard = {
        "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE","LEU","LYS","MET","PHE",
        "PRO","SER","THR","TRP","TYR","VAL","HOH"
    }

    fragments = {}
    small_mols = {}

    for frag_idx, frag in enumerate(Chem.rdmolops.GetMolFrags(pdb_mol, asMols=True)):
        a = frag.GetAtomWithIdx(0)
        rname = a.GetPDBResidueInfo().GetResidueName()

        if frag.GetNumAtoms() == 1: continue
        if rname in standard: continue

        rid = a.GetPDBResidueInfo().GetResidueNumber()
        chain = a.GetPDBResidueInfo().GetChainId()

        # Must be unbound
        is_alone = True
        for aa in frag.GetAtoms():
            if rid != aa.GetPDBResidueInfo().GetResidueNumber():
                is_alone = False
                break
        if not is_alone: continue

        smiles = get_rcsb_ligand_smiles(rname)
        if smiles is None:
            print(f"[WARN] No template for ligand: {rname}")
            continue

        small_mols[rname] = smiles
        frag = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smiles), frag)
        frag = Chem.AddHs(frag, addCoords=True)
        fragments[f"{chain}-{rname}-{rid}"] = frag

    # Remove old residues
    to_delete = []
    for residue in modeller.topology.residues():
        if residue.name not in standard:
            key = f"{residue.chain.id}-{residue.name}-{residue.id}"
            if key in fragments:
                to_delete.append(residue)
    modeller.delete(to_delete)

    # Add modified fragments
    for k, frag in fragments.items():
        molblock = Chem.MolToMolBlock(frag)
        frag_mol = Molecule.from_rdkit(Chem.MolFromMolBlock(molblock), allow_undefined_stereo=True)
        top = frag_mol.to_topology()
        modeller.add(top.to_openmm(), top.get_positions().to_openmm())
        print(f"[ligand] Added {k}")

    return list(small_mols.values())


def add_ff_templates(forcefield, smiles_list, cache_path):
    mols = []
    for sm in smiles_list:
        rd = Chem.MolFromSmiles(sm)
        mols.append(Molecule.from_rdkit(rd, allow_undefined_stereo=True))

    gaff = GAFFTemplateGenerator(molecules=mols, cache=cache_path)
    forcefield.registerTemplateGenerator(gaff.generator)
    print(f"[ff] Added {len(mols)} ligand templates")


# ============================================================
#  PREPROCESS FUNCTION
# ============================================================

def preprocess(pdbid: str, remove_ligands=False, implicit_solvent=False):

    # normalize ID
    local_pdb = None
    if pdbid.endswith(".pdb"):
        local_pdb = pdbid
        pdbid = os.path.splitext(os.path.basename(local_pdb))[0]

    print(f"[run] Preprocessing {pdbid}")
    create_dirs(pdbid)

    raw_path = wp_path(pdbid, f"raw/{pdbid}.pdb")

    # DOWNLOAD / COPY
    if local_pdb:
        shutil.copyfile(local_pdb, raw_path)
        print(f"[io] Using local file {local_pdb}")
    else:
        url = f"https://files.rcsb.org/download/{pdbid}.pdb"
        if not os.path.exists(raw_path):
            r = requests.get(url)
            r.raise_for_status()
            with open(raw_path, "wb") as f:
                f.write(r.content)
            print(f"[io] Downloaded {pdbid}.pdb")
        else:
            print(f"[io] Using cached raw/{pdbid}.pdb")

    # FIX
    fixer = pdbfixer.PDBFixer(raw_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    modeller = app.Modeller(fixer.topology, fixer.positions)

    if remove_ligands:
        small = []
    else:
        small = replace_ligands(raw_path, modeller)

    # FORCEFIELD
    if implicit_solvent:
        ff_cfg = ["amber14-all.xml", "amber14/DNA.OL15.xml", "implicit/gbn2.xml"]
    else:
        ff_cfg = ["amber14-all.xml", "amber14/DNA.OL15.xml", "amber14/tip3pfb.xml"]

    proc_folder = wp_path(pdbid, "processed")
    with open(os.path.join(proc_folder, "forcefield.json"), "w") as f:
        json.dump(ff_cfg, f)

    ff = app.ForceField(*ff_cfg)

    if small:
        smiles_path = wp_path(pdbid, f"processed/{pdbid}_ligands_smiles.json")
        cache_path = wp_path(pdbid, f"processed/{pdbid}_ligands_cache.json")
        with open(smiles_path, "w") as f:
            json.dump(small, f)
        add_ff_templates(ff, small, cache_path)

    if implicit_solvent:
        modeller.deleteWater()
    else:
        modeller.addSolvent(ff, padding=1.0*unit.nanometers, ionicStrength=0.15*unit.molar)

    # WRITE OUTPUT
    final_pdb = wp_path(pdbid, f"processed/{pdbid}_processed.pdb")
    with open(final_pdb, "w") as f:
        app.PDBFile.writeFile(modeller.getTopology(), modeller.getPositions(), f)

    print(f"[done] Wrote processed PDB → {final_pdb}")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./preprocess_single.py PDBID")
        sys.exit(1)

    preprocess(sys.argv[1])

