#!/usr/bin/env python3
from __future__ import annotations

"""NERSC Distributed Molecular Simulation: UI, config reads, and preprocessing."""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.parse

try:
    import requests
except ImportError:  # pragma: no cover - optional for read-config path
    requests = None

try:
    import streamlit as st
except ImportError:  # pragma: no cover - streamlit only needed for UI paths
    st = None

APP_DIR = pathlib.Path(__file__).resolve().parent
CONFIG_EXAMPLE_PATH = APP_DIR / "config.example.json"
PDB_IDS_PATH = APP_DIR / "pdb_ids.json"

RCSB_BASE = "https://search.rcsb.org"
RCSB_SEARCH_URL = f"{RCSB_BASE}/rcsbsearch/v2/query"
RCSB_SUGGEST_URL = f"{RCSB_BASE}/rcsbsearch/v2/suggest"
RCSB_UNRELEASED_URL = f"{RCSB_BASE}/rcsbsearch/v2/query/unreleased"
RCSB_META_URLS = {
    "structure": f"{RCSB_BASE}/rcsbsearch/v2/metadata/schema",
    "chemical": f"{RCSB_BASE}/rcsbsearch/v2/metadata/chemical/schema",
    "unreleased": f"{RCSB_BASE}/rcsbsearch/v2/metadata/unreleased/schema",
}
NERSC_HOST = "perlmutter.nersc.gov"
MAX_GET_URL_LEN = 2000

RCSB_STATUS_MSG = {
    200: "OK",
    204: "No Content: query valid, zero results.",
    400: "Bad Request: malformed request or invalid query syntax.",
    404: "Not Found: requested resource does not exist.",
    408: "Request Timeout: server could not finish in time. Simplify the query.",
    415: "Unsupported Media Type: payload format not accepted.",
    500: "Internal Server Error: unexpected server failure.",
    501: "Not Implemented: requested functionality not supported.",
    503: "Service Unavailable: temporary overload. Retry later.",
}

TEXT_OPERATORS = [
    "exact_match",
    "contains_phrase",
    "contains_words",
    "equals",
    "greater",
    "greater_or_equal",
    "less",
    "less_or_equal",
    "range",
    "in",
    "exists",
]
SEARCH_SERVICES = [
    "text",
    "full_text",
    "text_chem",
    "structure",
    "sequence",
    "seqmotif",
    "strucmotif",
    "chemical",
]

ANSI_RE = re.compile(r"\x1B\[[0-9;]*[mK]")

_PREPROCESS_DEPS_READY = False


def _streamlit_required():
    if st is None:
        raise RuntimeError("streamlit is required for UI mode")


def _requests_required():
    if requests is None:
        raise RuntimeError("requests is required for this command path")


def _runtime_config_dir() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("NDMS_CONFIG_DIR")
        or os.environ.get("HELIXNET_CONFIG_DIR", str(APP_DIR))
    )


def _runtime_config_path() -> pathlib.Path:
    return _runtime_config_dir() / "config.json"


def load_runtime_config() -> dict:
    path = _runtime_config_path()
    if not path.exists():
        fallback = _runtime_config_dir() / "config.example.json"
        if fallback.exists():
            path = fallback
        elif CONFIG_EXAMPLE_PATH.exists():
            path = CONFIG_EXAMPLE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No config file found. Searched: {_runtime_config_path()}, "
            f"{_runtime_config_dir() / 'config.example.json'}, {CONFIG_EXAMPLE_PATH}. "
            f"Copy config.example.json to config.json and edit it."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_config_value(dot_path: str):
    cfg = load_runtime_config()
    value = cfg
    for key in dot_path.split("."):
        value = value[key]
    return value


def load_config() -> dict:
    return load_runtime_config()


def save_config(cfg: dict):
    with open(_runtime_config_path(), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def load_pdb_ids() -> list[str]:
    if PDB_IDS_PATH.exists():
        with open(PDB_IDS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_pdb_ids(ids: list[str]):
    with open(PDB_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f)
        f.write("\n")


def detect_execution_mode() -> str:
    hostname = os.uname().nodename.lower()
    return "local" if ("nersc" in hostname or "perlmutter" in hostname) else "ssh"


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _get_ssh_client(cfg: dict):
    _streamlit_required()
    try:
        import paramiko
    except ImportError:
        st.error("paramiko is required for SSH mode. Install with: pip install paramiko")
        return None
    user = cfg.get("execution", {}).get("nersc_user", "")
    if not user:
        st.error("Set execution.nersc_user in the Configuration tab.")
        return None
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(NERSC_HOST, username=user)
    except Exception as exc:  # pragma: no cover - network dependent
        st.error(f"SSH connection failed: {exc}")
        return None
    return client


def run_script(cfg: dict, command: str, placeholder) -> str:
    mode = detect_execution_mode()
    project_dir = cfg.get("paths", {}).get("project_dir", str(APP_DIR))
    if mode == "ssh":
        client = _get_ssh_client(cfg)
        if client is None:
            return ""
        try:
            cmd = f"cd {project_dir} && bash -lc {json.dumps(command)}"
            _, stdout, stderr = client.exec_command(cmd, get_pty=True)
            lines: list[str] = []
            for line in stdout:
                lines.append(strip_ansi(line))
                placeholder.code("".join(lines))
            err = stderr.read().decode()
            if err:
                lines.append(err)
                placeholder.code("".join(lines))
            return "".join(lines)
        finally:
            client.close()
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        lines: list[str] = []
        if proc.stdout is None:
            raise RuntimeError("subprocess stdout is None despite PIPE")
        for line in proc.stdout:
            lines.append(strip_ansi(line))
            placeholder.code("".join(lines))
        return "".join(lines)
    finally:
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_remote_cmd(cfg: dict, cmd: str) -> str:
    client = _get_ssh_client(cfg)
    if client is None:
        return ""
    try:
        _, stdout, _ = client.exec_command(cmd)
        return stdout.read().decode()
    finally:
        client.close()


def _auto_method(payload: dict) -> str:
    encoded = json.dumps(payload)
    url_len = len(RCSB_SEARCH_URL) + len("?json=") + len(urllib.parse.quote(encoded))
    return "get" if url_len <= MAX_GET_URL_LEN else "post"


def _rcsb_handle(resp: requests.Response) -> tuple[dict | None, str | None]:
    if resp.status_code == 204:
        return {"total_count": 0, "result_set": []}, None
    msg = RCSB_STATUS_MSG.get(resp.status_code, f"HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        if resp.status_code == 200:
            return None, "HTTP 200 but non-JSON body."
        return None, f"{msg}\n{resp.text[:500]}"
    if resp.status_code != 200:
        return data, msg
    return data, None


def build_rcsb_payload(cfg: dict) -> dict:
    search = cfg.get("rcsb_search", {})
    keywords = search.get(
        "keywords",
        ["DNA BINDING PROTEIN, DNA", "RNA BINDING PROTEIN, RNA", "DNA", "RNA"],
    )
    operator = search.get("keyword_operator", "contains_phrase")
    organism = search.get("organism", "Homo sapiens")
    min_res = search.get("min_resolution")
    max_res = search.get("max_resolution", 2.5)
    return_type = search.get("return_type", "entry")

    kw_nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "struct_keywords.pdbx_keywords",
                "operator": operator,
                "value": kw,
            },
        }
        for kw in keywords
    ]
    nodes = [
        {"type": "group", "logical_operator": "or", "nodes": kw_nodes},
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entity_source_organism.scientific_name",
                "operator": "exact_match",
                "value": organism,
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": max_res,
            },
        },
    ]
    if min_res is not None:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "greater_or_equal",
                    "value": min_res,
                },
            }
        )
    return {"query": {"type": "group", "logical_operator": "and", "nodes": nodes}, "return_type": return_type}


def execute_rcsb_search(
    payload: dict,
    *,
    method: str = "post",
    request_options: dict | None = None,
) -> tuple[list[str], dict, dict]:
    _requests_required()
    if request_options:
        payload = {**payload, "request_options": request_options}
    sent = dict(payload)
    try:
        if method == "get":
            resp = requests.get(RCSB_SEARCH_URL, params={"json": json.dumps(payload)}, timeout=60)
        else:
            resp = requests.post(RCSB_SEARCH_URL, json=payload, timeout=60)
    except requests.RequestException as exc:
        return [], {"error": str(exc)}, sent

    data, err = _rcsb_handle(resp)
    if err:
        error_data = data if data is not None else {}
        error_data["error"] = err
        return [], error_data, sent
    if data is None:
        return [], {"error": "Empty response"}, sent

    total = data.get("total_count", 0)
    ids = [r["identifier"] for r in data.get("result_set", [])]

    if total > len(ids):
        rows_per_page = max(len(ids), 100)
        fetched = len(ids)
        while fetched < total:
            page_opts = {"paginate": {"start": fetched, "rows": rows_per_page}}
            if request_options:
                page_opts = {**request_options, **page_opts}
            page_payload = {**payload, "request_options": page_opts}
            try:
                if method == "get":
                    pr = requests.get(RCSB_SEARCH_URL, params={"json": json.dumps(page_payload)}, timeout=60)
                else:
                    pr = requests.post(RCSB_SEARCH_URL, json=page_payload, timeout=60)
            except requests.RequestException:
                break
            if pr.status_code != 200:
                break
            try:
                pd = pr.json()
            except ValueError:
                break
            page_ids = [r["identifier"] for r in pd.get("result_set", [])]
            if not page_ids:
                break
            ids.extend(page_ids)
            fetched += len(page_ids)
        data["result_set"] = [{"identifier": i} for i in ids]
        data["total_count"] = len(ids)

    return ids, data, sent


def rcsb_suggest(text: str, attributes: list[str] | None = None) -> tuple[dict[str, list[dict]], str | None]:
    _requests_required()
    suggest_body: dict = {"text": text}
    if attributes:
        suggest_body["attributes"] = attributes
    payload = {"type": "basic", "suggest": suggest_body}
    try:
        resp = requests.get(RCSB_SUGGEST_URL, params={"json": json.dumps(payload)}, timeout=10)
    except requests.RequestException as exc:
        return {}, str(exc)
    if resp.status_code == 204:
        return {}, None
    if resp.status_code == 500:
        return {}, RCSB_STATUS_MSG[500]
    try:
        data = resp.json()
    except ValueError:
        return {}, f"HTTP {resp.status_code}: non-JSON response"
    if "message" in data and resp.status_code != 200:
        return {}, data["message"]
    return data.get("suggestions", {}), None


def rcsb_search_unreleased(query: dict) -> tuple[list[str], dict]:
    _requests_required()
    payload = {"query": query, "return_type": "unreleased_entry"}
    try:
        resp = requests.get(RCSB_UNRELEASED_URL, params={"json": json.dumps(payload)}, timeout=60)
    except requests.RequestException as exc:
        return [], {"error": str(exc)}
    data, err = _rcsb_handle(resp)
    if err:
        error_data = data if data is not None else {}
        error_data["error"] = err
        return [], error_data
    if data is None:
        return [], {"error": "Empty response"}
    ids = [r.get("identifier", "") for r in data.get("result_set", [])]
    return ids, data


def rcsb_get_metadata(schema_type: str = "structure") -> tuple[dict | None, str | None]:
    _requests_required()
    url = RCSB_META_URLS.get(schema_type, RCSB_META_URLS["structure"])
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        return None, str(exc)
    if resp.status_code == 404:
        return None, "Not Found."
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    try:
        return resp.json(), None
    except ValueError:
        return None, "Non-JSON response."


def _resolve_out_dir(cfg: dict, base: pathlib.Path) -> pathlib.Path:
    out_raw = (cfg.get("paths", {}).get("out_dir") or "").strip()
    if not out_raw:
        return base
    if out_raw.startswith("/"):
        return pathlib.Path(out_raw)
    return base / out_raw


def scan_wp_dirs(cfg: dict) -> list[dict]:
    mode = detect_execution_mode()
    project_dir = cfg.get("paths", {}).get("project_dir", str(APP_DIR))
    out_dir = _resolve_out_dir(cfg, pathlib.Path(project_dir))
    out_dir_str = str(out_dir)

    if mode == "ssh":
        listing = run_remote_cmd(cfg, f"ls -d {out_dir_str}/*_WP 2>/dev/null || true")
        dirs = [os.path.basename(d.strip()) for d in listing.strip().splitlines() if d.strip()]
    else:
        if out_dir.is_dir():
            dirs = sorted(d.name for d in out_dir.iterdir() if d.is_dir() and d.name.endswith("_WP"))
        else:
            dirs = []

    rows: list[dict] = []
    for directory in dirs:
        pdb_id = directory.replace("_WP", "")
        row = {"PDB ID": pdb_id, "west.h5": False, "Iterations": "-", "Status": "unknown"}
        if mode == "ssh":
            check = run_remote_cmd(cfg, f"test -s {out_dir_str}/{directory}/west.h5 && echo yes || echo no").strip()
            row["west.h5"] = check == "yes"
            if row["west.h5"]:
                iters = run_remote_cmd(
                    cfg,
                    f"h5ls {out_dir_str}/{directory}/west.h5/iterations 2>/dev/null"
                    " | awk '/^iter_/ {split($1,a,\"_\"); v=a[2]} END {print v+0}'",
                ).strip()
                row["Iterations"] = iters if iters else "0"
        else:
            h5 = out_dir / directory / "west.h5"
            row["west.h5"] = h5.exists() and h5.stat().st_size > 0
            if row["west.h5"]:
                try:
                    out = subprocess.check_output(["h5ls", f"{out_dir / directory}/west.h5/iterations"], text=True, stderr=subprocess.DEVNULL)
                    nums = [int(m.group(1)) for m in re.finditer(r"iter_(\d+)", out)]
                    row["Iterations"] = str(max(nums)) if nums else "0"
                except Exception as exc:
                    row["Iterations"] = f"err: {exc}"

        target = cfg.get("westpa", {}).get("target_iterations", 12500)
        try:
            it = int(row["Iterations"])
            if it >= target:
                row["Status"] = "done"
            elif row["west.h5"]:
                row["Status"] = "running"
            else:
                row["Status"] = "error"
        except (ValueError, TypeError):
            row["Status"] = "error"
        rows.append(row)

    return rows


def _import_preprocess_deps():
    global _PREPROCESS_DEPS_READY
    if _PREPROCESS_DEPS_READY:
        return
    global np, pdbfixer, Chem, AllChem, Molecule, GAFFTemplateGenerator, unit, PDBFile, openmm_app
    import numpy as np  # type: ignore
    import pdbfixer  # type: ignore
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    from openff.toolkit import Molecule  # type: ignore
    from openmm import unit  # type: ignore
    from openmm.app import PDBFile  # type: ignore
    import openmm.app as openmm_app  # type: ignore
    from openmmforcefields.generators import GAFFTemplateGenerator  # type: ignore
    _PREPROCESS_DEPS_READY = True


def validate_pdb_id(pdbid: str) -> None:
    if not pdbid or len(pdbid) != 4 or not pdbid.isalnum():
        raise ValueError("PDB ID must be exactly 4 alphanumeric characters")


def create_folder(folder_path: str):
    os.makedirs(f"{folder_path}/raw", exist_ok=True)
    os.makedirs(f"{folder_path}/processed", exist_ok=True)


def get_rcsb_ligand_smiles(comp_id):
    try:
        return get_rcsb_ligand_smiles_exc(comp_id)
    except Exception:
        return None


def get_rcsb_ligand_smiles_exc(comp_id):
    _requests_required()
    comp_id = str(comp_id)
    if not comp_id or len(comp_id) > 3:
        raise RuntimeError("Invalid comp_id, must be a 1-3 character string.")
    query_string = (
        '{chem_comp(comp_id:"'
        + comp_id
        + '"){chem_comp{id,name,formula},rcsb_chem_comp_descriptor{SMILES,SMILES_stereo}}}'
    )
    query_url = "https://data.rcsb.org/graphql?query=" + urllib.parse.quote(query_string)
    response = requests.get(query_url, timeout=30)
    response.raise_for_status()
    try:
        return response.json()["data"]["chem_comp"]["rcsb_chem_comp_descriptor"]["SMILES_stereo"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected RCSB GraphQL response for {comp_id}: {exc}") from exc


def replace_ligands(pdb_filename, modeller, smiles_templates=True):
    _import_preprocess_deps()
    pdb_mol = Chem.rdmolfiles.MolFromPDBFile(pdb_filename, removeHs=False, proximityBonding=True)
    if pdb_mol is None:
        return []
    standard_residues = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "ASX",
        "CYS",
        "GLU",
        "GLN",
        "GLX",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "HOH",
        "DA", "DT", "DC", "DG", "DU",
        "A", "U", "C", "G", "N",
    }
    fragments = {}
    small_molecules_seen = {}
    for frag in Chem.rdmolops.GetMolFrags(pdb_mol, asMols=True):
        atom0 = frag.GetAtomWithIdx(0)
        info0 = atom0.GetPDBResidueInfo()
        if info0 is None:
            continue
        r_name = info0.GetResidueName()
        if frag.GetNumAtoms() == 1 or r_name in standard_residues:
            continue
        r_id = info0.GetResidueNumber()
        r_chain = info0.GetChainId()
        if all((ai := a.GetPDBResidueInfo()) is not None and r_id == ai.GetResidueNumber() for a in frag.GetAtoms()):
            rcsb_smiles = get_rcsb_ligand_smiles(r_name)
            if rcsb_smiles is None:
                continue
            template = Chem.MolFromSmiles(rcsb_smiles)
            small_molecules_seen[r_name] = rcsb_smiles if smiles_templates else template
            frag = AllChem.AssignBondOrdersFromTemplate(template, frag)
            frag = Chem.AddHs(frag, addCoords=True)
            fragments[f"{r_chain}-{r_name}-{r_id}"] = frag

    to_delete = []
    for residue in modeller.topology.residues():
        if residue.name not in standard_residues:
            query_key = f"{residue.chain.id}-{residue.name}-{residue.id}"
            if query_key in fragments:
                to_delete.append(residue)
    modeller.delete(to_delete)

    if smiles_templates:
        small_molecules = list(small_molecules_seen.values())
    else:
        small_molecules = []
        for template in small_molecules_seen.values():
            small_molecules.append(Molecule.from_rdkit(template, allow_undefined_stereo=True))

    for frag in fragments.values():
        frag_mol = Chem.MolToMolBlock(frag)
        frag_mol = Chem.MolFromMolBlock(frag_mol)
        frag_mol = Molecule.from_rdkit(frag_mol, allow_undefined_stereo=True)
        frag_top = frag_mol.to_topology()
        modeller.add(frag_top.to_openmm(), frag_top.get_positions().to_openmm())
    return small_molecules


def add_ff_template_generator_from_smiles(forcefield, small_molecules_smiles, cache_path=None):
    _import_preprocess_deps()
    small_molecules = []
    for smiles in small_molecules_smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            continue
        small_molecules.append(Molecule.from_rdkit(template, allow_undefined_stereo=True))
    gaff = GAFFTemplateGenerator(molecules=small_molecules, cache=cache_path)
    forcefield.registerTemplateGenerator(gaff.generator)


def prepare_protein(pdbid: str):
    _import_preprocess_deps()
    _requests_required()
    cfg = load_runtime_config()
    create_folder(f"{pdbid}_WP")
    pdb_path = f"{pdbid}_WP/raw/{pdbid}.pdb"
    pdb_url = f"https://files.rcsb.org/download/{pdbid}.pdb"
    if not os.path.exists(pdb_path):
        response = requests.get(pdb_url, timeout=60)
        response.raise_for_status()
        with open(pdb_path, "wb") as f:
            f.write(response.content)

    fixer = pdbfixer.PDBFixer(pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    chains = list(fixer.topology.chains())
    keys = list(fixer.missingResidues.keys())
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    preproc = cfg.get("preprocessing", {})
    ph = preproc.get("ph", 7.0)
    padding_nm = preproc.get("padding_nm", 1.0)
    ionic_strength = preproc.get("ionic_strength_M", 0.15)
    fixer.addMissingHydrogens(ph)

    modeller = openmm_app.Modeller(fixer.topology, fixer.positions)
    small_molecules = replace_ligands(pdb_path, modeller)
    forcefield_configs = cfg.get("openmm", {}).get("forcefield", ["amber14-all.xml", "amber14/tip3pfb.xml"])
    with open(f"{pdbid}_WP/processed/forcefield.json", "w", encoding="utf-8") as f:
        json.dump(forcefield_configs, f)

    forcefield = openmm_app.ForceField(*forcefield_configs)
    if small_molecules:
        with open(f"{pdbid}_WP/processed/{pdbid}_processed_ligands_smiles.json", "w", encoding="utf-8") as f:
            json.dump(small_molecules, f)
        template_cache_path = f"{pdbid}_WP/processed/{pdbid}_processed_ligands_cache.json"
        add_ff_template_generator_from_smiles(forcefield, small_molecules, cache_path=template_cache_path)

    unmatched_residues = [r for r in forcefield.getUnmatchedResidues(modeller.topology) if r.name != "UNK"]
    if unmatched_residues:
        raise RuntimeError(f"Structure still contains unmatched residues after fixup: {unmatched_residues}")

    modeller.addSolvent(forcefield, padding=padding_nm * unit.nanometers, ionicStrength=ionic_strength * unit.molar)
    top = modeller.getTopology()
    pos = modeller.getPositions()
    with open(f"{pdbid}_WP/processed/{pdbid}_processed.pdb", "w", encoding="utf-8") as f:
        openmm_app.PDBFile.writeFile(top, pos, f)


def _display_results(ids: list[str], raw: dict, prefix: str):
    _streamlit_required()
    if ids:
        total = raw.get("total_count", len(ids))
        st.success(f"**{len(ids)}** entries fetched (API total: {total})")
        st.code(", ".join(ids[:100]) + ("..." if len(ids) > 100 else ""))
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Replace pdb_ids.json", key=f"{prefix}_repl"):
                save_pdb_ids(ids)
                st.success(f"Saved {len(ids)} IDs")
        with c2:
            if st.button("Append to pdb_ids.json", key=f"{prefix}_app"):
                existing = load_pdb_ids()
                merged = list(dict.fromkeys(existing + ids))
                save_pdb_ids(merged)
                st.success(f"Merged to {len(merged)} IDs")
        with c3:
            st.download_button(
                "Download IDs",
                data=json.dumps(ids, indent=2),
                file_name="pdb_ids.json",
                mime="application/json",
                key=f"{prefix}_dl",
            )
    elif raw.get("total_count") == 0:
        st.warning("Query returned zero results.")
    elif "error" in raw:
        st.error(f"Search failed: {raw['error']}")


def main():
    _streamlit_required()
    st.set_page_config(page_title="NDMS", layout="wide")
    st.title("NERSC Distributed Molecular Simulation")
    cfg = load_config()
    tab_cfg, tab_search, tab_pipe, tab_status = st.tabs(["Configuration", "RCSB Search", "Pipeline", "Status"])

    with tab_cfg:
        st.header("Configuration")
        nersc_user = st.text_input("NERSC User", cfg.get("execution", {}).get("nersc_user", ""))
        cfg.setdefault("execution", {})["nersc_user"] = nersc_user
        project_dir = st.text_input("Project directory", cfg.get("paths", {}).get("project_dir", ""))
        out_dir = st.text_input("Out directory", cfg.get("paths", {}).get("out_dir", "out"))
        cfg.setdefault("paths", {}).update({"project_dir": project_dir, "out_dir": out_dir})
        if st.button("Save Configuration", type="primary"):
            save_config(cfg)
            st.success("Saved to config.json")

    with tab_search:
        st.header("RCSB Search")
        if st.button("Run Search", type="primary"):
            payload = build_rcsb_payload(cfg)
            method = _auto_method(payload)
            ids, raw, _ = execute_rcsb_search(payload, method=method, request_options={"paginate": {"start": 0, "rows": 25}})
            st.session_state["search_ids"] = ids
            st.session_state["search_raw"] = raw
        if "search_ids" in st.session_state:
            _display_results(st.session_state["search_ids"], st.session_state["search_raw"], "search")

    with tab_pipe:
        st.header("Pipeline Control")
        col1, col2, col3, col4 = st.columns(4)
        run_setup = col1.button("Setup Target")
        run_batch = col2.button("Batch Setup")
        run_jobs = col3.button("Run Jobs")
        run_full = col4.button("Full Pipeline")
        target_id = st.text_input("Setup PDB ID", value="1ABC")
        output_area = st.empty()
        if run_setup:
            try:
                validate_pdb_id(target_id)
            except ValueError as exc:
                st.error(str(exc))
            else:
                run_script(cfg, f"./run.sh setup {target_id}", output_area)
        elif run_batch:
            run_script(cfg, "./run.sh batch-setup", output_area)
        elif run_jobs:
            run_script(cfg, "./run.sh run", output_area)
        elif run_full:
            run_script(cfg, "./run.sh batch", output_area)

    with tab_status:
        st.header("Simulation Status")
        if st.button("Refresh"):
            st.session_state["status_rows"] = scan_wp_dirs(cfg)
        rows = st.session_state.get("status_rows", [])
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("Click Refresh to scan for *_WP directories.")


def _build_parser():
    parser = argparse.ArgumentParser(prog="benchmark.py")
    sub = parser.add_subparsers(dest="command")
    read_cfg = sub.add_parser("read-config", help="Read dot-path value from config.json")
    read_cfg.add_argument("key")
    preprocess = sub.add_parser("preprocess", help="Run PDB preprocessing")
    preprocess.add_argument("pdb_id")
    sub.add_parser("ui", help="Launch Streamlit UI mode")
    return parser


def _run_cli(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "read-config":
        try:
            value = get_config_value(args.key)
        except (FileNotFoundError, KeyError, TypeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(value if isinstance(value, str) else json.dumps(value))
        return 0
    if args.command == "preprocess":
        try:
            validate_pdb_id(args.pdb_id)
            prepare_protein(args.pdb_id)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "ui":
        main()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    if sys.argv[1:]:
        sys.exit(_run_cli(sys.argv[1:]))
    elif st is not None:
        main()
