#!/usr/bin/env python3
"""HelixNet - Streamlit control panel for WESTPA/OpenMM DNA-protein MD."""

import json
import os
import pathlib
import re
import subprocess

import requests
import streamlit as st

APP_DIR = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
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
    "exact_match", "contains_phrase", "contains_words",
    "equals", "greater", "greater_or_equal", "less", "less_or_equal",
    "range", "in", "exists",
]
COMMON_ATTRIBUTES = [
    "struct_keywords.pdbx_keywords",
    "struct_keywords.text",
    "struct.title",
    "rcsb_entity_source_organism.scientific_name",
    "rcsb_entry_info.resolution_combined",
    "rcsb_entry_info.molecular_weight",
    "rcsb_entry_info.deposited_atom_count",
    "rcsb_entry_info.experimental_method",
    "entity_poly.rcsb_entity_polymer_type",
    "rcsb_entry_info.polymer_entity_count_DNA",
    "rcsb_entry_info.polymer_entity_count_RNA",
    "rcsb_entry_info.polymer_entity_count_protein",
]
SEARCH_SERVICES = [
    "text", "full_text", "text_chem",
    "structure", "sequence", "seqmotif", "strucmotif", "chemical",
]

MAX_GET_URL_LEN = 2000


def _auto_method(payload: dict) -> str:
    """POST for large payloads, GET when the URL-encoded JSON fits safely."""
    encoded = json.dumps(payload)
    if len(encoded) <= MAX_GET_URL_LEN:
        return "get"
    return "post"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else CONFIG_EXAMPLE_PATH
    with open(path) as f:
        return json.load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def load_pdb_ids() -> list[str]:
    if PDB_IDS_PATH.exists():
        with open(PDB_IDS_PATH) as f:
            return json.load(f)
    return []


def save_pdb_ids(ids: list[str]):
    with open(PDB_IDS_PATH, "w") as f:
        json.dump(ids, f)
        f.write("\n")


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def detect_execution_mode() -> str:
    hostname = os.uname().nodename.lower()
    return "local" if ("nersc" in hostname or "perlmutter" in hostname) else "ssh"


ANSI_RE = re.compile(r"\x1B\[[0-9;]*[mK]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _get_ssh_client(cfg: dict):
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
    except Exception as e:
        st.error(f"SSH connection failed: {e}")
        return None
    return client


def run_script(cfg: dict, script: str, placeholder) -> str:
    mode = detect_execution_mode()
    project_dir = cfg.get("paths", {}).get("project_dir", str(APP_DIR))
    if mode == "ssh":
        client = _get_ssh_client(cfg)
        if client is None:
            return ""
        cmd = f"cd {project_dir} && bash {script}"
        _, stdout, stderr = client.exec_command(cmd, get_pty=True)
        lines: list[str] = []
        for line in stdout:
            lines.append(strip_ansi(line))
            placeholder.code("".join(lines))
        err = stderr.read().decode()
        if err:
            lines.append(err)
            placeholder.code("".join(lines))
        client.close()
        return "".join(lines)
    proc = subprocess.Popen(
        ["bash", script], cwd=str(APP_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(strip_ansi(line))
        placeholder.code("".join(lines))
    proc.wait()
    return "".join(lines)


def run_remote_cmd(cfg: dict, cmd: str) -> str:
    client = _get_ssh_client(cfg)
    if client is None:
        return ""
    _, stdout, _ = client.exec_command(cmd)
    result = stdout.read().decode()
    client.close()
    return result


# ---------------------------------------------------------------------------
# RCSB API - shared response handler
# ---------------------------------------------------------------------------

def _rcsb_handle(resp: requests.Response) -> tuple[dict | None, str | None]:
    """Parse an RCSB response. Returns (data, error_message)."""
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


# ---------------------------------------------------------------------------
# RCSB API - POST & GET /rcsbsearch/v2/query
# ---------------------------------------------------------------------------

def build_rcsb_payload(cfg: dict) -> dict:
    search = cfg.get("rcsb_search", {})
    keywords = search.get("keywords", [
        "DNA BINDING PROTEIN, DNA", "RNA BINDING PROTEIN, RNA", "DNA", "RNA",
    ])
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
            "type": "terminal", "service": "text",
            "parameters": {
                "attribute": "rcsb_entity_source_organism.scientific_name",
                "operator": "exact_match",
                "value": organism,
            },
        },
        {
            "type": "terminal", "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": max_res,
            },
        },
    ]
    if min_res is not None:
        nodes.append({
            "type": "terminal", "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "greater_or_equal",
                "value": min_res,
            },
        })
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": return_type,
    }


def execute_rcsb_search(
    payload: dict,
    *,
    method: str = "post",
    request_options: dict | None = None,
) -> tuple[list[str], dict, dict]:
    """Run RCSB search. Returns (pdb_ids, raw_response, request_sent).

    Handles all v2 status codes per OpenAPI spec:
      200 results    204 no content    400 bad request
      404 not found  408 timeout       415 unsupported media
      500 server err 501 not impl      503 unavailable
    Auto-paginates when total_count > returned rows.
    """
    if request_options:
        payload = {**payload, "request_options": request_options}
    sent = dict(payload)

    try:
        if method == "get":
            resp = requests.get(
                RCSB_SEARCH_URL,
                params={"json": json.dumps(payload)},
                timeout=60,
            )
        else:
            resp = requests.post(RCSB_SEARCH_URL, json=payload, timeout=60)
    except requests.RequestException as e:
        return [], {"error": str(e)}, sent

    data, err = _rcsb_handle(resp)
    if err:
        return [], data or {"error": err}, sent
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
                    pr = requests.get(
                        RCSB_SEARCH_URL,
                        params={"json": json.dumps(page_payload)},
                        timeout=60,
                    )
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


# ---------------------------------------------------------------------------
# RCSB API - GET /rcsbsearch/v2/suggest
# ---------------------------------------------------------------------------

def rcsb_suggest(
    text: str,
    attributes: list[str] | None = None,
) -> tuple[dict[str, list[dict]], str | None]:
    """Autocomplete via basic suggest. Returns ({attr: [{text,score}]}, error).

    Payload schema: {"type":"basic","suggest":{"text":"...","attributes":[...]}}
    Status codes per spec: 200, 204, 500.
    """
    suggest_body: dict = {"text": text}
    if attributes:
        suggest_body["attributes"] = attributes
    payload = {"type": "basic", "suggest": suggest_body}
    try:
        resp = requests.get(
            RCSB_SUGGEST_URL,
            params={"json": json.dumps(payload)},
            timeout=10,
        )
    except requests.RequestException as e:
        return {}, str(e)
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


# ---------------------------------------------------------------------------
# RCSB API - GET /rcsbsearch/v2/query/unreleased
# ---------------------------------------------------------------------------

def rcsb_search_unreleased(query: dict) -> tuple[list[str], dict]:
    """Search unreleased entries. Returns (ids, raw_data).

    Status codes per spec: 200, 204, 400, 404, 408, 415, 500, 501, 503.
    """
    payload = {"query": query, "return_type": "unreleased_entry"}
    try:
        resp = requests.get(
            RCSB_UNRELEASED_URL,
            params={"json": json.dumps(payload)},
            timeout=60,
        )
    except requests.RequestException as e:
        return [], {"error": str(e)}
    data, err = _rcsb_handle(resp)
    if err:
        return [], data or {"error": err}
    if data is None:
        return [], {"error": "Empty response"}
    ids = [r.get("identifier", "") for r in data.get("result_set", [])]
    return ids, data


# ---------------------------------------------------------------------------
# RCSB API - GET /rcsbsearch/v2/metadata/*/schema
# ---------------------------------------------------------------------------

def rcsb_get_metadata(schema_type: str = "structure") -> tuple[dict | None, str | None]:
    """Fetch metadata schema. Returns (schema_dict, error_msg).

    Status codes per spec: 200, 404.
    """
    url = RCSB_META_URLS.get(schema_type, RCSB_META_URLS["structure"])
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        return None, str(e)
    if resp.status_code == 404:
        return None, "Not Found."
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    try:
        return resp.json(), None
    except ValueError:
        return None, "Non-JSON response."


# ---------------------------------------------------------------------------
# Status scanner
# ---------------------------------------------------------------------------

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
        scan_root = _resolve_out_dir(cfg, APP_DIR)
        if scan_root.is_dir():
            dirs = sorted(
                d.name for d in scan_root.iterdir() if d.is_dir() and d.name.endswith("_WP")
            )
        else:
            dirs = []

    rows: list[dict] = []
    for d in dirs:
        pdb_id = d.replace("_WP", "")
        row = {"PDB ID": pdb_id, "west.h5": False, "Iterations": "-", "Status": "unknown"}
        if mode == "ssh":
            check = run_remote_cmd(cfg, f"test -s {out_dir_str}/{d}/west.h5 && echo yes || echo no").strip()
            row["west.h5"] = check == "yes"
            if row["west.h5"]:
                iters = run_remote_cmd(
                    cfg,
                    f"h5ls {out_dir_str}/{d}/west.h5/iterations 2>/dev/null"
                    " | awk '/^iter_/ {split($1,a,\"_\"); v=a[2]} END {print v+0}'",
                ).strip()
                row["Iterations"] = iters if iters else "0"
        else:
            scan_root = _resolve_out_dir(cfg, APP_DIR)
            h5 = scan_root / d / "west.h5"
            row["west.h5"] = h5.exists() and h5.stat().st_size > 0
            if row["west.h5"]:
                try:
                    out = subprocess.check_output(
                        ["h5ls", f"{scan_root / d}/west.h5/iterations"],
                        text=True, stderr=subprocess.DEVNULL,
                    )
                    nums = [int(m.group(1)) for m in re.finditer(r"iter_(\d+)", out)]
                    row["Iterations"] = str(max(nums)) if nums else "0"
                except Exception:
                    row["Iterations"] = "err"
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


# ---------------------------------------------------------------------------
# Credentials gate
# ---------------------------------------------------------------------------

def credentials_gate_passed(cfg: dict) -> bool:
    if detect_execution_mode() == "local":
        return True
    return bool((cfg.get("execution", {}).get("nersc_user") or "").strip())


def show_credentials_gate(cfg: dict) -> bool:
    st.header("NERSC credentials")
    st.write(f"Running remotely. Enter your NERSC username for SSH ({NERSC_HOST}).")
    exc = cfg.get("execution", {})
    nersc_user = st.text_input(
        "NERSC username", value=exc.get("nersc_user", ""), placeholder="username",
    )
    if not nersc_user.strip():
        st.warning("Required for SSH connection.")
        return False
    if st.button("Continue", type="primary"):
        cfg.setdefault("execution", {})["nersc_user"] = nersc_user.strip()
        save_config(cfg)
        st.session_state["credentials_set"] = True
        st.rerun()
    return False


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _init_query_nodes():
    if "query_nodes" not in st.session_state:
        st.session_state["query_nodes"] = [
            {
                "service": "text",
                "attribute": "rcsb_entry_info.struct_keywords",
                "operator": "contains_phrase",
                "value": "DNA",
            },
        ]


def _display_results(ids: list[str], raw: dict, prefix: str):
    """Shared result display for search endpoints."""
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
    else:
        st.error("Unexpected response.")
        with st.expander("Raw response"):
            st.json(raw)


# ---------------------------------------------------------------------------
# UI main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="HelixNet", layout="wide")
    st.title("HelixNet")

    cfg = load_config()

    if not credentials_gate_passed(cfg) and not st.session_state.get("credentials_set"):
        show_credentials_gate(cfg)
        return

    _init_query_nodes()

    tab_cfg, tab_search, tab_api, tab_pipe, tab_status = st.tabs(
        ["Configuration", "RCSB Search", "RCSB API", "Pipeline", "Status"]
    )

    # ==================================================================
    # Tab 1 - Configuration
    # ==================================================================
    with tab_cfg:
        st.header("Configuration")

        with st.expander("Execution", expanded=True):
            mode = detect_execution_mode()
            st.caption(f"Detected: **{mode}** (local = on NERSC, ssh = from Mac)")
            nersc_user = st.text_input(
                "NERSC User", cfg.get("execution", {}).get("nersc_user", ""),
            )
            cfg.setdefault("execution", {})["nersc_user"] = nersc_user

        with st.expander("Paths"):
            project_dir = st.text_input(
                "Project directory", cfg.get("paths", {}).get("project_dir", ""),
            )
            out_dir = st.text_input(
                "Out directory", cfg.get("paths", {}).get("out_dir", "out"),
                help="Relative to project_dir, or absolute. Where *_WP dirs are created.",
            )
            mamba_pre = st.text_input(
                "Micromamba prefix (openmm)",
                cfg.get("paths", {}).get("micromamba_prefix", ""),
            )
            westpa_pre = st.text_input(
                "WESTPA env prefix",
                cfg.get("paths", {}).get("westpa_env_prefix", ""),
            )
            cfg.setdefault("paths", {}).update({
                "project_dir": project_dir, "out_dir": out_dir,
                "micromamba_prefix": mamba_pre, "westpa_env_prefix": westpa_pre,
            })

        with st.expander("RCSB Search"):
            keywords = st.text_area(
                "Keywords (one per line)",
                "\n".join(cfg.get("rcsb_search", {}).get("keywords", [])),
            )
            kw_op = st.selectbox(
                "Keyword operator", ["contains_phrase", "exact_match"],
                index=(
                    0 if cfg.get("rcsb_search", {}).get("keyword_operator")
                    == "contains_phrase" else 1
                ),
            )
            organism = st.text_input(
                "Organism", cfg.get("rcsb_search", {}).get("organism", ""),
            )
            min_res = cfg.get("rcsb_search", {}).get("min_resolution")
            min_res_val = st.number_input(
                "Min resolution (A)",
                value=float(min_res) if min_res is not None else 0.0,
                step=0.1, format="%.1f", help="Leave 0 to skip.",
            )
            max_res = st.number_input(
                "Max resolution (A)",
                value=cfg.get("rcsb_search", {}).get("max_resolution", 2.5),
                step=0.1, format="%.1f",
            )
            ret_type = st.selectbox(
                "Return type", ["entry", "polymer_entity", "assembly"],
                index=["entry", "polymer_entity", "assembly"].index(
                    cfg.get("rcsb_search", {}).get("return_type", "entry"),
                ),
            )
            cfg.setdefault("rcsb_search", {}).update({
                "keywords": [k.strip() for k in keywords.splitlines() if k.strip()],
                "keyword_operator": kw_op, "organism": organism,
                "min_resolution": min_res_val if min_res_val > 0 else None,
                "max_resolution": max_res, "return_type": ret_type,
            })

        with st.expander("Slurm"):
            col1, col2 = st.columns(2)
            slurm = cfg.get("slurm", {})
            with col1:
                account = st.text_input("Account", slurm.get("account", "m4229"))
                constraint = st.text_input("Constraint", slurm.get("constraint", "gpu"))
                qos = st.text_input("QoS", slurm.get("qos", "regular"))
                walltime = st.text_input("Walltime", slurm.get("walltime", "48:00:00"))
            with col2:
                nodes = st.number_input("Nodes", value=slurm.get("nodes", 1), min_value=1, step=1)
                ntasks = st.number_input("Tasks per node", value=slurm.get("ntasks_per_node", 4), min_value=1, step=1)
                cpus = st.number_input("CPUs per task", value=slurm.get("cpus_per_task", 8), min_value=1, step=1)
                gpus = st.number_input("GPUs per task", value=slurm.get("gpus_per_task", 1), min_value=0, step=1)
            cfg["slurm"] = {
                "account": account, "constraint": constraint, "qos": qos,
                "walltime": walltime, "nodes": int(nodes),
                "ntasks_per_node": int(ntasks), "cpus_per_task": int(cpus),
                "gpus_per_task": int(gpus),
            }

        with st.expander("WESTPA"):
            westpa = cfg.get("westpa", {})
            target_it = st.number_input(
                "Target iterations",
                value=westpa.get("target_iterations", 12500),
                min_value=1, step=100,
            )
            max_wc = st.text_input(
                "Max run wallclock", westpa.get("max_run_wallclock", "72:00:00"),
            )
            c1, c2 = st.columns(2)
            with c1:
                pndim = st.number_input("pcoord_ndim", value=westpa.get("pcoord_ndim", 1), min_value=1, step=1)
                plen = st.number_input("pcoord_len", value=westpa.get("pcoord_len", 11), min_value=1, step=1)
            with c2:
                nbins = st.number_input("nbins", value=westpa.get("nbins", 9), min_value=1, step=1)
                btc = st.number_input("bin_target_counts", value=westpa.get("bin_target_counts", 6), min_value=1, step=1)
            cfg["westpa"] = {
                "target_iterations": int(target_it), "max_run_wallclock": max_wc,
                "pcoord_ndim": int(pndim), "pcoord_len": int(plen),
                "nbins": int(nbins), "bin_target_counts": int(btc),
            }

        with st.expander("OpenMM"):
            omm = cfg.get("openmm", {})
            c1, c2, c3 = st.columns(3)
            with c1:
                temp = st.number_input("Temperature (K)", value=omm.get("temperature", 300.0), step=1.0)
                ts = st.number_input("Timestep (fs)", value=omm.get("timestep", 4.0), step=0.5)
                fric = st.number_input("Friction (1/ps)", value=omm.get("friction", 1.0), step=0.1)
                pres = st.number_input("Pressure (atm)", value=omm.get("pressure", 1.0), step=0.1)
            with c2:
                baro = st.number_input("Barostat interval", value=omm.get("barostat_interval", 25), min_value=1, step=5)
                ctol = st.text_input("Constraint tolerance", str(omm.get("constraint_tolerance", 1e-6)))
                hmass = st.number_input("Hydrogen mass (amu)", value=omm.get("hydrogen_mass", 1.5), step=0.1)
            with c3:
                steps = st.number_input("Steps", value=omm.get("steps", 1000), min_value=1, step=100)
                save_s = st.number_input("Save steps", value=omm.get("save_steps", 100), min_value=1, step=10)
                gpu_p = st.selectbox(
                    "GPU precision", ["mixed", "single", "double"],
                    index=["mixed", "single", "double"].index(
                        omm.get("gpu_precision", "mixed"),
                    ),
                )
            ff = st.text_area(
                "Forcefield XMLs (one per line)",
                "\n".join(omm.get("forcefield", ["amber14-all.xml", "amber14/tip3pfb.xml"])),
            )
            cfg["openmm"] = {
                "temperature": float(temp), "timestep": float(ts),
                "friction": float(fric), "pressure": float(pres),
                "barostat_interval": int(baro),
                "constraint_tolerance": float(ctol),
                "hydrogen_mass": float(hmass), "steps": int(steps),
                "save_steps": int(save_s), "gpu_precision": gpu_p,
                "forcefield": [x.strip() for x in ff.splitlines() if x.strip()],
            }

        with st.expander("Preprocessing"):
            preproc = cfg.get("preprocessing", {})
            pad = st.number_input("Padding (nm)", value=preproc.get("padding_nm", 1.0), step=0.1)
            ionic = st.number_input(
                "Ionic strength (M)",
                value=preproc.get("ionic_strength_M", 0.15),
                step=0.01, format="%.3f",
            )
            ph = st.number_input("pH", value=preproc.get("ph", 7.0), step=0.1)
            cfg["preprocessing"] = {
                "padding_nm": float(pad),
                "ionic_strength_M": float(ionic),
                "ph": float(ph),
            }

        if st.button("Save Configuration", type="primary"):
            save_config(cfg)
            st.success("Saved to config.json")

    # ==================================================================
    # Tab 2 - RCSB Search (POST & GET /rcsbsearch/v2/query)
    # ==================================================================
    with tab_search:
        st.header("RCSB PDB Search")

        # --- Keyword autocomplete via /rcsbsearch/v2/suggest -----------
        with st.expander("Keyword suggestions (/v2/suggest)", expanded=False):
            sug_val = st.text_input("Type to autocomplete", key="sug_val")
            if sug_val.strip():
                sug_results, sug_err = rcsb_suggest(sug_val.strip())
                if sug_err:
                    st.error(sug_err)
                elif sug_results:
                    for attr, items in sug_results.items():
                        if items:
                            st.caption(attr)
                            for s in items[:5]:
                                label = re.sub(r"</?em>", "", s.get("text", ""))
                                st.write(f"- {label}")
                else:
                    st.caption("No suggestions.")

        # --- Config summary --------------------------------------------
        search_cfg = cfg.get("rcsb_search", {})
        st.write(f"**Keywords:** {', '.join(search_cfg.get('keywords', []))}")
        min_r = search_cfg.get("min_resolution")
        max_r = search_cfg.get("max_resolution", "-")
        res_str = f"{min_r}-{max_r} A" if min_r is not None else f"<={max_r} A"
        st.write(
            f"**Organism:** {search_cfg.get('organism', '-')} "
            f"| **Resolution:** {res_str}"
        )

        # --- Search controls -------------------------------------------
        col_r, col_s = st.columns(2)
        with col_r:
            rows_pp = st.number_input(
                "Results per page", value=25, min_value=1,
                max_value=10000, step=25, key="s_rows",
            )
        with col_s:
            sort_field = st.text_input(
                "Sort field (blank = relevance)", key="s_sort", value="",
            )

        req_opts: dict | None = None
        extra_opts = rows_pp != 25 or sort_field.strip()
        if extra_opts:
            req_opts = {}
            if rows_pp != 25:
                req_opts["paginate"] = {"start": 0, "rows": int(rows_pp)}
            if sort_field.strip():
                sort_dir = st.selectbox(
                    "Sort direction", ["desc", "asc"], key="s_sort_dir",
                )
                req_opts["sort"] = [
                    {"sort_by": sort_field.strip(), "direction": sort_dir},
                ]

        if st.button("Run Search", type="primary", key="btn_search"):
            payload = build_rcsb_payload(cfg)
            method = _auto_method(payload)
            with st.spinner("Querying RCSB..."):
                ids, raw, sent = execute_rcsb_search(
                    payload,
                    method=method,
                    request_options=req_opts,
                )
            st.session_state["search_ids"] = ids
            st.session_state["search_raw"] = raw
            st.session_state["search_sent"] = sent

        if "search_ids" in st.session_state:
            _display_results(
                st.session_state["search_ids"],
                st.session_state["search_raw"],
                "search",
            )

        # --- Raw request / response ------------------------------------
        if "search_sent" in st.session_state:
            with st.expander("Request payload"):
                st.json(st.session_state["search_sent"])
            with st.expander("Raw response"):
                raw = st.session_state["search_raw"]
                preview = dict(raw)
                rs = preview.get("result_set", [])
                if len(rs) > 20:
                    preview["result_set"] = rs[:20]
                    preview["_truncated"] = f"showing 20 of {len(rs)}"
                st.json(preview)

        # --- pdb_ids.json editor ---------------------------------------
        st.divider()
        st.subheader("pdb_ids.json")
        current_ids = load_pdb_ids()
        st.caption(f"{len(current_ids)} IDs loaded")
        edited = st.text_area(
            "Edit PDB IDs (JSON array)",
            json.dumps(current_ids),
            height=150,
            key="pdb_edit",
        )
        if st.button("Save pdb_ids.json", key="btn_save_pdb"):
            try:
                parsed = json.loads(edited)
                if isinstance(parsed, list):
                    save_pdb_ids(parsed)
                    st.success(f"Saved {len(parsed)} IDs")
                else:
                    st.error("Must be a JSON array")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # ==================================================================
    # Tab 3 - RCSB API (query builder, suggest, unreleased, metadata)
    # ==================================================================
    with tab_api:
        st.header("RCSB API Tools")

        # --- Custom query builder (POST & GET /v2/query) ---------------
        with st.expander("Custom query builder (/v2/query)", expanded=True):
            st.caption(
                "Build an arbitrary RCSB search query with multiple terminal nodes."
            )
            qnodes = st.session_state["query_nodes"]

            col_lo, col_rt = st.columns(2)
            with col_lo:
                logical_op = st.radio(
                    "Combine nodes with", ["and", "or"],
                    horizontal=True, key="q_logic",
                )
            with col_rt:
                q_return_type = st.selectbox(
                    "Return type",
                    ["entry", "polymer_entity", "assembly", "non_polymer_entity"],
                    key="q_ret",
                )

            for i, nd in enumerate(qnodes):
                st.markdown(f"---\n**Node {i + 1}**")
                c1, c2 = st.columns(2)
                with c1:
                    cur_svc = nd.get("service", "text")
                    if cur_svc not in SEARCH_SERVICES:
                        cur_svc = "text"
                    svc = st.selectbox(
                        "Service",
                        SEARCH_SERVICES,
                        index=SEARCH_SERVICES.index(cur_svc),
                        key=f"qn_svc_{i}",
                    )
                    nd["service"] = svc

                    if svc != "full_text":
                        attr = st.text_input(
                            "Attribute",
                            value=nd.get("attribute", ""),
                            key=f"qn_attr_{i}",
                            help="e.g. struct_keywords.pdbx_keywords",
                        )
                        nd["attribute"] = attr
                with c2:
                    if svc != "full_text":
                        cur_op = nd.get("operator", TEXT_OPERATORS[0])
                        if cur_op not in TEXT_OPERATORS:
                            cur_op = TEXT_OPERATORS[0]
                        op = st.selectbox(
                            "Operator", TEXT_OPERATORS,
                            index=TEXT_OPERATORS.index(cur_op),
                            key=f"qn_op_{i}",
                        )
                        nd["operator"] = op

                    val = st.text_input(
                        "Value",
                        value=str(nd.get("value", "")),
                        key=f"qn_val_{i}",
                    )
                    nd["value"] = val

            c_add, c_rm = st.columns(2)
            with c_add:
                if st.button("+ Add node", key="q_add"):
                    qnodes.append({
                        "service": "text", "attribute": "",
                        "operator": "contains_phrase", "value": "",
                    })
                    st.rerun()
            with c_rm:
                if len(qnodes) > 1 and st.button("- Remove last", key="q_rm"):
                    qnodes.pop()
                    st.rerun()

            if st.button("Execute query", type="primary", key="btn_cq"):
                terminals = []
                for n in qnodes:
                    tn = {"type": "terminal", "service": n["service"]}
                    params: dict = {"value": n["value"]}
                    if n["service"] != "full_text":
                        params["attribute"] = n["attribute"]
                        params["operator"] = n["operator"]
                    tn["parameters"] = params
                    terminals.append(tn)

                if len(terminals) == 1:
                    query = terminals[0]
                else:
                    query = {
                        "type": "group",
                        "logical_operator": logical_op,
                        "nodes": terminals,
                    }
                payload = {"query": query, "return_type": q_return_type}
                method = _auto_method(payload)
                with st.spinner("Running custom query..."):
                    ids, raw, sent = execute_rcsb_search(
                        payload, method=method,
                    )
                st.session_state["cq_ids"] = ids
                st.session_state["cq_raw"] = raw
                st.session_state["cq_sent"] = sent

            if "cq_ids" in st.session_state:
                _display_results(
                    st.session_state["cq_ids"],
                    st.session_state["cq_raw"],
                    "cq",
                )
                with st.expander("Request payload"):
                    st.json(st.session_state.get("cq_sent", {}))
                with st.expander("Raw response"):
                    st.json(st.session_state.get("cq_raw", {}))

        # --- Unreleased entry search (/v2/query/unreleased) ------------
        with st.expander("Unreleased entry search (/v2/query/unreleased)"):
            st.caption(
                "Search upcoming/unreleased PDB entries by title (text service only)."
            )
            unrel_kw = st.text_input("Search unreleased titles", key="unrel_kw")
            if st.button("Search unreleased", key="btn_unrel"):
                if not unrel_kw.strip():
                    st.warning("Enter a query.")
                else:
                    query = {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_repository_holdings_unreleased.title",
                            "operator": "contains_phrase",
                            "value": unrel_kw.strip(),
                        },
                    }
                    with st.spinner("Searching unreleased..."):
                        u_ids, u_raw = rcsb_search_unreleased(query)
                    st.session_state["unrel_ids"] = u_ids
                    st.session_state["unrel_raw"] = u_raw

            if "unrel_ids" in st.session_state:
                u_ids = st.session_state["unrel_ids"]
                u_raw = st.session_state["unrel_raw"]
                if u_ids:
                    st.success(f"{len(u_ids)} unreleased entries")
                    st.code(
                        ", ".join(u_ids[:100])
                        + ("..." if len(u_ids) > 100 else "")
                    )
                    st.download_button(
                        "Download unreleased IDs",
                        data=json.dumps(u_ids, indent=2),
                        file_name="unreleased_ids.json",
                        mime="application/json",
                        key="unrel_dl",
                    )
                elif u_raw.get("total_count") == 0:
                    st.info("No unreleased entries matched.")
                elif "error" in u_raw:
                    st.error(u_raw["error"])
                else:
                    st.json(u_raw)

        # --- Suggest / autocomplete (/v2/suggest) ----------------------
        with st.expander("Suggest / autocomplete (/v2/suggest)"):
            st.caption(
                "RCSB search-as-you-type suggestions. "
                "Status codes: 200 (results), 204 (none), 500 (error)."
            )
            sug2_val = st.text_input("Search text", key="sug2_val")
            sug2_attr_filter = st.text_input(
                "Limit to attributes (comma-separated, blank = all)",
                key="sug2_attrs",
            )
            if st.button("Get suggestions", key="btn_sug2"):
                if not sug2_val.strip():
                    st.warning("Enter a value.")
                else:
                    attrs = (
                        [a.strip() for a in sug2_attr_filter.split(",") if a.strip()]
                        if sug2_attr_filter.strip() else None
                    )
                    suggestions, err = rcsb_suggest(sug2_val.strip(), attrs)
                    if err:
                        st.error(err)
                    elif suggestions:
                        for attr, items in suggestions.items():
                            if items:
                                st.caption(attr)
                                for s in items:
                                    label = re.sub(r"</?em>", "", s.get("text", ""))
                                    st.write(f"- {label} (score: {s.get('score', '-')})")
                    else:
                        st.info("No suggestions (HTTP 204 No Content).")

        # --- Metadata schemas (/v2/metadata/*/schema) ------------------
        with st.expander("Metadata schemas (/v2/metadata/*/schema)"):
            st.caption(
                "Browse searchable attribute schemas. "
                "Endpoints: structure, chemical, unreleased. "
                "Status codes: 200 (schema), 404 (not found)."
            )
            schema_type = st.selectbox(
                "Schema", ["structure", "chemical", "unreleased"], key="meta_schema",
            )
            meta_filter = st.text_input(
                "Filter attributes (substring)", key="meta_filter",
            )
            if st.button("Fetch schema", key="btn_meta"):
                with st.spinner("Fetching schema..."):
                    schema, err = rcsb_get_metadata(schema_type)
                if err:
                    st.error(err)
                elif schema:
                    st.session_state["meta_data"] = schema
                else:
                    st.error("Empty schema.")

            if "meta_data" in st.session_state:
                schema = st.session_state["meta_data"]
                if meta_filter.strip():
                    filt = meta_filter.strip().lower()
                    filtered = {
                        k: v for k, v in schema.items()
                        if filt in k.lower()
                        or filt in json.dumps(v).lower()
                    }
                    st.write(f"**{len(filtered)}** matching keys")
                    st.json(filtered)
                else:
                    st.json(schema)

        # --- HTTP status code reference --------------------------------
        with st.expander("API status code reference"):
            for code, desc in sorted(RCSB_STATUS_MSG.items()):
                st.write(f"**{code}** - {desc}")

    # ==================================================================
    # Tab 4 - Pipeline
    # ==================================================================
    with tab_pipe:
        st.header("Pipeline Control")
        mode_label = "SSH" if detect_execution_mode() == "ssh" else "Local"
        st.info(f"Execution mode: **{mode_label}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            run_batch = st.button("Batch Setup", help="Runs batch_wp.sh")
        with col2:
            run_jobs = st.button("Run Jobs", help="Runs run_wp.sh")
        with col3:
            run_full = st.button(
                "Full Pipeline", help="Runs batch_wp.sh (which calls run_wp.sh)",
            )

        output_area = st.empty()

        if run_batch:
            run_script(cfg, "batch_wp.sh", output_area)
        elif run_jobs:
            run_script(cfg, "run_wp.sh", output_area)
        elif run_full:
            run_script(cfg, "batch_wp.sh", output_area)

    # ==================================================================
    # Tab 5 - Status
    # ==================================================================
    with tab_status:
        st.header("Simulation Status")
        if st.button("Refresh"):
            st.session_state["status_rows"] = scan_wp_dirs(cfg)

        rows = st.session_state.get("status_rows", [])
        if rows:
            target = cfg.get("westpa", {}).get("target_iterations", 12500)
            st.write(f"**{len(rows)}** targets | target iterations: **{target}**")
            st.dataframe(rows, use_container_width=True)

            done = sum(1 for r in rows if r["Status"] == "done")
            running = sum(1 for r in rows if r["Status"] == "running")
            error = sum(1 for r in rows if r["Status"] == "error")
            c1, c2, c3 = st.columns(3)
            c1.metric("Done", done)
            c2.metric("Running", running)
            c3.metric("Error", error)
        else:
            st.info("Click Refresh to scan for *_WP directories.")


if __name__ == "__main__":
    main()
