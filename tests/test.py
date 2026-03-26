"""Consolidated NDMS test suite."""

import importlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
from unittest.mock import MagicMock

import pytest
import responses


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SAMPLE_CONFIG = {
    "execution": {"nersc_user": "testuser"},
    "paths": {
        "project_dir": "/tmp/ndms_test",
        "out_dir": "out",
        "micromamba_prefix": "/fake/envs/openmm",
        "westpa_env_prefix": "/fake/envs/westpa_env",
        "mamba_exe": "/fake/bin/micromamba",
        "mamba_root_prefix": "/fake/micromamba_root",
    },
    "rcsb_search": {
        "keywords": ["DNA BINDING PROTEIN, DNA", "RNA"],
        "keyword_operator": "contains_phrase",
        "organism": "Homo sapiens",
        "min_resolution": None,
        "max_resolution": 2.5,
        "return_type": "entry",
    },
    "slurm": {
        "account": "m4229",
        "constraint": "gpu",
        "qos": "regular",
        "walltime": "48:00:00",
        "nodes": 1,
        "ntasks_per_node": 4,
        "cpus_per_task": 8,
        "gpus_per_task": 1,
    },
    "westpa": {
        "target_iterations": 12500,
        "max_run_wallclock": "72:00:00",
        "pcoord_ndim": 1,
        "pcoord_len": 11,
        "nbins": 9,
        "bin_target_counts": 6,
    },
    "openmm": {
        "temperature": 300.0,
        "timestep": 4.0,
        "friction": 1.0,
        "pressure": 1.0,
        "barostat_interval": 25,
        "constraint_tolerance": 1e-6,
        "hydrogen_mass": 1.5,
        "steps": 1000,
        "save_steps": 100,
        "gpu_precision": "mixed",
        "forcefield": ["amber14-all.xml", "amber14/tip3pfb.xml"],
    },
    "preprocessing": {
        "padding_nm": 1.0,
        "ionic_strength_M": 0.15,
        "ph": 7.0,
    },
}


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="ndms_test_")
    yield pathlib.Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_config():
    return json.loads(json.dumps(SAMPLE_CONFIG))


@pytest.fixture
def config_dir(tmp_dir, sample_config):
    cfg_path = tmp_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(sample_config, f)
    return tmp_dir


@pytest.fixture
def config_env(config_dir, monkeypatch):
    monkeypatch.setenv("NDMS_CONFIG_DIR", str(config_dir))
    return config_dir


def _import_benchmark(monkeypatch):
    st = types.ModuleType("streamlit")
    st.error = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.success = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "streamlit", st)
    import benchmark as _benchmark

    importlib.reload(_benchmark)
    return _benchmark


class TestConfig:
    def test_load_and_save_config(self, tmp_dir, sample_config, monkeypatch):
        app = _import_benchmark(monkeypatch)
        monkeypatch.setenv("NDMS_CONFIG_DIR", str(tmp_dir))
        app.save_config(sample_config)
        loaded = app.load_config()
        assert loaded["execution"]["nersc_user"] == "testuser"

    def test_pdb_ids_roundtrip(self, tmp_dir, monkeypatch):
        app = _import_benchmark(monkeypatch)
        pdb_path = tmp_dir / "pdb_ids.json"
        monkeypatch.setattr(app, "PDB_IDS_PATH", pdb_path)
        ids = ["1ABC", "2DEF", "3GHI"]
        app.save_pdb_ids(ids)
        assert app.load_pdb_ids() == ids

    def test_read_config_cli(self, config_env):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "benchmark.py"), "read-config", "execution.nersc_user"],
            capture_output=True,
            text=True,
            env={**dict(os.environ), "NDMS_CONFIG_DIR": str(config_env)},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "testuser"


class TestHelpers:
    def test_detect_execution_mode_local(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        monkeypatch.setattr("os.uname", lambda: types.SimpleNamespace(nodename="login.perlmutter.nersc.gov"))
        assert app.detect_execution_mode() == "local"

    def test_detect_execution_mode_ssh(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        monkeypatch.setattr("os.uname", lambda: types.SimpleNamespace(nodename="my-laptop.local"))
        assert app.detect_execution_mode() == "ssh"

    def test_auto_method(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        assert app._auto_method({"query": {"type": "terminal"}}) == "get"
        assert app._auto_method({"x": "a" * 5000}) == "post"

    def test_resolve_out_dir(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        base = pathlib.Path("/tmp/base")
        assert app._resolve_out_dir({"paths": {"out_dir": "out"}}, base) == base / "out"
        assert app._resolve_out_dir({"paths": {"out_dir": "/abs/out"}}, base) == pathlib.Path("/abs/out")

    def test_validate_pdb_id(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        app.validate_pdb_id("1ABC")
        with pytest.raises(ValueError):
            app.validate_pdb_id("BADID")


class TestRcsbApi:
    @responses.activate
    def test_execute_search_204(self, monkeypatch, sample_config):
        app = _import_benchmark(monkeypatch)
        responses.add(responses.POST, app.RCSB_SEARCH_URL, status=204)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert ids == []
        assert raw["total_count"] == 0

    @responses.activate
    def test_execute_search_200_post(self, monkeypatch, sample_config):
        app = _import_benchmark(monkeypatch)
        body = {"total_count": 2, "result_set": [{"identifier": "1ABC"}, {"identifier": "2DEF"}]}
        responses.add(responses.POST, app.RCSB_SEARCH_URL, json=body, status=200)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert ids == ["1ABC", "2DEF"]
        assert raw["total_count"] == 2

    @responses.activate
    def test_execute_search_get_method(self, monkeypatch, sample_config):
        app = _import_benchmark(monkeypatch)
        body = {"total_count": 1, "result_set": [{"identifier": "3GHI"}]}
        responses.add(responses.GET, app.RCSB_SEARCH_URL, json=body, status=200)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="get")
        assert ids == ["3GHI"]

    @responses.activate
    def test_execute_search_pagination(self, monkeypatch, sample_config):
        app = _import_benchmark(monkeypatch)
        page1 = {"total_count": 3, "result_set": [{"identifier": "1ABC"}, {"identifier": "2DEF"}]}
        page2 = {"total_count": 3, "result_set": [{"identifier": "3GHI"}]}
        responses.add(responses.POST, app.RCSB_SEARCH_URL, json=page1, status=200)
        responses.add(responses.POST, app.RCSB_SEARCH_URL, json=page2, status=200)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert set(ids) == {"1ABC", "2DEF", "3GHI"}
        assert raw["total_count"] == 3

    @responses.activate
    def test_execute_search_request_error(self, monkeypatch, sample_config):
        import requests as req
        app = _import_benchmark(monkeypatch)
        responses.add(responses.POST, app.RCSB_SEARCH_URL, body=req.ConnectionError("fail"))
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert ids == []
        assert "error" in raw

    @responses.activate
    def test_execute_search_400(self, monkeypatch, sample_config):
        app = _import_benchmark(monkeypatch)
        responses.add(responses.POST, app.RCSB_SEARCH_URL, json={"message": "bad"}, status=400)
        payload = app.build_rcsb_payload(sample_config)
        ids, raw, _ = app.execute_rcsb_search(payload, method="post")
        assert ids == []
        assert "error" in raw

    @responses.activate
    def test_metadata_404(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        responses.add(responses.GET, app.RCSB_META_URLS["structure"], status=404)
        schema, err = app.rcsb_get_metadata("structure")
        assert schema is None
        assert "Not Found" in err

    @responses.activate
    def test_metadata_200(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        responses.add(responses.GET, app.RCSB_META_URLS["structure"], json={"version": "2"}, status=200)
        schema, err = app.rcsb_get_metadata("structure")
        assert schema == {"version": "2"}
        assert err is None

    @responses.activate
    def test_suggest_success(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        body = {"suggestions": {"attr1": [{"value": "DNA"}]}}
        responses.add(responses.GET, app.RCSB_SUGGEST_URL, json=body, status=200)
        result, err = app.rcsb_suggest("DNA")
        assert err is None
        assert "attr1" in result

    @responses.activate
    def test_suggest_204(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        responses.add(responses.GET, app.RCSB_SUGGEST_URL, status=204)
        result, err = app.rcsb_suggest("nothing")
        assert result == {}
        assert err is None

    @responses.activate
    def test_search_unreleased_success(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        body = {"total_count": 1, "result_set": [{"identifier": "U001"}]}
        responses.add(responses.GET, app.RCSB_UNRELEASED_URL, json=body, status=200)
        ids, data = app.rcsb_search_unreleased({"type": "terminal", "service": "text", "parameters": {}})
        assert ids == ["U001"]

    @responses.activate
    def test_search_unreleased_error(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        responses.add(responses.GET, app.RCSB_UNRELEASED_URL, json={"message": "bad"}, status=400)
        ids, data = app.rcsb_search_unreleased({"type": "terminal"})
        assert ids == []
        assert "error" in data


class TestExecution:
    def test_run_remote_cmd(self, monkeypatch, sample_config):
        app = _import_benchmark(monkeypatch)
        mock_client = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"ok\n"
        mock_client.exec_command.return_value = (None, mock_stdout, None)
        monkeypatch.setattr(app, "_get_ssh_client", lambda cfg: mock_client)
        assert app.run_remote_cmd(sample_config, "echo ok") == "ok\n"

    def test_run_script_ssh_command(self, monkeypatch, sample_config):
        app = _import_benchmark(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "ssh")
        mock_client = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.__iter__ = lambda self: iter(["line1\n"])
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_client.exec_command.return_value = (None, mock_stdout, mock_stderr)
        monkeypatch.setattr(app, "_get_ssh_client", lambda cfg: mock_client)
        placeholder = MagicMock()
        placeholder.code = MagicMock()
        app.run_script(sample_config, "./run.sh batch", placeholder)
        mock_client.exec_command.assert_called_once_with(
            'cd /tmp/ndms_test && bash -lc "./run.sh batch"', get_pty=True
        )


class TestTemplateExpansion:
    def test_templates_expand_without_placeholders(self, sample_config, tmp_dir):
        template_dir = REPO_ROOT / "westpa_template"
        slurm = sample_config["slurm"]
        westpa = sample_config["westpa"]
        omm = sample_config["openmm"]
        pdb_id = "1XYZ"
        out_dir = str(tmp_dir / "out")
        os.makedirs(out_dir, exist_ok=True)

        sed_expr = (
            f"s|{{{{PDB_ID}}}}|{pdb_id}|g; "
            f"s|{{{{ACCOUNT}}}}|{slurm['account']}|g; "
            f"s|{{{{CONSTRAINT}}}}|{slurm['constraint']}|g; "
            f"s|{{{{QOS}}}}|{slurm['qos']}|g; "
            f"s|{{{{WALLTIME}}}}|{slurm['walltime']}|g; "
            f"s|{{{{NODES}}}}|{slurm['nodes']}|g; "
            f"s|{{{{NTASKS}}}}|{slurm['ntasks_per_node']}|g; "
            f"s|{{{{CPUS}}}}|{slurm['cpus_per_task']}|g; "
            f"s|{{{{GPUS}}}}|{slurm['gpus_per_task']}|g; "
            f"s|{{{{PROJECT_DIR}}}}|{out_dir}|g; "
            f"s|{{{{TARGET_ITERATIONS}}}}|{westpa['target_iterations']}|g; "
            f"s|{{{{MAX_RUN_WALLCLOCK}}}}|{westpa['max_run_wallclock']}|g; "
            f"s|{{{{PCOORD_NDIM}}}}|{westpa['pcoord_ndim']}|g; "
            f"s|{{{{PCOORD_LEN}}}}|{westpa['pcoord_len']}|g; "
            f"s|{{{{NBINS}}}}|{westpa['nbins']}|g; "
            f"s|{{{{BIN_TARGET_COUNTS}}}}|{westpa['bin_target_counts']}|g; "
            f"s|{{{{NUM_GPUS}}}}|{slurm['gpus_per_task']}|g; "
            f"s|{{{{GPU_PRECISION}}}}|{omm['gpu_precision']}|g"
        )
        out_file = tmp_dir / "run.slurm"
        result = subprocess.run(
            ["sed", sed_expr, str(template_dir / "run.slurm.template")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        out_file.write_text(result.stdout)
        assert "{{" not in out_file.read_text()

    def test_west_cfg_template_expands_fully(self, sample_config, tmp_dir):
        template_dir = REPO_ROOT / "westpa_template"
        slurm = sample_config["slurm"]
        westpa = sample_config["westpa"]
        omm = sample_config["openmm"]
        pdb_id = "1XYZ"
        out_dir = str(tmp_dir / "out")

        ff_list = "\n".join(f"      - {ff}" for ff in omm["forcefield"])

        sed_expr = (
            f"s|{{{{PDB_ID}}}}|{pdb_id}|g; "
            f"s|{{{{PROJECT_DIR}}}}|{out_dir}|g; "
            f"s|{{{{TARGET_ITERATIONS}}}}|{westpa['target_iterations']}|g; "
            f"s|{{{{MAX_RUN_WALLCLOCK}}}}|{westpa['max_run_wallclock']}|g; "
            f"s|{{{{PCOORD_NDIM}}}}|{westpa['pcoord_ndim']}|g; "
            f"s|{{{{PCOORD_LEN}}}}|{westpa['pcoord_len']}|g; "
            f"s|{{{{NBINS}}}}|{westpa['nbins']}|g; "
            f"s|{{{{BIN_TARGET_COUNTS}}}}|{westpa['bin_target_counts']}|g; "
            f"s|{{{{NUM_GPUS}}}}|{slurm['gpus_per_task']}|g; "
            f"s|{{{{GPU_PRECISION}}}}|{omm['gpu_precision']}|g; "
            f"s|{{{{TEMPERATURE}}}}|{omm['temperature']}|g; "
            f"s|{{{{TIMESTEP}}}}|{omm['timestep']}|g; "
            f"s|{{{{FRICTION}}}}|{omm['friction']}|g; "
            f"s|{{{{PRESSURE}}}}|{omm['pressure']}|g; "
            f"s|{{{{BAROSTAT_INTERVAL}}}}|{omm['barostat_interval']}|g; "
            f"s|{{{{CONSTRAINT_TOLERANCE}}}}|{omm['constraint_tolerance']}|g; "
            f"s|{{{{HYDROGEN_MASS}}}}|{omm['hydrogen_mass']}|g; "
            f"s|{{{{STEPS}}}}|{omm['steps']}|g; "
            f"s|{{{{SAVE_STEPS}}}}|{omm['save_steps']}|g"
        )
        out_file = tmp_dir / "west.cfg"
        result = subprocess.run(
            ["sed", sed_expr, str(template_dir / "west.cfg.template")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        content = result.stdout.replace("{{FF_LIST}}", ff_list)
        out_file.write_text(content)
        assert "{{" not in content, f"Unexpanded placeholders remain in west.cfg: {[m for m in __import__('re').findall(r'\\{\\{.*?\\}\\}', content)]}"

    def test_btxt_template_expands_fully(self, tmp_dir):
        template_dir = REPO_ROOT / "westpa_template"
        pdb_id = "1XYZ"
        sed_expr = f"s|{{{{PDB_ID}}}}|{pdb_id}|g"
        result = subprocess.run(
            ["sed", sed_expr, str(template_dir / "b.txt.template")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        content = result.stdout
        assert "{{" not in content
        assert f"processed/{pdb_id}_processed.pdb" in content

    def test_env_sh_template_expands_fully(self, tmp_dir):
        template_dir = REPO_ROOT / "westpa_template"
        sed_expr = (
            "s|{{REPO_DIR}}|/tmp/repo|g; "
            "s|{{MAMBA_EXE}}|/tmp/mamba|g; "
            "s|{{MAMBA_ROOT_PREFIX}}|/tmp/mamba_root|g"
        )
        result = subprocess.run(
            ["sed", sed_expr, str(template_dir / "env.sh.template")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        content = result.stdout
        assert "{{" not in content, f"Unexpanded placeholders remain in env.sh: {[m for m in __import__('re').findall(r'\\{\\{.*?\\}\\}', content)]}"


class TestPreprocessHelpers:
    def test_create_folder(self, tmp_dir, monkeypatch):
        app = _import_benchmark(monkeypatch)
        target = str(tmp_dir / "1ABC_WP")
        app.create_folder(target)
        assert (tmp_dir / "1ABC_WP" / "raw").is_dir()
        assert (tmp_dir / "1ABC_WP" / "processed").is_dir()
        app.create_folder(target)

    def test_strip_ansi(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        assert app.strip_ansi("\x1B[31mred\x1B[0m") == "red"
        assert app.strip_ansi("plain") == "plain"

    @responses.activate
    def test_get_rcsb_ligand_smiles_exc_success(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        body = {"data": {"chem_comp": {"rcsb_chem_comp_descriptor": {"SMILES_stereo": "CCO"}}}}
        responses.add(responses.GET, "https://data.rcsb.org/graphql", json=body, status=200)
        assert app.get_rcsb_ligand_smiles_exc("ATP") == "CCO"

    def test_get_rcsb_ligand_smiles_exc_invalid_id(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        with pytest.raises(RuntimeError, match="1-3 character"):
            app.get_rcsb_ligand_smiles_exc("TOOLONG")
        with pytest.raises(RuntimeError, match="1-3 character"):
            app.get_rcsb_ligand_smiles_exc("")

    @responses.activate
    def test_get_rcsb_ligand_smiles_exc_bad_response(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        responses.add(responses.GET, "https://data.rcsb.org/graphql", json={"data": None}, status=200)
        with pytest.raises(RuntimeError, match="Unexpected RCSB GraphQL"):
            app.get_rcsb_ligand_smiles_exc("ATP")

    @responses.activate
    def test_get_rcsb_ligand_smiles_returns_none_on_error(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        responses.add(responses.GET, "https://data.rcsb.org/graphql", json={"data": None}, status=200)
        assert app.get_rcsb_ligand_smiles("ATP") is None

    def test_validate_pdb_id_edge_cases(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        app.validate_pdb_id("1A2B")
        with pytest.raises(ValueError):
            app.validate_pdb_id("")
        with pytest.raises(ValueError):
            app.validate_pdb_id("AB")
        with pytest.raises(ValueError):
            app.validate_pdb_id("1AB!")

    def test_cli_read_config_bad_key(self, config_env):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "benchmark.py"), "read-config", "nonexistent.key"],
            capture_output=True, text=True,
            env={**dict(os.environ), "NDMS_CONFIG_DIR": str(config_env)},
        )
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_cli_no_command(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "benchmark.py"), "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_load_runtime_config_no_file(self, tmp_dir, monkeypatch):
        app = _import_benchmark(monkeypatch)
        monkeypatch.setenv("NDMS_CONFIG_DIR", str(tmp_dir / "nonexistent"))
        monkeypatch.setattr(app, "CONFIG_EXAMPLE_PATH", tmp_dir / "also_missing.json")
        with pytest.raises(FileNotFoundError, match="No config file found"):
            app.load_runtime_config()


class TestScanWpDirs:
    def test_scan_local_dirs(self, tmp_dir, sample_config, monkeypatch):
        app = _import_benchmark(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "local")
        out = tmp_dir / "out"
        out.mkdir()
        (out / "1ABC_WP").mkdir()
        (out / "2DEF_WP").mkdir()
        (out / "2DEF_WP" / "west.h5").write_bytes(b"\x00" * 100)
        cfg = dict(sample_config)
        cfg["paths"] = {"project_dir": str(tmp_dir), "out_dir": "out"}
        rows = app.scan_wp_dirs(cfg)
        assert len(rows) == 2
        pdb_ids = {r["PDB ID"] for r in rows}
        assert pdb_ids == {"1ABC", "2DEF"}
        abc = next(r for r in rows if r["PDB ID"] == "1ABC")
        assert abc["west.h5"] is False
        assert abc["Status"] == "error"

    def test_scan_local_empty(self, tmp_dir, sample_config, monkeypatch):
        app = _import_benchmark(monkeypatch)
        monkeypatch.setattr(app, "detect_execution_mode", lambda: "local")
        cfg = dict(sample_config)
        cfg["paths"] = {"project_dir": str(tmp_dir), "out_dir": "nonexistent"}
        rows = app.scan_wp_dirs(cfg)
        assert rows == []


class TestPropagator:
    def test_propagator_compiles(self):
        path = REPO_ROOT / "westpa_template" / "openmm_explicit_rmsd_p_ca_propagator.py"
        source = path.read_text()
        compile(source, str(path), "exec")


class TestShellScripts:
    @pytest.mark.parametrize("script", ["run.sh", "test.sh"])
    def test_shell_syntax(self, script):
        path = REPO_ROOT / script
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_mock_script_runs(self):
        result = subprocess.run(["bash", str(REPO_ROOT / "test.sh"), "mock"], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_e2e_structure_present(self):
        content = (REPO_ROOT / "test.sh").read_text()
        for stage in ["RCSB API search", "benchmark.py preprocess", "run.sh setup", "sbatch", "Wait for", "Validate output"]:
            assert stage in content
