"""Consolidated HelixNet test suite."""

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
import requests
import responses


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SAMPLE_CONFIG = {
    "execution": {"nersc_user": "testuser"},
    "paths": {
        "project_dir": "/tmp/helixnet_test",
        "out_dir": "out",
        "micromamba_prefix": "/fake/envs/openmm",
        "westpa_env_prefix": "/fake/envs/westpa_env",
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
    d = tempfile.mkdtemp(prefix="helixnet_test_")
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
    monkeypatch.setenv("HELIXNET_CONFIG_DIR", str(config_dir))
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
        config_path = tmp_dir / "config.json"
        monkeypatch.setattr(app, "CONFIG_PATH", config_path)
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
            env={**dict(os.environ), "HELIXNET_CONFIG_DIR": str(config_env)},
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "testuser"


class TestHelpers:
    def test_detect_execution_mode(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        monkeypatch.setattr("os.uname", lambda: types.SimpleNamespace(nodename="login.perlmutter.nersc.gov"))
        assert app.detect_execution_mode() == "local"

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
    def test_metadata_404(self, monkeypatch):
        app = _import_benchmark(monkeypatch)
        responses.add(responses.GET, app.RCSB_META_URLS["structure"], status=404)
        schema, err = app.rcsb_get_metadata("structure")
        assert schema is None
        assert "Not Found" in err


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
            'cd /tmp/helixnet_test && bash -lc "./run.sh batch"', get_pty=True
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
            f"s|{{{{GPU_PRECISION}}}}|{omm['gpu_precision']}|g; "
            f"s|{{{{FF_0}}}}|{omm['forcefield'][0]}|g; "
            f"s|{{{{FF_1}}}}|{omm['forcefield'][1]}|g"
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


class TestShellScripts:
    @pytest.mark.parametrize("script", ["run.sh", "test.sh", "westpa_template/env.sh"])
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
