import json
import os
import pathlib
import shutil
import tempfile

import pytest

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
    """Create a temp directory with config.json written."""
    cfg_path = tmp_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(sample_config, f)
    return tmp_dir


@pytest.fixture
def config_env(config_dir, monkeypatch):
    """Set HELIXNET_CONFIG_DIR to a temp config directory."""
    monkeypatch.setenv("HELIXNET_CONFIG_DIR", str(config_dir))
    return config_dir


@pytest.fixture
def repo_root():
    return REPO_ROOT
