"""Tests for WESTPA template expansion.

Runs the same sed commands that setup_wp.sh uses to expand templates,
then verifies all {{PLACEHOLDER}} tokens are replaced with config values.
"""

import json
import os
import pathlib
import re
import subprocess

import pytest

from tests.conftest import REPO_ROOT, SAMPLE_CONFIG

TEMPLATE_DIR = REPO_ROOT / "westpa_template"
PLACEHOLDER_RE = re.compile(r"\{\{[A-Z_]+\}\}")


@pytest.fixture
def expanded_files(tmp_dir, sample_config):
    """Expand all templates using the same sed logic as setup_wp.sh."""
    cfg = sample_config
    pdb_id = "1XYZ"
    out_dir = str(tmp_dir / "out")
    os.makedirs(out_dir, exist_ok=True)

    slurm = cfg["slurm"]
    westpa = cfg["westpa"]
    omm = cfg["openmm"]
    ff = omm["forcefield"]

    sed_slurm = (
        f"s|{{{{PDB_ID}}}}|{pdb_id}|g; "
        f"s|{{{{ACCOUNT}}}}|{slurm['account']}|g; "
        f"s|{{{{CONSTRAINT}}}}|{slurm['constraint']}|g; "
        f"s|{{{{QOS}}}}|{slurm['qos']}|g; "
        f"s|{{{{WALLTIME}}}}|{slurm['walltime']}|g; "
        f"s|{{{{NODES}}}}|{slurm['nodes']}|g; "
        f"s|{{{{NTASKS}}}}|{slurm['ntasks_per_node']}|g; "
        f"s|{{{{CPUS}}}}|{slurm['cpus_per_task']}|g; "
        f"s|{{{{GPUS}}}}|{slurm['gpus_per_task']}|g"
    )

    sed_west = (
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
        f"s|{{{{FF_0}}}}|{ff[0]}|g; "
        f"s|{{{{FF_1}}}}|{ff[1]}|g; "
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

    sed_bstate = f"s|{{{{PDB_ID}}}}|{pdb_id}|g"
    sed_env = f"s|{{{{REPO_DIR}}}}|{REPO_ROOT}|g"

    files = {}

    for name, template, sed_expr in [
        ("run.slurm", "run.slurm.template", sed_slurm),
        ("west.cfg", "west.cfg.template", sed_west),
        ("b.txt", "b.txt.template", sed_bstate),
        ("env.sh", "env.sh", sed_env),
    ]:
        src = TEMPLATE_DIR / template
        if not src.exists():
            continue
        dst = tmp_dir / name
        result = subprocess.run(
            ["sed", sed_expr, str(src)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"sed failed for {name}: {result.stderr}"
        dst.write_text(result.stdout)
        files[name] = dst

    files["pdb_id"] = pdb_id
    files["out_dir"] = out_dir
    return files


class TestRunSlurmExpansion:
    def test_no_remaining_placeholders(self, expanded_files):
        content = expanded_files["run.slurm"].read_text()
        remaining = PLACEHOLDER_RE.findall(content)
        assert remaining == [], f"Unexpanded placeholders: {remaining}"

    def test_account_substituted(self, expanded_files):
        content = expanded_files["run.slurm"].read_text()
        assert "#SBATCH -A m4229" in content

    def test_pdb_id_in_job_name(self, expanded_files):
        content = expanded_files["run.slurm"].read_text()
        assert "1XYZ_WP" in content

    def test_gpu_count(self, expanded_files):
        content = expanded_files["run.slurm"].read_text()
        assert "--gpus-per-task=1" in content

    def test_walltime(self, expanded_files):
        content = expanded_files["run.slurm"].read_text()
        assert "48:00:00" in content

    def test_qos(self, expanded_files):
        content = expanded_files["run.slurm"].read_text()
        assert "regular" in content


class TestWestCfgExpansion:
    def test_no_remaining_placeholders(self, expanded_files):
        content = expanded_files["west.cfg"].read_text()
        remaining = PLACEHOLDER_RE.findall(content)
        assert remaining == [], f"Unexpanded placeholders: {remaining}"

    def test_topology_path(self, expanded_files):
        content = expanded_files["west.cfg"].read_text()
        assert "1XYZ_WP/processed/1XYZ_processed.pdb" in content

    def test_target_iterations(self, expanded_files):
        content = expanded_files["west.cfg"].read_text()
        assert "12500" in content

    def test_forcefield_files(self, expanded_files):
        content = expanded_files["west.cfg"].read_text()
        assert "amber14-all.xml" in content
        assert "amber14/tip3pfb.xml" in content

    def test_temperature(self, expanded_files):
        content = expanded_files["west.cfg"].read_text()
        assert "300.0" in content

    def test_timestep(self, expanded_files):
        content = expanded_files["west.cfg"].read_text()
        assert "4.0" in content

    def test_pcoord_ndim(self, expanded_files):
        content = expanded_files["west.cfg"].read_text()
        assert "pcoord_ndim: 1" in content

    def test_pcoord_len(self, expanded_files):
        content = expanded_files["west.cfg"].read_text()
        assert "pcoord_len: 11" in content

    def test_nbins(self, expanded_files):
        content = expanded_files["west.cfg"].read_text()
        assert "nbins: [9]" in content

    def test_propagator_class(self, expanded_files):
        content = expanded_files["west.cfg"].read_text()
        assert "OpenMMExplicitPropagator" in content


class TestBstateTxtExpansion:
    def test_no_remaining_placeholders(self, expanded_files):
        content = expanded_files["b.txt"].read_text()
        remaining = PLACEHOLDER_RE.findall(content)
        assert remaining == [], f"Unexpanded placeholders: {remaining}"

    def test_pdb_id_in_bstate(self, expanded_files):
        content = expanded_files["b.txt"].read_text()
        assert "1XYZ_processed.pdb" in content


class TestEnvShExpansion:
    def test_no_remaining_placeholders(self, expanded_files):
        content = expanded_files["env.sh"].read_text()
        remaining = PLACEHOLDER_RE.findall(content)
        assert remaining == [], f"Unexpanded placeholders: {remaining}"

    def test_repo_dir_substituted(self, expanded_files):
        content = expanded_files["env.sh"].read_text()
        assert str(REPO_ROOT) in content

    def test_valid_bash(self, expanded_files):
        result = subprocess.run(
            ["bash", "-n", str(expanded_files["env.sh"])],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"env.sh syntax error: {result.stderr}"
