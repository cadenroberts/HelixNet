"""Shell script validation tests.

- bash -n syntax check on all .sh files
- test_wp.sh execution
- test_pipeline.sh structure verification (cannot run without NERSC creds)
"""

import os
import pathlib
import re
import subprocess

import pytest

from tests.conftest import REPO_ROOT

SHELL_SCRIPTS = [
    "batch_wp.sh",
    "run_wp.sh",
    "setup_wp.sh",
    "run_ui.sh",
    "test_wp.sh",
    "test_pipeline.sh",
    "scripts/demo.sh",
    "westpa_template/env.sh",
]


class TestBashSyntax:
    @pytest.mark.parametrize("script", SHELL_SCRIPTS)
    def test_syntax_valid(self, script):
        path = REPO_ROOT / script
        if not path.exists():
            pytest.skip(f"{script} not found")
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Syntax error in {script}: {result.stderr}"


class TestTestWpSh:
    def test_runs_successfully(self):
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "test_wp.sh")],
            capture_output=True, text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"test_wp.sh failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


class TestTestPipelineShStructure:
    """Verify test_pipeline.sh has all expected stages without running it."""

    def test_has_all_stages(self):
        content = (REPO_ROOT / "test_pipeline.sh").read_text()
        expected_stages = [
            "RCSB API search",
            "preprocess_pdb.py",
            "setup_wp.sh",
            "sbatch",
            "Wait for",
            "Validate output",
        ]
        for stage in expected_stages:
            assert stage in content, f"Missing stage: {stage}"

    def test_has_cleanup(self):
        content = (REPO_ROOT / "test_pipeline.sh").read_text()
        assert "cleanup" in content.lower()

    def test_has_summary(self):
        content = (REPO_ROOT / "test_pipeline.sh").read_text()
        assert "print_summary" in content

    def test_has_ssh_connectivity_check(self):
        content = (REPO_ROOT / "test_pipeline.sh").read_text()
        assert "ssh_cmd" in content
        assert "SSH OK" in content or "echo ok" in content

    def test_uses_set_euo_pipefail(self):
        content = (REPO_ROOT / "test_pipeline.sh").read_text()
        assert "set -euo pipefail" in content

    def test_default_pdb_id(self):
        content = (REPO_ROOT / "test_pipeline.sh").read_text()
        assert '1JEY' in content

    def test_stage_pass_fail_tracking(self):
        content = (REPO_ROOT / "test_pipeline.sh").read_text()
        assert "stage_pass" in content
        assert "stage_fail" in content
        assert "PASS=" in content or "PASS++" in content
        assert "FAIL=" in content or "FAIL++" in content

    def test_exit_code_on_failure(self):
        content = (REPO_ROOT / "test_pipeline.sh").read_text()
        assert "exit 1" in content
