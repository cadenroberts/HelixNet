"""Shell script validation tests.

- bash -n syntax check on all .sh files
- test.sh mock execution
- test.sh e2e structure verification (cannot run without NERSC creds)
"""

import os
import pathlib
import re
import subprocess

import pytest

from tests.conftest import REPO_ROOT

SHELL_SCRIPTS = [
    "run.sh",
    "test.sh",
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


class TestTestShMock:
    def test_runs_successfully(self):
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "test.sh"), "mock"],
            capture_output=True, text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"test.sh mock failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


class TestTestShE2EStructure:
    """Verify test.sh e2e has expected stages without running it."""

    def test_has_all_stages(self):
        content = (REPO_ROOT / "test.sh").read_text()
        expected_stages = [
            "RCSB API search",
            "benchmark.py preprocess",
            "run.sh setup",
            "sbatch",
            "Wait for",
            "Validate output",
        ]
        for stage in expected_stages:
            assert stage in content, f"Missing stage: {stage}"

    def test_has_cleanup(self):
        content = (REPO_ROOT / "test.sh").read_text()
        assert "cleanup" in content.lower()

    def test_has_summary(self):
        content = (REPO_ROOT / "test.sh").read_text()
        assert "print_summary" in content

    def test_has_ssh_connectivity_check(self):
        content = (REPO_ROOT / "test.sh").read_text()
        assert "ssh_cmd" in content
        assert "SSH OK" in content or "echo ok" in content

    def test_uses_set_euo_pipefail(self):
        content = (REPO_ROOT / "test.sh").read_text()
        assert "set -euo pipefail" in content

    def test_default_pdb_id(self):
        content = (REPO_ROOT / "test.sh").read_text()
        assert '1JEY' in content

    def test_stage_pass_fail_tracking(self):
        content = (REPO_ROOT / "test.sh").read_text()
        assert "stage_pass" in content
        assert "stage_fail" in content
        assert "pass=0" in content or "pass++" in content
        assert "fail=0" in content or "fail++" in content

    def test_exit_code_on_failure(self):
        content = (REPO_ROOT / "test.sh").read_text()
        assert "exit 1" in content
