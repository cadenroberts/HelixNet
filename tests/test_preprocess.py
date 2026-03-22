"""Tests for preprocess_pdb.py functions.

The module imports openmm/pdbfixer/rdkit at top level, so the entire test file
is skipped when those packages are unavailable.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import preprocess_pdb
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="openmm/pdbfixer/rdkit not installed")


@pytest.fixture(autouse=True)
def _skip_if_no_deps():
    if not HAS_DEPS:
        pytest.skip("openmm/pdbfixer/rdkit not installed")


class TestValidatePdbId:
    def test_valid_4char(self):
        preprocess_pdb.validate_pdb_id("1ABC")

    def test_valid_lowercase(self):
        preprocess_pdb.validate_pdb_id("1abc")

    def test_valid_mixed(self):
        preprocess_pdb.validate_pdb_id("1L2Y")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            preprocess_pdb.validate_pdb_id("")

    def test_too_short(self):
        with pytest.raises(ValueError):
            preprocess_pdb.validate_pdb_id("1AB")

    def test_too_long(self):
        with pytest.raises(ValueError):
            preprocess_pdb.validate_pdb_id("1ABCD")

    def test_special_chars(self):
        with pytest.raises(ValueError):
            preprocess_pdb.validate_pdb_id("1A-C")


class TestCreateFolder:
    def test_creates_structure(self, tmp_dir):
        os.chdir(tmp_dir)
        folder = str(tmp_dir / "TEST_WP")
        preprocess_pdb.create_folder(folder)
        assert os.path.isdir(folder)
        assert os.path.isdir(os.path.join(folder, "raw"))
        assert os.path.isdir(os.path.join(folder, "processed"))

    def test_existing_folder_no_error(self, tmp_dir):
        folder = str(tmp_dir / "TEST_WP")
        os.makedirs(folder)
        preprocess_pdb.create_folder(folder)


class TestGetRcsbLigandSmiles:
    def test_known_ligand(self):
        smiles = preprocess_pdb.get_rcsb_ligand_smiles("ATP")
        assert smiles is not None
        assert isinstance(smiles, str)
        assert len(smiles) > 5

    def test_invalid_comp_id(self):
        result = preprocess_pdb.get_rcsb_ligand_smiles("XX")
        assert result is None

    def test_nonexistent_comp_id(self):
        result = preprocess_pdb.get_rcsb_ligand_smiles("ZZZ")
        assert result is None or isinstance(result, str)


class TestLoadConfig:
    def test_loads_from_env_dir(self, config_env):
        cfg = preprocess_pdb.load_config()
        assert cfg["execution"]["nersc_user"] == "testuser"


class TestPrepareProtein:
    def test_1l2y_preprocessing(self, tmp_dir, config_env, sample_config):
        os.chdir(tmp_dir)
        sample_config["preprocessing"] = {"padding_nm": 1.0, "ionic_strength_M": 0.15, "ph": 7.0}
        with open(config_env / "config.json", "w") as f:
            json.dump(sample_config, f)

        preprocess_pdb.prepare_protein("1L2Y")

        wp_dir = tmp_dir / "1L2Y_WP"
        assert (wp_dir / "raw" / "1L2Y.pdb").exists()
        assert (wp_dir / "processed" / "1L2Y_processed.pdb").exists()
        assert (wp_dir / "processed" / "forcefield.json").exists()

        with open(wp_dir / "processed" / "forcefield.json") as f:
            ff = json.load(f)
        assert "amber14-all.xml" in ff

        raw_size = (wp_dir / "raw" / "1L2Y.pdb").stat().st_size
        proc_size = (wp_dir / "processed" / "1L2Y_processed.pdb").stat().st_size
        assert proc_size > raw_size
