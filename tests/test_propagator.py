"""Tests for the WESTPA/OpenMM propagator module.

Requires mdtraj and numpy. Skipped entirely if mdtraj is unavailable.
Does NOT require GPU or openmm for most tests.
"""

import os
import sys
import tempfile

import pytest
import numpy as np

try:
    import mdtraj
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False

pytestmark = pytest.mark.skipif(not HAS_MDTRAJ, reason="mdtraj not installed")

if HAS_MDTRAJ:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "westpa_template"))
    from openmm_explicit_rmsd_p_ca_propagator import (
        RMSDProgressCoordinate,
        get_solute_indices,
        write_dcd_from_positions,
        SoluteDCDReporter,
        FullDCDReporter,
    )


def _make_topology(n_residues, names=None, res_names=None):
    """Build a simple mdtraj topology."""
    top = mdtraj.Topology()
    chain = top.add_chain()
    if names is None:
        names = ["CA"] * n_residues
    if res_names is None:
        res_names = ["ALA"] * n_residues
    for i in range(n_residues):
        res = top.add_residue(res_names[i], chain)
        top.add_atom(names[i], mdtraj.element.carbon, res)
    return top


class TestRMSDProgressCoordinate:
    def test_zero_rmsd_for_identical(self):
        n_atoms = 10
        coords = np.random.randn(1, n_atoms, 3).astype(np.float64) * 10.0
        pcoord = RMSDProgressCoordinate()
        result = pcoord.calculate(coords)
        assert result.shape == (1, 1)
        assert np.allclose(result, 0.0, atol=1e-5)

    def test_nonzero_rmsd_for_different(self):
        n_atoms = 10
        ref = np.zeros((1, n_atoms, 3), dtype=np.float64)
        shifted = ref.copy()
        shifted[:, :, 0] += 10.0
        data = np.concatenate([ref, shifted], axis=0)
        pcoord = RMSDProgressCoordinate()
        result = pcoord.calculate(data)
        assert result.shape == (2, 1)
        assert result[0, 0] < 0.1
        assert result[1, 0] > 0.5

    def test_output_shape_multi_frame(self):
        n_frames = 11
        n_atoms = 5
        data = np.random.randn(n_frames, n_atoms, 3).astype(np.float64) * 10.0
        pcoord = RMSDProgressCoordinate()
        result = pcoord.calculate(data)
        assert result.shape == (n_frames, 1)

    def test_with_reference_pdb(self, tmp_path):
        top = _make_topology(5, names=["P", "CA", "P", "CA", "P"],
                             res_names=["DA", "ALA", "DT", "GLY", "DC"])
        coords = np.random.randn(1, 5, 3).astype(np.float32) * 0.5
        traj = mdtraj.Trajectory(coords, top)
        pdb_path = str(tmp_path / "ref.pdb")
        traj.save_pdb(pdb_path)

        pcoord = RMSDProgressCoordinate(
            reference_pdb_path=pdb_path,
            atom_selection="name P or name CA",
        )
        assert pcoord.atom_indices is not None
        assert len(pcoord.atom_indices) == 5

        test_data = coords * 10.0
        result = pcoord.calculate(test_data)
        assert result.shape == (1, 1)

    def test_invalid_shape_raises(self):
        pcoord = RMSDProgressCoordinate()
        with pytest.raises(ValueError):
            pcoord.calculate(np.zeros((10, 3)))

    def test_multi_component(self):
        n_atoms = 5
        data = np.random.randn(3, n_atoms, 3).astype(np.float64) * 10.0
        pcoord = RMSDProgressCoordinate(components=[0, 0])
        result = pcoord.calculate(data)
        assert result.shape == (3, 2)
        assert np.allclose(result[:, 0], result[:, 1])


class TestGetSoluteIndices:
    def test_protein_only(self):
        top = _make_topology(3, names=["CA", "CA", "CA"], res_names=["ALA", "GLY", "LEU"])
        indices = get_solute_indices(top)
        assert len(indices) == 3
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_excludes_water(self):
        top = mdtraj.Topology()
        chain = top.add_chain()
        r1 = top.add_residue("ALA", chain)
        top.add_atom("CA", mdtraj.element.carbon, r1)
        r2 = top.add_residue("HOH", chain)
        top.add_atom("O", mdtraj.element.oxygen, r2)
        indices = get_solute_indices(top)
        assert len(indices) == 1
        assert indices[0] == 0

    def test_excludes_ions(self):
        top = mdtraj.Topology()
        chain = top.add_chain()
        r1 = top.add_residue("ALA", chain)
        top.add_atom("CA", mdtraj.element.carbon, r1)
        top.add_atom("CB", mdtraj.element.carbon, r1)
        r2 = top.add_residue("NA", chain)
        top.add_atom("NA", mdtraj.element.sodium, r2)
        indices = get_solute_indices(top)
        assert len(indices) == 2
        np.testing.assert_array_equal(indices, [0, 1])

    def test_all_solvent_raises(self):
        top = mdtraj.Topology()
        chain = top.add_chain()
        r = top.add_residue("HOH", chain)
        top.add_atom("O", mdtraj.element.oxygen, r)
        top.add_atom("H1", mdtraj.element.hydrogen, r)
        with pytest.raises(RuntimeError, match="No solute atoms"):
            get_solute_indices(top)


class TestWriteDcdFromPositions:
    def test_write_and_read(self, tmp_path):
        n_atoms = 10
        positions = np.random.randn(5, n_atoms, 3).astype(np.float32)
        filepath = str(tmp_path / "test.dcd")
        write_dcd_from_positions(filepath, positions)

        assert os.path.exists(filepath)
        top = _make_topology(n_atoms)
        traj = mdtraj.load_dcd(filepath, top=top)
        assert traj.n_frames == 5
        assert traj.n_atoms == n_atoms

    def test_2d_input_promoted(self, tmp_path):
        n_atoms = 5
        positions = np.random.randn(n_atoms, 3).astype(np.float32)
        filepath = str(tmp_path / "test2d.dcd")
        write_dcd_from_positions(filepath, positions)

        top = _make_topology(n_atoms)
        traj = mdtraj.load_dcd(filepath, top=top)
        assert traj.n_frames == 1


class TestSoluteDCDReporter:
    def test_write_solute_only(self, tmp_path):
        top = mdtraj.Topology()
        chain = top.add_chain()
        for i in range(5):
            r = top.add_residue("ALA", chain)
            top.add_atom("CA", mdtraj.element.carbon, r)
        for i in range(3):
            r = top.add_residue("HOH", chain)
            top.add_atom("O", mdtraj.element.oxygen, r)

        solute_indices = np.array([0, 1, 2, 3, 4])
        filepath = str(tmp_path / "solute.dcd")
        reporter = SoluteDCDReporter(filepath, 1, solute_indices)

        from unittest.mock import MagicMock
        from openmm.unit import nanometer
        state = MagicMock()
        positions = np.random.randn(8, 3).astype(np.float64)
        state.getPositions.return_value = positions * nanometer

        reporter.report(None, state)
        del reporter

        solute_top = top.subset(solute_indices)
        traj = mdtraj.load_dcd(filepath, top=solute_top)
        assert traj.n_frames == 1
        assert traj.n_atoms == 5
