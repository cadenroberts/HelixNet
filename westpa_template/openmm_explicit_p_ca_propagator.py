import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from openmm.app import PME, HBonds, PDBFile, ForceField, Simulation, HBonds
from openmm import MonteCarloBarostat, Platform, LangevinMiddleIntegrator, XmlSerializer
from openmm.unit import atmospheres, kelvin, nanometer, picosecond, femtosecond, kilojoule_per_mole
import mdtraj
import numpy as np
import os
import westpa
from westpa.core.states import BasisState, InitialState
from westpa.core.segment import Segment
from westpa.core.propagators import WESTPropagator
import time
import random
from datetime import datetime


class BasePropagator(WESTPropagator):
    
    def __init__(self, rc=None):
        super(BasePropagator, self).__init__(rc)
        self._load_config()
        self._init_pcoord_calculator()
    
    def _load_config(self):
        raise NotImplementedError
    
    def _get_save_format(self, config_path):
        config = self.rc.config
        for key in config_path:
            config = config.get(key, {})
        return config.get('save_format', 'dcd').lower()
    
    def _init_pcoord_calculator(self):
        pcoord_config = self._get_pcoord_config()
        if pcoord_config:
            pcoord_config = dict(pcoord_config)
            class_path = pcoord_config.pop("class")
            calculator_class = westpa.core.extloader.get_object(class_path)
            self.pcoord_calculator = calculator_class(**pcoord_config)
    
    def _get_pcoord_config(self):
        raise NotImplementedError
    
    def get_pcoord(self, state):
        raise NotImplementedError
    
    def propagate(self, segments):
        raise NotImplementedError
    
    def _get_segment_outdir(self, segment):
        segment_pattern = self.rc.config['west', 'data', 'data_refs', 'segment']
        return os.path.expandvars(segment_pattern.format(segment=segment))
    
    def _get_parent_outdir(self, segment):
        parent = Segment(n_iter=segment.n_iter - 1, seg_id=segment.parent_id)
        return self._get_segment_outdir(parent)
    
    def _finalize_segment(self, segment, starttime):
        segment.status = Segment.SEG_STATUS_COMPLETE
        segment.walltime = time.time() - starttime
    
    def _print_completion(self, num_segments, elapsed_time):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time}: Finished {num_segments} segments in {elapsed_time:0.2f}s")

def save_openmm_npz(output_dir, times, forces, energy_k, energy_u, positions=None):
    data = {
        'times': np.array(times),
        'forces': np.array(forces),
        'energy_k': np.array(energy_k),
        'energy_u': np.array(energy_u)
    }
    if positions is not None:
        data['positions'] = np.array(positions)
    
    np.savez(os.path.join(output_dir, 'seg.npz'), **data)

class OpenMMPropagator(BasePropagator):
    
    def __init__(self, rc=None):
        super(OpenMMPropagator, self).__init__(rc)
    
    def _load_config(self):
        config = self.rc.config['west']['openmm']
        self.temperature = float(config.get('temperature', 300.0))
        self.timestep = float(config.get('timestep', 2.0))
        self.friction = float(config.get('friction', 1.0))
        self.pressure = float(config.get('pressure', 1.0))
        self.barostatInterval = int(config.get('barostatInterval', 25))
        self.constraintTolerance = float(config.get('constraintTolerance', 1e-6))
        self.hydrogenMass = float(config.get('hydrogenMass', 1.5))
        self.implicit_solvent = config.get('implicit_solvent', False)
        self.steps = config['steps']
        self.save_steps = config['save_steps']
        self.save_format = self._get_save_format(['west', 'openmm'])
        
        try:
            platform = Platform.getPlatformByName('CUDA')
            self.num_gpus = int(config.get('num_gpus', 1))
            if self.num_gpus == -1:
                default_device = platform.getPropertyDefaultValue('CudaDeviceIndex')
                self.num_gpus = default_device.count(',') + 1 if ',' in default_device else 1
        except Exception:
            self.num_gpus = 1
        
        self.gpu_precision = config.get('gpu_precision', 'single')
        
        self.topology_path = os.path.expandvars(config['topology_path'])
        self.forcefield_files = config['forcefield']
        self.pdb = PDBFile(self.topology_path)
        self.forcefield = ForceField(*self.forcefield_files)
        self.nonbondedMethod = None
    
    def _get_pcoord_config(self):
        return self.rc.config['west']['openmm'].get('pcoord_calculator')
    
    def get_pcoord(self, state):
        if isinstance(state, BasisState):
            p_ca_indices = [atom.index for atom in self.pdb.topology.atoms() if atom.name == 'P' or atom.name == 'CA']
            positions = self.pdb.positions
            p_ca_positions = np.array([positions[i].value_in_unit(nanometer) for i in p_ca_indices])
            p_ca_positions = p_ca_positions[np.newaxis, :, :] * 10.0
            state.pcoord = self.pcoord_calculator.calculate(p_ca_positions)
            return
        elif isinstance(state, InitialState):
            raise NotImplementedError
        raise NotImplementedError
    
    def _get_next_gpu_index(self, segment_id):
        return segment_id % self.num_gpus
    
    def _get_platform(self, seg_id):
        try:
            platform = Platform.getPlatformByName('CUDA')
            gpu_index = self._get_next_gpu_index(seg_id)
            print(f"Using gpu {gpu_index}")
            properties = {'CudaDeviceIndex': str(gpu_index), 'Precision': self.gpu_precision}
        except Exception:
            print("CUDA not available, using CPU.")
            platform = Platform.getPlatformByName('CPU')
            properties = {}
        return platform, properties
    
    def _create_system(self):
        raise NotImplementedError
    
    def _create_simulation(self, seg_id):
        platform, properties = self._get_platform(seg_id)
        system = self._create_system()
        
        integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin,
            self.friction / picosecond,
            self.timestep * femtosecond
        )
        integrator.setConstraintTolerance(self.constraintTolerance)
        integrator.setRandomNumberSeed(random.randint(1, 1000000))
        
        return Simulation(self.pdb.topology, system, integrator, platform, properties)
    
    def _init_segment_state(self, simulation, segment):
        if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
            parent_outdir = self._get_parent_outdir(segment)
            state_file = os.path.join(parent_outdir, "seg.xml")
            
            print(f"Loading parent state from {state_file}")
            with open(state_file, 'r') as f:
                simulation.context.setState(XmlSerializer.deserialize(f.read()))
            
            state = simulation.context.getState(getPositions=True)
            initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            return np.array([initial_pos])
            
        elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
            print(f"Initializing new trajectory {segment.seg_id}")
            simulation.context.setPositions(self.pdb.positions)
            simulation.minimizeEnergy()
            simulation.context.setVelocitiesToTemperature(self.temperature)
            state = simulation.context.getState(getPositions=True)
            initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            return np.array([initial_pos])
        else:
            raise ValueError(f"Unsupported segment initpoint type: {segment.initpoint_type}")
    
    def _setup_reporters(self, simulation, segment_outdir):
        raise NotImplementedError
    
    def _run_simulation(self, simulation):
        print(f"Running {self.steps} steps")
        
        times, forces, energy_k, energy_u = [], [], [], []
        positions_list = []
        
        assert self.steps % self.save_steps == 0
        
        for i in range(self.steps // self.save_steps):
            simulation.step(self.save_steps)
            state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
            
            pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            positions_list.append(pos_nm)
            
            forces.append(state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer))
            times.append(state.getTime().value_in_unit(picosecond))
            energy_k.append(state.getKineticEnergy().value_in_unit(kilojoule_per_mole))
            energy_u.append(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
        
        return times, forces, energy_k, energy_u, positions_list
    
    def _save_final_state(self, simulation, segment_outdir):
        state = simulation.context.getState(
            getPositions=True, getVelocities=True, getForces=True,
            getEnergy=True, enforcePeriodicBox=True
        )
        with open(os.path.join(segment_outdir, "seg.xml"), 'w') as f:
            f.write(XmlSerializer.serialize(state))
    
    def _calculate_pcoord(self, segment_outdir, initial_pos):
        raise NotImplementedError
    
    def propagate(self, segments):
        starttime = time.time()
        simulation = self._create_simulation(segments[0].seg_id)
        
        for segment in segments:
            segment_outdir = self._get_segment_outdir(segment)
            os.makedirs(segment_outdir, exist_ok=True)
            
            initial_pos = self._init_segment_state(simulation, segment)
            self._setup_reporters(simulation, segment_outdir)
            times, forces, energy_k, energy_u, positions_list = self._run_simulation(simulation)
            
            if self.save_format == 'npz':
                save_openmm_npz(segment_outdir, times, forces, energy_k, energy_u, positions_list)
            else:
                save_openmm_npz(segment_outdir, times, forces, energy_k, energy_u)
            
            self._save_final_state(simulation, segment_outdir)
            segment.pcoord = self._calculate_pcoord(segment_outdir, initial_pos)
            self._finalize_segment(segment, starttime)
        
        self._print_completion(len(segments), time.time() - starttime)
        return segments

class BaseDCDReporter:
    
    def __init__(self, file, reportInterval, enforcePeriodicBox=False, append=False):
        self._file = file
        self._reportInterval = int(reportInterval)
        self._enforce = enforcePeriodicBox
        self._append = append
        self._dcd = None
    
    def describeNextReport(self, simulation):
        return (self._reportInterval, True, False, False, False, self._enforce)
    
    def _ensure_open(self, natoms):
        if self._dcd is None:
            self._dcd = mdtraj.formats.DCDTrajectoryFile(
                self._file, 'a' if self._append else 'w', force_overwrite=True
            )
    
    def _get_positions(self, state):
        raise NotImplementedError
    
    def report(self, simulation, state):
        xyz = self._get_positions(state)
        self._ensure_open(xyz.shape[1])
        self._dcd.write(xyz)
    
    def __del__(self):
        try:
            if self._dcd is not None:
                self._dcd.close()
        except Exception:
            pass


class FullDCDReporter(BaseDCDReporter):
    
    def _get_positions(self, state):
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        return pos_nm[np.newaxis, :, :]


class SoluteDCDReporter(BaseDCDReporter):
    
    def __init__(self, file, reportInterval, atom_indices, enforcePeriodicBox=False, append=False):
        super().__init__(file, reportInterval, enforcePeriodicBox, append)
        self._atom_indices = np.asarray(atom_indices, dtype=int)
    
    def _get_positions(self, state):
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        sel = pos_nm[self._atom_indices]
        return sel[np.newaxis, :, :]


def write_dcd_from_positions(filepath, positions, topology=None):
    if positions.ndim == 2:
        positions = positions[np.newaxis, :, :]
    
    with mdtraj.formats.DCDTrajectoryFile(filepath, 'w', force_overwrite=True) as dcd:
        dcd.write(positions)


def get_solute_indices(topology):
    solvent_resnames = {
        "HOH", "WAT", "TIP3", "TIP4", "TIP5", "SPC", "SOL",
        "NA", "K", "CL", "CL-", "BR", "I", "LI", "RB", "CS",
        "MG", "CA", "ZN", "MN", "FE", "CU", "SR", "CD"
    }
    
    solute_indices = []
    for atom in topology.atoms:
        r = atom.residue
        is_water = getattr(r, "is_water", False)
        is_mono = (len(list(r.atoms)) == 1)
        is_solvent = (r.name.upper() in solvent_resnames)
        
        if not (is_water or is_mono or is_solvent):
            solute_indices.append(atom.index)
    
    if len(solute_indices) == 0:
        raise RuntimeError("No solute atoms found")
    
    return np.array(solute_indices, dtype=int)

class OpenMMExplicitPropagator(OpenMMPropagator):
    
    def __init__(self, rc=None):
        super(OpenMMExplicitPropagator, self).__init__(rc)
        if self.implicit_solvent:
            raise ValueError("OpenMMExplicitPropagator requires implicit_solvent=False")
        self.nonbondedMethod = PME
        
        md_top_full = mdtraj.Topology.from_openmm(self.pdb.topology)
        self.solute_atom_indices = get_solute_indices(md_top_full)
        self.md_top_solute = md_top_full.subset(self.solute_atom_indices)
    
    def _create_system(self):
        system = self.forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod=self.nonbondedMethod
,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=self.hydrogenMass
        )
        system.addForce(MonteCarloBarostat(
            self.pressure * atmospheres,
            self.temperature * kelvin,
            self.barostatInterval
        ))
        return system
    
    def _setup_reporters(self, simulation, segment_outdir):
        if self.save_format == 'dcd':
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            simulation.reporters.clear()
            simulation.reporters.append(
                SoluteDCDReporter(dcd_path, self.save_steps, self.solute_atom_indices,
                                enforcePeriodicBox=False, append=False)
            )
    
    def _calculate_pcoord(self, segment_outdir, initial_pos):
        initial_pos_solute = initial_pos[:, self.solute_atom_indices, :]
        
        if self.save_format == 'dcd':
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            traj = mdtraj.load_dcd(dcd_path, top=self.md_top_solute)
            all_positions = np.concatenate([initial_pos_solute * 10, traj.xyz * 10])
        else:
            npz_path = os.path.join(segment_outdir, 'seg.npz')
            positions = np.load(npz_path)['positions']
            positions_solute = positions[:, self.solute_atom_indices, :]
            all_positions = np.concatenate([initial_pos_solute * 10, positions_solute * 10])
        
        p_ca_indices = [atom.index for atom in self.md_top_solute.atoms if atom.name == 'P' or atom.name == 'CA']
        p_ca_positions = all_positions[:, p_ca_indices, :]
        return self.pcoord_calculator.calculate(p_ca_positions)
