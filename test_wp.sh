#!/bin/bash
# Unified test script for batch_wp.sh and run_wp.sh
# Runs in isolated temp directory

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDIR=$(mktemp -d)
trap "rm -rf $TESTDIR" EXIT
cd "$TESTDIR"

TARGET_ITERATIONS=12500

color_char() {
    case "$1" in
    𐄂) tput setaf 1 ;;  # red
    +) tput setaf 2 ;;    # green
    ★) tput setaf 3 ;;      # yellow
    ✔) tput setaf 4 ;;      # blue
        *) tput sgr0 ;;
    esac
    printf "%s" "$1"
    tput sgr0
}
pad() {
  local width="$1" left="$2" sep="$3" right="$4"
  local pad=$((width - ${#left} - ${#right}))
  ((pad<1)) && pad=1

  for ((i=0; i<${#left}; i++)); do
    color_char "${left:i:1}"
  done

  printf "%*s" "$pad" "" | tr ' ' "$sep"

  for ((i=0; i<${#right}; i++)); do
    color_char "${right:i:1}"
  done

  printf "\n"
}

# Create mock westpa_template directory
mkdir -p westpa_template
for f in run.slurm.template west.cfg.template b.txt.template; do
  echo 'PDBID={{PDB_ID}}' > westpa_template/$f
  done
echo '# propagator' > westpa_template/openmm_explicit_rmsd_p_ca_propagator.py
echo '#!/bin/bash' > westpa_template/env.sh

# Create mock setup_wp.sh
cat > setup_wp.sh << 'SETUP'
#!/bin/bash
pdb_id="$1"
[ -z "$pdb_id" ] && exit 1
mkdir -p "${pdb_id}_WP"
echo "topology" > "${pdb_id}_WP/topology.pdb"
sed "s/{{PDB_ID}}/$pdb_id/g" westpa_template/run.slurm.template > "${pdb_id}_WP/run.slurm"
sed "s/{{PDB_ID}}/$pdb_id/g" westpa_template/west.cfg.template > "${pdb_id}_WP/west.cfg"
sed "s/{{PDB_ID}}/$pdb_id/g" westpa_template/b.txt.template > "${pdb_id}_WP/b.txt"
cp westpa_template/openmm_explicit_rmsd_p_ca_propagator.py "${pdb_id}_WP/"
cp westpa_template/env.sh "${pdb_id}_WP/"
chmod +x "${pdb_id}_WP/env.sh" "${pdb_id}_WP/run.slurm"
mkdir -p "${pdb_id}_WP/traj_segs"
echo "test" > "${pdb_id}_WP/west.h5"
exit 0
SETUP
chmod +x setup_wp.sh

cat > run_wp.sh << 'RUN'
#!/bin/bash
# stub: nothing to print for test
RUN
chmod +x run_wp.sh

echo '["1ABC", "2DEF", "3GHI", "4JKL", "5MNO", "6PQR"]' > pdb_ids.json

# Config with project_dir=TESTDIR so scripts use local paths
cat > config.json << 'CONFIG'
{"paths":{"project_dir":"__TESTDIR__","out_dir":"out"},"westpa":{"target_iterations":12500},"slurm":{"account":"x","constraint":"gpu","qos":"regular","walltime":"01:00:00","nodes":1,"ntasks_per_node":1,"cpus_per_task":1,"gpus_per_task":1}}
CONFIG
sed -i.bak "s|__TESTDIR__|$TESTDIR|g" config.json && rm -f config.json.bak
export HELIXNET_CONFIG_DIR="$TESTDIR"

# Pre-create some existing WP directories with various states (in out/ per config)
mkdir -p out
mkdir -p out/1ABC_WP && echo "test" > out/1ABC_WP/west.h5 && echo "#!/bin/bash" > out/1ABC_WP/run.slurm && chmod +x out/1ABC_WP/run.slurm

# Mock h5ls for iteration counts (paths may be out/1ABC_WP or 1ABC_WP)
h5ls() {
    case "$1" in
        *1ABC_WP*) echo "iter_00100 Group"; echo "iter_12500 Group" ;;  # done
        *2DEF_WP*) echo "iter_00100 Group"; echo "iter_05000 Group" ;;  # running
        *3GHI_WP*) echo "iter_00100 Group"; echo "iter_02500 Group" ;;  # needs submit
        *4JKL_WP*) echo "iter_00100 Group"; echo "iter_08000 Group" ;;  # running
        *5MNO_WP*) echo "iter_00100 Group"; echo "iter_10000 Group" ;;  # needs submit
        *6PQR_WP*) echo "iter_00100 Group"; echo "iter_12500 Group" ;;  # done
        *) return 1 ;;
    esac
}
export -f h5ls

SQUEUE_COUNTER="$PWD/.squeue_count"
echo 0 > "$SQUEUE_COUNTER"

squeue() {
    COUNT=$(<"$SQUEUE_COUNTER")
    if [[ "$*" == *"--noheader"* ]]; then
        echo "zn_prod  R    4:15:00  1 shared_mi 44911401 nid004125"
        echo "2DEF_WP  R    0:45:30  1 gpu_mi    44911403 nid002847"
        echo "4JKL_WP  R    1:22:15  1 shared_mi 44911404 nid003921"
        ((COUNT > 0)) && echo "3GHI_WP  PD   0:00:00  1 shared_mi 44911500 (Priority)"
        ((COUNT > 1)) && echo "5MNO_WP  PD   0:00:00  1 shared_mi 44911501 (Priority)"
    else
        echo "NAME     ST TIME       NODES PARTITION JOBID    NODELIST(REASON)"
        echo "zn_prod  R  4:15:00    1     shared_mi 44911401 nid004125"
        echo "2DEF_WP  R  0:45:30    1     gpu_mi    44911403 nid002847"
        echo "4JKL_WP  R  1:22:15    1     shared_mi 44911404 nid003921"
        ((COUNT > 0)) && echo "3GHI_WP  PD 0:00:00    1     shared_mi 44911500 (Priority)"
        ((COUNT > 1)) && echo "5MNO_WP  PD 0:00:00    1     shared_mi 44911501 (Priority)"
    fi
}
export -f squeue
export SQUEUE_COUNTER

sbatch() {
    echo $(($(<"$SQUEUE_COUNTER") + 1)) > "$SQUEUE_COUNTER"
}
export -f sbatch

echo ""
echo "=== Running batch_wp.sh -> run_wp.sh flow ==="
echo ""

# Run the real scripts directly (batch_wp invokes run_wp internally)
"$REPO_DIR/batch_wp.sh"

# Cleanup handled by trap
