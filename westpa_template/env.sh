module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi
module load cudatoolkit

export TMPDIR=$PSCRATCH
mkdir -p $TMPDIR

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'micromamba shell init' !!
export MAMBA_EXE='/global/homes/c/cawrober/micromamba/micromamba'
export MAMBA_ROOT_PREFIX='/global/homes/c/cawrober/micromamba'
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
#eval "$($MAMBA_EXE shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate --prefix HelixNet/micromamba_root/envs/westpa_env
# <<< mamba initialize <<<

export HDF5_USE_FILE_LOCKING=0
export MPI=1

export WEST_SIM_ROOT="$PWD"
export SIM_NAME=$(basename $WEST_SIM_ROOT)

export WM_N_WORKERS=1
