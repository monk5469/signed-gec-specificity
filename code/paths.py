"""Location of the data tree used by all scripts.

Set the environment variable SGEC_DATA_ROOT to the directory that holds the data tree, or leave it unset to use
workspace/ in the repository root. Expected layout under the root:
    raw/          inputs (HCP files, the original authors' example data and structural connectivity, atlases)
    work/         intermediate parcellated time series
    results/p33/  outputs, freeze records and registration records
The scripts are run from the code/ directory (they hash sibling scripts by file name).
The comparisons with the authors' MATLAB code (vg_run.py, vh_run.py, vj1_run.py) call the MATLAB executable named by the environment
variable SGEC_MATLAB ("matlab" on the PATH if unset)."""
import os

DATA_ROOT = os.path.abspath(os.environ.get("SGEC_DATA_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                                                           "workspace")).replace(os.sep, "/").rstrip("/") + "/"

MATLAB_EXE = os.environ.get("SGEC_MATLAB") or "matlab"
