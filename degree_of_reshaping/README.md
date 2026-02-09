### `degree_of_reshaping/`

#### > `degree_reshaped.py`
Primary analysis script for LUCS designs. Performs structural analyses for LUCS designs' Rosetta models and predicted structures (from models including AlphaFold2/3, RoseTTAFold2, ESMFold, etc.). Compares LUCS designs' Rosetta models and predicted structures to reference structures (such as the starting structure used in generating the LUCS designs).

**Analyses in degree_reshaped.py:**
- Helix RMSD calculations, using multiple helix definitions and alignment methods (helix RMSD is defined as the RMSD between the longest common helical stretches of two topologically matched helices, keeping the central helical residues and trimming the C-terminal residue where necessary).
- Helix displacement and angle (after aligning on beta sheet or non-loop-helix-loop residues) between design models, predicted structures, and reference structures
- Reshaped region (loop-helix-loop elements) RMSD, full backbone RMSD, and TM-score and prediction confidence scores (e.g., pLDDT and pAE)
- Secondary structure analysis with DSSP

**Usage:**
```bash
python degree_reshaped.py --lucs_data /path/to/designs \
    --af_data_dirs /path/to/af_predictions \
    --starting_structure_path reference.pdb \
    --starting_structure_insertion_points insertion_points.json \
    --outfolder ./output --outfile results.csv
```

- Note: insertion_points.json is in the format specified in [Loop-Helix-Loop Reshaping](https://github.com/Kortemme-Lab/loop_helix_loop_reshaping)

#### > `utils_degree_reshaped.py`
Utility functions for protein structural analysis:
- Secondary structure geometric analysis
- Structure alignment and RMSD
- Residue selection and manipulation in PyRosetta

This module is used by most analysis scripts in degree_of_reshaping/

#### > `all_by_all_rmsds.py`
Computes pairwise RMSD and TM-score matrices for all structure pairs in a dataset. Generates symmetric RMSD matrices and asymmetric TM-score matrices (normalized by each structure's length). Supports parallel processing for large datasets.

**Output Files:**
- `rmsd.npy`: all-by-all RMSD matrix
- `tm1.npy`: all-by-all TM-scores normalized by structure 1 length
- `tm2.npy`: all-by-all TM-scores normalized by structure 2 length
- `pdb_order.json`: PDB/design name ordering for matrix indexing

**Usage:**
```bash
python all_by_all_rmsds.py --pdb_dirs ./structures \
    --outdir ./matrices --num_tasks 10 --task_id $SGE_TASK_ID
```

#### > `all_by_all_helix_rmsds.py`
Similar to `all_by_all_rmsds.py` but calculates helix RMSD matrices. Calculates helix RMSD for multiple alignment methods. Helix RMSD is defined as the RMSD between the longest common helical stretches of two topologically matched helices (keeping the central helical residues and trimming the C-terminal residue where necessary).

**Usage:**
```bash
python all_by_all_helix_rmsds.py --data_dir ./structures \
    --include_designs_df ./designs_to_include.csv \
    --output_dir ./matrices --num_tasks 10 --task_id $SGE_TASK_ID
```

#### > `combine_matrices.py`
Combines partial all-by-all matrices from distributed calculations into complete matrices. Supports (i) combining multiple partial matrices from parallel jobs and (ii) filling missing values in an existing matrix.

**Usage:**
```bash
python combine_matrices.py --data_folder ./matrices \
    --file_prefixes rmsd tm1 tm2
```

#### > `helix_rmsds_to_ref.py`
Calculates helix-specific RMSDs between designs and a reference structure. Focuses on helix regions to evaluate reshaping quality independent of loop conformations.

#### > `tm_score_to_ref.py`
Calculates TM-scores between provided PDBs and a reference structure.
