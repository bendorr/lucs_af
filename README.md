# Code for "An improved model for prediction of de novo designed proteins with diverse geometries"

Authors: Benjamin Orr*, Stephanie E. Crilly*, Deniz Akpinaroglu, Eleanor Zhu, Michael J. Keiser, Tanja Kortemme

[Paper](https://www.biorxiv.org/content/10.1101/2025.06.02.657515v1)

Data for this manuscript can be found on [zenodo](https://zenodo.org/records/16938925).

See instructions for running fine-tuned AF2 models in ft_models/ or on [zenodo](https://zenodo.org/records/15580792).

## Overview

Code for (i) analyzing LUCS (Loop-Helix-Loop Unit Combinatorial Sampling) designs (de novo designed proteins) and their predicted structures (AlphaFold2/3, RoseTTAFold2, ESMFold, etc.) and (ii) fine-tuning AlphaFold2.

Includes tools for:
- Fine-tuning AlphaFold2 models on custom datasets
- Fine-tuning classifier heads on top of AlphaFold2 for protein property prediction from sequence
- Running classic and fine-tuned AlphaFold2 predictions with and without templates
- Calculating protein structural diversity and similarity (RMSD, TM-score, helix displacement and angle)
- Analyzing and visualizing helix geometry/diversity, e.g., the extent to which LUCS helices were reshaped from their starting positions
- Analyzing structure prediction confidence metrics (pLDDT, pAE, pTM), structure-based metrics, and Rosetta metrics

## Directory Structure

#### `alphafold_finetune/`
Tools for fine-tuning AlphaFold2 models and analyzing predictions from both classic and fine-tuned models. Includes training scripts, prediction utilities, and analysis tools for evaluating model performance against ground truth (or, as in the publication, Rosetta-modeled) structures.

#### `data/`
Organization of data associated with the manuscript on zenodo and descriptions of data used in each manuscript figure.

#### `degree_of_reshaping/`
Analysis tools for calculating structural metrics for LUCS designs, e.g. between Rosetta models and AlphaFold predictions, or between these models and reference structures (such as LUCS starting structures, before loop-helix-loop reshaping). Includes utilities for RMSD calculations, helix geometry analysis, confidence metric evaluation, and all-by-all structural comparisons.

#### `env/`
YAML files containing dependencies for alphafold fine-tuning and Rosetta/structure-based analysis.

#### `ft_models/`
Instructions for running fine-tuned AF2 models from the paper using localcolabfold or Google Colab.

#### `helix_vectors/`
Tools for calculating and analyzing 6D helix vectors (3D position + 3D direction) to characterize helix geometry in protein structures. Used for quantifying helix diversity through helix position and orientation.

#### `tests/`
Unit tests for configuration management, RMSD calculations, and structural alignment.

---

## Scripts

### `alphafold_finetune/`

#### `run_finetuning.py`
Fine-tune AlphaFold2 models on custom datasets. Supports training with optional classification heads (pLDDT-based and pAE-based) for predicting experimentally determined properties such as protease resistance or protein-protein binding. Includes:
- Custom loss functions for structural accuracy and classification
- Optional template-based training
- Checkpoint saving and validation
- Weighted sampling (for class balancing in classification tasks) and data augmentation

**Usage:**
```bash
python run_finetuning.py --data_dir /path/to/af_params \
    --train_dataset train.tsv --valid_dataset valid.tsv \
    --outprefix finetuned_model_2_ptm --model_name model_2_ptm \
    --save_steps 2000 --valid_steps 200 --apply_every 16 \
    --patience 20 --binder_loss_weight 0 --lr_coef 0.01 \
    --num_epochs 100
```

#### `run_prediction.py`
Template-based AlphaFold2 inference for predicting target sequences. Uses provided templates to guide structure prediction and outputs predicted structures with confidence metrics.

**Usage:**
```bash
python run_prediction.py --targets targets.tsv \
    --data_dir /path/to/af_params \
    --outfile_prefix predictions/finetuned_model_2_ptm_predictions \
    --model_names model_2_ptm finetuned_model_2_ptm \
    --model_params_files classic /path/to/finetuned_model_2_ptm_params.pkl \
    --completed_pred_dir predictions/ \
    --ignore_identities
```

#### `train_utils.py`
Training utilities:
- Template and MSA featurization for training examples
- Structure manipulation and coordinate transformations
- Padding inputs

#### `predict_utils.py`
Prediction utilities:
- Template and MSA featurization for inference targets
- Structure manipulation and coordinate transformations
- Padding inputs

#### `analyze_finetuned_predictions.py`
Comprehensive analysis of RMSDs and confidence metrics between classic/fine-tuned AlphaFold2 predictions and ground truth structures. Generates dataframes with structural metrics, confidence scores, and PDB paths for downstream analysis.

**Key Analyses:**
- RMSD calculations between predictions and ground truth
- pLDDT and pAE confidence metric evaluation
- Helix RMSD calculations for LUCS designs
- Comparison of classic vs. fine-tuned model performance

#### `analyze_checkpointed_models_metrics.py`
Analyzes metrics across AlphaFold2 training checkpoints to evaluate model improvement during fine-tuning. Includes:
- RMSD analysis across training steps
- Confidence score distributions
- Precision-Recall curves for stability prediction
- Visualization of model performance over training

**Usage:**
```bash
python analyze_checkpointed_models_metrics.py --job_name experiment1 \
    --checkpoint_model_data_folder ./checkpoints --outfolder_parent ./analysis \
    --rmsd_analysis --stability_classifier --precision_recall
```

#### `read_loss_from_txt.py`
Utility for parsing and analyzing training loss from log files. Extracts training and validation losses from text logs, enabling visualization of training curves and identification of optimal checkpoints.

---

### `degree_of_reshaping/`

#### `degree_reshaped.py`
Primary analysis script for LUCS designs. Calculates comprehensive structural metrics and confidence scores by comparing Rosetta models and AlphaFold predictions against reference structures.

**Key Metrics:**
- Full backbone RMSD and reshaped region RMSD
- Helix-specific RMSD calculations
- TM-scores for global structural alignment
- Helix displacement and angle deviations
- pLDDT and pAE confidence scores
- Secondary structure analysis with DSSP

**Usage:**
```bash
python degree_reshaped.py --lucs_data /path/to/designs \
    --af_data_dirs /path/to/af_predictions \
    --starting_structure_path reference.pdb \
    --starting_structure_insertion_points insertion_points.json \
    --outfolder ./output --outfile results.csv
```

#### `utils_degree_reshaped.py`
Extensive utility library (2,347 lines) containing core functions for structural analysis:
- PDB/structure I/O operations
- RMSD calculation functions (multiple implementations)
- Helix and sheet geometry calculations
- Structural alignment utilities
- Residue selection and manipulation
- PyRosetta helper functions
- DataFrame manipulation for results

This module is used by most analysis scripts in this directory.

#### `all_by_all_rmsds.py`
Computes pairwise RMSD and TM-score matrices for all structure pairs in a dataset. Generates symmetric RMSD matrices and asymmetric TM-score matrices (normalized by each structure's length). Supports parallel processing for large datasets.

**Output Files:**
- `rmsd.npy`: Symmetric RMSD matrix
- `tm1.npy`: TM-scores normalized by structure 1
- `tm2.npy`: TM-scores normalized by structure 2
- `pdb_order.json`: Structure ordering for matrix indexing

**Usage:**
```bash
python all_by_all_rmsds.py --pdb_dirs ./structures \
    --outdir ./matrices --num_tasks 10 --task_id $SGE_TASK_ID
```

#### `all_by_all_helix_rmsds.py`
Similar to `all_by_all_rmsds.py` but calculates RMSD and TM-score matrices specifically for helix regions. Useful for analyzing helix-specific structural diversity in LUCS design libraries.

#### `combine_matrices.py`
Combines partial all-by-all matrices from distributed calculations into complete matrices. Supports two modes:
1. Combining multiple partial matrices from parallel jobs
2. Filling missing values in an existing matrix

**Usage:**
```bash
python combine_matrices.py --data_folder ./matrices \
    --file_prefixes rmsd tm1 tm2
```

#### `helix_rmsds_to_ref.py`
Calculates helix-specific RMSDs between designs and a reference structure. Focuses on helix regions to evaluate reshaping quality independent of loop conformations.

#### `tm_score_to_ref.py`
Calculates TM-scores between designs and reference structures. TM-score is a length-independent metric better suited than RMSD for comparing structures of different sizes or evaluating global topology.

#### `rmsd_LUCS_to_RosettaRelax.py`
Compares LUCS designs to Rosetta FastRelax structures. Evaluates how well designs match Rosetta-relaxed structures, useful for validating design quality and identifying outliers.

---

### `degree_of_reshaping/common/`

#### `structural_alignment.py`
Unified module for structural alignment operations. Consolidates Kabsch algorithm implementations and provides clean API for structure superposition.

#### `rmsd_calculations.py`
Consolidated RMSD calculation functions supporting both BioPython and PyRosetta backends. Provides multiple RMSD calculation modes:
- Full structure RMSD
- Backbone-only RMSD
- Residue subset RMSD
- Segment-wise RMSD

#### `pyrosetta_utils.py`
PyRosetta-specific helper functions including:
- PDB to PyRosetta residue number conversion
- Pose manipulation utilities
- Residue selector helpers

---

### `helix_vectors/`

#### `calc_6d_helix_vectors.py`
Calculates 6D helix vectors combining position (helix centroid) and direction (normalized carbonyl vector sum) for protein helices. All coordinates are calculated in a reference frame defined by beta sheet residues or user-specified projection residues.

**Applications:**
- Quantitative helix geometry comparison
- Analyzing helix repositioning in designs
- Characterizing helix orientation changes

**Usage:**
```bash
python calc_6d_helix_vectors.py --ref_pdb reference.pdb \
    --ref_insertion_points insertion_points.json \
    --designs_df_path designs.csv --output_dir ./output \
    --output_df_name helix_vectors.csv --num_lhls 2
```

#### `analyze_helix_vectors.py`
Analyzes helix vector data to extract geometric relationships and patterns. Processes output from `calc_6d_helix_vectors.py` to generate statistics, visualizations, and comparisons between design variants.

---

### `tests/`

#### `test_config.py`
Unit tests for configuration management system. Tests YAML loading, environment variable overrides, argument parsing integration, and default value handling.

#### `test_rmsd_calculations.py`
Tests for RMSD calculation functions across different implementations. Validates correctness of calculations and consistency between BioPython and PyRosetta backends.

#### `test_structural_alignment.py`
Tests for structural alignment algorithms including Kabsch superposition. Verifies optimal alignment calculation and rotation/translation matrix accuracy.

---

### Dependencies

- Python 3.9+
- NumPy, pandas, matplotlib
- BioPython
- PyRosetta
- JAX (for AlphaFold training)
- tmtools (TM-score calculation)
- scipy (statistical analysis)
- seaborn (visualization)
- DSSP (secondary structure assignment)
- argparse

## Author

Ben Orr

Version 2.0.0

## License

MIT License

Copyright (c) 2025 Ben Orr
