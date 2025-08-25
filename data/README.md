# Organization of data

Benjamin Orr

For "An improved model for prediction of de novo designed proteins with diverse geometries"

Authors: Benjamin Orr*, Stephanie E. Crilly*, Deniz Akpinaroglu, Eleanor Zhu, Michael J. Keiser, Tanja Kortemme

Data files can be found on zenodo at DOI: 10.5281/zenodo.16938925

Code for AF2 fine-tuning and structural analysis can be found at github.com/bendorr/lucs_af

## Directory structure

```text
├── data/
│   ├── README.md/
│   ├── dataframes/
│   │   ├── fig1/
│   │   │   ├── lucs_df.csv
│   │   │   └── nmr_df.csv
│   │   ├── fig2/
│   │   │   ├── 39_cfps_sequences.csv
│   │   │   ├── af2_rank_1_vs_ros_reshaped_f2s_scores.csv
│   │   │   ├── chymotrypsin_ngs.tsv
│   │   │   ├── optimized_dna_seqs.xlsx
│   │   │   └── trypsin_ngs.tsv
│   │   ├── fig3/
│   │   │   ├── cath4p2_predictions/
│   │   │   │   ├── Classic_cath4p2.csv
│   │   │   │   ├── StableStructSplit_cath4p2.csv
│   │   │   │   └── Stable_cath4p2.csv
│   │   │   ├── de_novo_testset_predictions/
│   │   │   │   ├── StableStructSplit_de_novo_testset.csv
│   │   │   │   ├── StableUnstable_de_novo_testset.csv
│   │   │   │   └── Stable_de_novo_testset.csv
│   │   │   ├── loss_values/
│   │   │   │   ├── StableStructSplit_training_loss.csv
│   │   │   │   ├── StableStructSplit_validation_loss.csv
│   │   │   │   ├── StableUnstable_training_loss.csv
│   │   │   │   ├── StableUnstable_validation_loss.csv
│   │   │   │   ├── Stable_training_loss.csv
│   │   │   │   └── Stable_validation_loss.csv
│   │   │   ├── lucs_test_set_fraction_bins.csv
│   │   │   ├── lucs_test_set_predictions/
│   │   │   │   └── lucs_test_set_predictions.zip
│   │   │   └── struct_split_test_set_predictions
│   │   │       └── StableStructSplit_testset.csv
│   │   └── supplement/
│   │       └── rosetta_seq_lucs_df.csv
│   ├── ft_models/
│   │   ├── FineTunedAlphaFold2.ipynb
│   │   ├── README.md
│   │   └── examples
│   │       ├── expected_outputs
│   │       │   ├── StableStructSplit_model2_2e12A00/
│   │       │   └── StableStructSplit_model2_8tnm/
│   │       └── inputs
│   │           ├── 2e12A00.fa
│   │           └── 8tnm.fa
│   ├── helix_vectors/
│   │   ├── lucs_helix_coords_lhl0.json
│   │   ├── lucs_helix_coords_lhl1.json
│   │   ├── natural_helix_coords_lhl0.json
│   │   ├── natural_helix_coords_lhl1.json
│   │   ├── rfdiffusion_helix_coords_lhl0.json
│   │   ├── rfdiffusion_helix_coords_lhl1.json
│   │   ├── sheet_coords_2lv8_strand0.json
│   │   ├── sheet_coords_2lv8_strand1.json
│   │   ├── sheet_coords_2lv8_strand2.json
│   │   └── sheet_coords_2lv8_strand3.json
│   ├── notebooks/
│   │   ├── fig1a.ipynb
│   │   ├── fig1f.ipynb
│   │   └── fig2c-e.ipynb
│   ├── pairwise_helix_rmsds/
│   │   ├── 44_natural_rossmann_folds_helix_rmsds.npy
│   │   ├── lucs_helix_rmsds.npy
│   │   └── rfdiffusion_helix_rmsds.npy
│   └── pdbs/
│       ├── 10k_lucs_designs/
│       │   ├── af2_preds/
│       │   │   └── 10k_lucs_designs_af2_preds.tar.gz
│       │   ├── af3_preds/
│       │   │   └── af3_preds.zip
│       │   ├── design_models/
│       │   │   └── 10k_lucs_designs_rosetta_models.tar.gz
│       │   └── esmfold_preds/
│       │       └── 10k_lucs_designs_esmfold_preds.zip
│       ├── 44_lucs_designs_pdbs/
│       │   └── {DESIGN_ID}.pdb
│       ├── 44_natural_rossmann_folds_pdbs/
│       │   └── {PDB_ID}_{CHAIN_ID}.pdb
│       ├── 44_rfdiffusion_pdbs/
│       │   └── {DESIGN_ID}.pdb
│       ├── 4k_rosetta_seq_lucs_designs/
│       │   ├── 4k_rosetta_seq_lucs_af2_preds_pdbs.zip
│       │   └── 4k_rosetta_seq_lucs_design_models_pdbs.zip
│       ├── LUCS_test_set/
│       │   ├── Stable/
│       │   │   └── pred_10k_{FOLD_NAME}_with_Stable.zip
│       │   ├── StablePlusUnstable/
│       │   │   └── pred_10k_{FOLD_NAME}_with_StablePlusUnstable.zip
│       │   ├── StableStructSplit/
│       │   │   └── pred_10k_{FOLD_NAME}_with_StableStructSplit.zip
│       │   └── design_models/
│       │       └── LUCS_test_set_60k_design_models.zip
│       └── nmr_structs/
│           ├── 2lv8.pdb
│           ├── af2_preds/
│           ├── esmfold_preds/
│           ├── lucs_design_models/
│           ├── nmr_structs/
│           ├── of_preds/
│           ├── rf2_preds/
│           └── rgn2_preds/
```

## Data for each figure in the manuscript

### Fig 1A

data/pdbs/nmr_structs/2lv8.pdb with residues 31-51 and 58-76 highlighted

### Fig 1B

Distributions of helix RMSD values in data/pairwise_helix_rmsds/44_natural_rossmann_folds_helix_rmsds.npy, lucs_helix_rmsds.npy, and rfdiffusion_helix_rmsds.npy. Distributions of non-zero values are shown (matrix indices where i (row index) is less than or equal to j (column index) are set to zero).

### Fig 1C

Overlay of 44 PDB files for each of 3 groups of Rossmann fold proteins. Natural: data/pdbs/44_natural_rossmann_folds_pdbs. LUCS: data/pdbs/44_lucs_designs_pdbs. RFdiffusion: data/pdbs/44_rfdiffusion_pdbs

### Fig 1D

Helix vectors for the 44 examples in each of the 3 groups of Rossmann fold proteins. Natural: data/helix_vectors/natural_helix_coords_lhl0.json and natural_helix_coords_lhl1.json. LUCS: data/helix_vectors/lucs_helix_coords_lhl0.json and lucs_helix_coords_lhl1.json. RFdiffusion: data/helix_vectors/rfdiffusion_helix_coords_lhl0.json and rfdiffusion_helix_coords_lhl1.json. Each json file contains a list of python dictionaries. Each dictionary contains keys 'design_id' and 'helixN_coords', where N={0,1}. 'helixN_coords' are 6-element arrays, where the first 3 elements are the x, y, and z coordinates of the vector tail, and the last 3 elements are the x, y, and z components of the vector direction.

### Fig 1E

Overlay of structures found in data/pdbs/nmr_structs. NMR: data/pdbs/nmr_structs/nmr_structs. LUCS: pdbs/nmr_structs/lucs_design_models. AF2: pdbs/nmr_structs/af2_preds. 2LV8: data/pdbs/nmr_structs/2lv8.pdb.

Helix residues are shown in color. RO2_1: 35-48, 65-81. RO2_20: 35-45, 61-75. RO2_25: 34-51, 68-77.

### Fig 1G

Using the dataframe: data/dataframes/fig1/lucs_df.csv. X values: 'helix{j+1}_rmsd_af2_model{i}_to_2lv8_all_heavy_bb' where j={0,1} (helix index) and i={1,2,3,4,5} (AF2 model index). Y values: 'helix{j+1}_rmsd_design_to_2lv8_all_heavy_bb' where j={0,1} (helix index; each Y value is duplicated 5 times for its 5 corresponding AF2 predictions).

Overlaid NMR values are found in the dataframe: data/dataframes/fig1/nmr_df.csv. X values: 'helix{j}_rmsd_af2_model{i}_to_2lv8_all_heavy_bb' where j={0,1} (helix index) and i={1,2,3,4,5} (AF2 model index). Y values: 'helix{j+1}_rmsd_design_to_2lv8_all_heavy_bb' where j={0,1} (helix index; each Y value is duplicated 5 times for its 5 corresponding AF2 predictions).

### Figs 2B and 2C

Chymotrypsin dataframe: data/dataframes/fig2/chymotrypsin_ngs.tsv

Trypsin datafrane: data/dataframes/fig2/rypsin_ngs.tsv

'Name' column contains design IDs. Values ending in 'full_scramble_sequence' are in the 'Full scramble' group. Values ending in 'patterned_scramble_sequence' are in the 'Patterned scramble' group. Values ending in neither of these suffixes are in the 'Design' group. Values in the 'ec50' column are plotted on the y-axis. Entries with 'ec50_95ci' column values > 2 are excluded from the analysis.

### Fig 2D

Using the dataframe: data/dataframes/fig1/lucs_df.csv. Y values: 'best_af2_value_rmsd_reshaped_all_heavy_bb_design_to_af2' (these values are the best-of-5 AF2 predictions' RMSDs between the reshaped LHL residues of the AF2 prediction and the Rosetta design model. The "best" AF2 prediction has the lowest-of-5 RMSDs to the Rosetta design model in the reshaped LHLs). X values: 'best_af2_model_by_rmsd_reshaped_all_heavy_bb_design_to_af2_value_for_ave_plddt_af2' (these are the average pLDDT values of the "best" AF2 predictions selected for the y-values). Only the stable designs are shown, whose values in the 'stable_by_ec50' column == 'stable'.

### Fig 2E

Using the dataframe: data/dataframes/fig1/lucs_df.csv. AF2-Pass designs are those with 'best_af2_value_rmsd_reshaped_all_heavy_bb_design_to_af2' values (see Fig 2D description above) < 1.5 and 'best_af2_model_by_rmsd_reshaped_all_heavy_bb_design_to_af2_value_for_ave_plddt_af2' values > 85. AF2-Fail designs are any design that do not meet one or both of these criteria. X values: 'helix_rmsd_design_to_2lv8_all_heavy_bb'. Only the stable designs are shown, whose values in the 'stable_by_ec50' column == 'stable'.

### Fig 2F

Using the dataframe: data/dataframes/fig2/af2_rank_1_vs_ros_reshaped_f2s_scores.csv. X values: 'rmsd_bin'. Y values: 'ave_f2s_score'. Hue: 'model' ('model'={'Rosetta','af2_rank_1'}).

### Fig 3C

Using the dataframes contained in: data/dataframes/fig3/lucs_test_set_predictions/lucs_test_set_predictions.zip (this zip file contains 18 dataframes -- 6 LUCS tasks x 3 fine-tuned AF2 models). X values: LUCS task names (one for each of 5 unique folds diversified with LUCS, plus one as 2LTA had either 2 or 3 LHLs reshaped). Y values: number of examples for classic and fine-tuned models with 'MODEL_gt_rmsd' < 1.5, divided by 10,000 (as there are 10,000 examples for each LUCS task). Column names are 'classic_gt_rmsd' for Classic AF2 and 'ft_gt_rmsd' for fine-tuned AF2 models. Hue: model name, as found in the dataframe names (Classic AF2 RMSD values are duplicated in each fine-tuned AF2 dataframe).

### Fig 3D

Using the dataframe: data/dataframes/fig3/lucs_test_set_fraction_bins.csv. X values: 'Model' ('Model'={'AF2','Stable+Unstable','Stable','Stable Struct-Split'}). Y values: 'Fraction'. Hue: 'name' (LUCS task names, split by reshaped LHL number).

### Fig 3F

Using the dataframes: data/dataframes/fig3/cath4p2_predictions/Classic_cath4p2.csv and StableStructSplit_cath4p2.csv. X values: 'rmsd' column in Classic_cath4p2.csv. Y values: 'rmsd' column in StableStructSplit_cath4p2.csv.

### Fig 3G

Using the dataframe: data/dataframes/fig3/de_novo_testset_predictions/StableStructSplit_de_novo_testset.csv. X values: 'classic_gt_rmsd'. Y values: 'ft_gt_rmsd'. Hue: 'topology' ('topology'={'ab','a','b'}).
