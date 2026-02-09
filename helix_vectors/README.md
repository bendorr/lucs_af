### `helix_vectors/`

#### > `calc_6d_helix_vectors.py`
Calculates 6D helix vectors combining 3 cartesian coordinates (helix backbone--N, Ca, C--centroid) and 3 direction (normalized carbonyl vector sum) components for alpha helices. Coordinates are calculated in a reference frame defined by beta sheet residues or user-specified projection residues on a reference structure (to which all other structures are aligned).

**Usage:**
```bash
python calc_6d_helix_vectors.py --ref_pdb reference.pdb \
    --ref_insertion_points ref_insertion_points.json \
    --designs_df_path designs.csv --output_dir ./output \
    --output_df_name helix_vectors.csv \
    --num_lhls 2 --align_by_non_bb_rem \
    --num_tasks 10 --task_id $SGE_TASK_ID
```

#### > `analyze_helix_vectors.py`
Analyzes helix and sheet coordinates (output by `calc_6d_helix_vectors.py`) to generate 3D visualizations and compute helix diversity by binning helix vectors by their cartesian and directional components.

**Usage:**
```bash
# Color by reshaped loop-helix-loop 1 vs. 2
python analyze_helix_vectors.py \
    --helix_coords_dir /path/to/helix_coords/2KL8/ \
    --sheet_coords_dir /path/to/sheet_coords/2KL8/ \
    --starting_structure_name 2KL8 \
    --designs_df_path /path/to/2KL8_lucs_designs.csv \
    --num_lhls 2 --num_strands 5 \
    --color_by indiv_#CBB2D7,#6B3E99 \
    --no_colorbar \
    --save_movie_frames \
    --elevation -137 \
    --azimuth -20 \
    --output_dir /path/to/output/
```
