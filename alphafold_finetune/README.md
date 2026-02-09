### `alphafold_finetune/`

#### > `af_classifier/`
Transfer learning with AlphaFold2: classification heads adapted from pLDDT head for protein property prediction. Used to predict experimentally measured protease resistance, but easily adapted for other classification tasks.

#### > `run_finetuning.py`
Fine-tune AlphaFold2 models on custom datasets. For transfer learning with AlphaFold2, see `af_classifier/`. Includes optional training with MSA and template inputs.

**Usage:**
```bash
python run_finetuning.py --data_dir /path/to/af_params \
    --train_dataset train.tsv --valid_dataset valid.tsv \
    --outprefix finetuned_model_2_ptm --model_name model_2_ptm \
    --save_steps 2000 --valid_steps 200 --apply_every 16 \
    --patience 20 --binder_loss_weight 0 --lr_coef 0.01 \
    --num_epochs 100
```

#### > `run_prediction.py`
AlphaFold2 inference for predicting target sequences. Can be run with or without MSA and template inputs.

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

#### > `train_utils.py`
Training utilities including (i) input, MSA, and template featurization for training examples and (ii) structure manipulation and coordinate transformations.

#### > `predict_utils.py`
Prediction utilities including (i) input, MSA, and template featurization for inference targets and (ii) structure manipulation and coordinate transformations.

#### > `analyze_finetuned_predictions.py`
Comprehensive analysis of RMSDs and confidence metrics between classic/fine-tuned AlphaFold2 predictions and ground truth structures. Generates dataframes with structural metrics, confidence scores, and PDB paths for downstream analysis.

**Analyses in analyze_finetuned_predictions.py:**
- RMSD between predictions and "ground truth" structures, over all and subsets of residues (e.g., reshaped loop-helix-loop elements in LUCS designs)
- Helix RMSDs between predicted structures and design models/reference structures
- Prediction confidence evaluation, for all and subsets of residues
- Classic vs. fine-tuned model performance w.r.t. classification tasks and structure prediction accuracy

#### > `analyze_checkpointed_models_metrics.py`
Analyzes metrics across AlphaFold2 training checkpoints to evaluate model improvement over fine-tuning, such as (i) RMSDs to ground truth structures, for all and subsets of residues, (ii) confidence score distributions, and (iii) Precision-Recall curves for classification tasks (e.g., stability prediction)

**Usage:**
```bash
python analyze_checkpointed_models_metrics.py --job_name experiment1 \
    --checkpoint_model_data_folder ./exp1_checkpoints --outfolder_parent ./exp1_analysis \
    --stability_data ./exp1_stability_data.csv \
    --rmsd_thresh 1.5 --rmsd_analysis \
    --plot_confidence_distributions --stability_classifier --precision_recall
```

#### > `read_loss_from_txt.py`
Utility for parsing and visualizing training and validation losses from text log files.
