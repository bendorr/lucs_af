# Transfer Learning with AlphaFold2: Protease EC50 Classifier

Code for training classifier heads on top of AlphaFold2 (AF2). Optional unfreezing of some or
all layers in the base AF2 model.

Classifier heads are called "pEC50 heads" for their initial application in predicting
experimentally-determined protease EC50 values for de novo designed proteins, but these can be
trained on any class-labeled protein sequence/structure data.

Classifier heads have architectures matching pLDDT heads (and can be initialized from pLDDT weights)
with two linear layers on top. Like the pLDDT heads, they take in the final, latent query sequence
representation from the AlphaFold2 structure module.

**pEC50 Head Architecture:**
```
input_layer_norm → act_0 → act_1 → logits → act_2 → act_3
└─────── pLDDT head (4 layers) ─────┘   └─ pEC50 addition (2 layers) ─┘
```

- **`alphafold/`** - Modified AlphaFold2 with classification heads. The classification heads imitate
the pLDDT head architecture and are initialized from pLDDT head weights for transfer learning.

- **`run_finetuning_pEC50.py`** - train/fine-tune AlphaFold2 on class-labeled sequence/structure data
   - `--train_last_n_layers N` - Unfreeze only the last N layers of pEC50 (classifier) heads (omit to train all layers)
   - `--only_fit_pec50` - Only train the pEC50 heads and freeze the base AF2 model (default when no AF2 flags are used)
   - `--unfreeze_af2_model` - Train both the pEC50 heads and the entire base AF2 model
   - `--af2_params_to_train <file>` - Train only specific AF2 layers/modules (haiku-style names listed in text file, one per line)

- **`run_prediction_pEC50.py`** - run inference with the fine-tuned AF2 and store pEC50 head outputs

- **`train_utils_pEC50.py`**, **`predict_utils_pEC50.py`** - utils for training and inference.


## Examples:

### Train pEC50 heads only, keep base AF2 frozen (default):

```bash
python run_finetuning_pEC50.py \
    --data_dir /path/to/af_classifier/params \
    --train_csv train.csv \
    --valid_csv valid.csv \
    --num_epochs 10 \
    --outprefix pEC50_only
```

### Train the last layers of the pEC50 heads only:

```bash
python run_finetuning_pEC50.py \
    --data_dir /path/to/af_classifier/params \
    --train_csv train.csv \
    --valid_csv valid.csv \
    --train_last_n_layers 2 \
    --num_epochs 10 \
    --outprefix pEC50_last_two
```

### Fine-tune both the pEC50 heads and the base AlphaFold2 model:

```bash
python run_finetuning_pEC50.py \
    --data_dir /path/to/af_classifier/params \
    --train_csv train.csv \
    --valid_csv valid.csv \
    --unfreeze_af2_model \
    --num_epochs 10 \
    --outprefix full_model
```

### Unfreeze specific AF2 layers/modules:

```bash
python run_finetuning_pEC50.py \
    --data_dir /path/to/af_classifier/params \
    --train_csv train.csv \
    --valid_csv valid.csv \
    --af2_params_to_train subset_af2_params.txt \
    --num_epochs 10 \
    --outprefix subset_af2
```
