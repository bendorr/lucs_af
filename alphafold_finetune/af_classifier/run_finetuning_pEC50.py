"""
run_finetuning_pEC50.py

Author: Ben Orr
Date: 2.22.24
Adapted from https://github.com/phbradley/alphafold_finetune

Fine-tune AlphaFold2 on a custom dataset, with optional training of a 
classification head (named "pEC50" head for predicting experimentally
determined protease EC50 "classes") on top of the model.

The classification heads are initialized from the pLDDT head architecture
and weights, with two linear layers appended to the end to predict protease 
EC50 classes for input sequences.

Parameters
----------
--data_dir : str, required
    Location of AlphaFold params/ folder
--train_dataset : str, required
    tsv file listing the dataset training samples. See phbradley/alphafold_finetune
    github README for format.
--valid_dataset : str, required
    tsv file listing the dataset validation samples
--outprefix : str
    Prefix for output filenames
--model_name : str
    AlphaFold model name

Binder Options:
--plddt_binder : flag
    Use pLDDT binder classifier
--plddt_binder_slope : float
    Slope for plddt-based binder head
--pae_binder_slope : float
    Slope for pae-based binder head
--binder_loss_weight : float
    Weight for binder classification loss
--binder_intercepts : float+
    Initial values for the switchpoint parameter in the logistic regression binder
    layer. This is the point where the prediction switches from binder to
    non-binder. Should be in values of 0.01*pLDDT or 0.1*PAE, where pLDDT ranges
    0-100 and PAE has values in the 2-20 ish range (domain is 0-31). See example
    command lines on the phbradley/alphafold_finetune github README.
--freeze_binder : flag
    Freeze binder model params
--only_fit_binder : flag
    Freeze alphafold model params
--freeze_everything : flag
    Dont apply the accumulated gradients

Model Architecture:
--crop_size : int
    Max size of training example; set this as low as possible for memory and speed
--extra_msa : int
    Number of extra MSA sequences
--msa_clusters : int
    Number of MSA cluster sequences
--num_evo_blocks : int
    Number of evoformer blocks
--struc_viol_weight : float
    Structural violation weight

Training:
--num_epochs : int
    Number of training epochs
--batch_size : int
    Batch size (only batch_size=1 tested)
--num_cpus : int
    Number of CPUs for multiprocessing
--fake_native_weight : float
    Weight to apply to structure loss when using a predicted structure as the
    ground truth
--random_recycling : flag
    if True, set num_iter_recycling randomly during training

Optimization:
--grad_norm_clip : float
    Clip gradient L2 norm to this value per update
--apply_every : int
    Accumulate gradients from this many batches before applying
--lr_coef : float
    Coefficient to multiply the learning rate
--no_ramp : flag
    Turn off the learning rate ramp-up

pEC50 Head Training:
--chymEC50_loss_weight : float
    Weight for Chymotrypsin EC50 head loss
--trypEC50_loss_weight : float
    Weight for Trypsin EC50 head loss
--train_last_n_layers : int
    Unfreeze only the last N layers of each pEC50 head (counting from the end)
--only_fit_pec50 : flag
    Unfreeze all pEC50 head layers and freeze base AF2 params (overrides train_last_n_layers)

AlphaFold2 Base Model Training:
--unfreeze_af2_model : flag
    Unfreeze the entire base AlphaFold2 model for training in addition to the pEC50 head
--af2_params_to_train : str
    Path to a txt file containing newline-separated AlphaFold2 parameter names to selectively
    unfreeze. Parameter names should match keys in the Haiku parameter tree. Ignored if
    --unfreeze_af2_model is True.

Checkpointing:
--save_steps : int
    Number of global_steps between model checkpoints
--print_steps : int
    Number of global_steps between printing averaged results
--valid_steps : int
    Number of global_steps between validation predictions
--patience : int
    Number of validation steps before early stopping. Training will terminate if
    validation loss does not decrease in <patience> validation epochs

Resume Training:
--resume_training : flag
    Resume fine-tuning of a fine-tuned model
--resume_params : str
    Path to params.pkl file from which to continue model training
--resume_train_step : int
    Global_step at which to resume fine-tuning. Use previous loss values before this
    step for early stopping.
--resume_train_loss : str
    txt file containing training loss of the previously fine-tuned model
--resume_valid_loss : str
    txt file containing validation loss of the previously fine-tuned model

Debugging:
--test_load : flag
    Loop through datasets to test loading
--notrain : flag
    Dont do training (debugging)
--no_valid : flag
    Skip validation steps
--dump_pdbs : flag
    Dump pdb files of modeled structures during training
--verbose : flag
    Verbose logging for debugging
--debug : flag
    Debug logging for debugging
--no_random : flag
    Set all random seeds to 0

Usage
-----
python3 run_finetuning_pEC50.py --data_dir /path/to/alphafold/params \
    --train_dataset train_samples.tsv --valid_dataset valid_samples.tsv \
    --outprefix my_finetune --model_name model_2_ptm --num_epochs 10
"""

import os
from os import popen
import sys
import shutil
import pandas as pd
import numpy as np
import pickle
import argparse
import optax
import jax
import jax.numpy as jnp
import haiku as hk
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.common.protein import Protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.model.all_atom import atom37_to_torsion_angles, atom37_to_frames
import torch

### Use WeightedRandomSampler to balance
### Protease EC50 classes.
from torch.utils.data import WeightedRandomSampler

import warnings
warnings.filterwarnings("ignore")
hk.vmap.require_split_rng = False

import train_utils_ec50 as train_utils
import predict_utils_pEC50 as predict_utils

print('done importing') ; sys.stdout.flush()


### Binder stuff (phbradley/alphafold_finetune) ################################

class PlddtBinderClassifier(hk.Module):
    """
    Avg pLDDT over peptide.

    Assumes we have defined in the input_features:
    * peptide_mask
    * binder_class_1hot
    * predicted_lddt dict
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name="PlddtBinderClassifier")

    def __call__(self, input_features):
        mask = input_features['peptide_mask']
        binder_class_1hot = input_features['binder_class_1hot']

        plddts = train_utils.compute_plddt_jax(
            input_features['predicted_lddt']['logits'])

        # multiply by 0.01 so goes from 0 to 1
        plddt = 0.01 * jnp.sum(plddts * mask, keepdims=True)/jnp.sum(mask)

        if FLAGS.only_fit_binder:
            plddt = jax.lax.stop_gradient(plddt)

        x_intercept_init_values = jnp.array(FLAGS.binder_intercepts, dtype=plddt.dtype)
        num_binder_classes = len(FLAGS.binder_intercepts)
        slope_init_value = FLAGS.plddt_binder_slope
        assert binder_class_1hot.shape[-1] == num_binder_classes

        x_intercept_init = lambda s,d:jnp.stack([x_intercept_init_values]*s[0])
        x_intercept = hk.get_parameter(
            "x_intercept", shape=[plddt.shape[-1], num_binder_classes],
            dtype=plddt.dtype, init = x_intercept_init)
        slope_init = lambda s,d:jnp.full(s, slope_init_value, dtype=d)
        slope = hk.get_parameter(
            "slope", shape=[plddt.shape[-1]], dtype=plddt.dtype, init=slope_init)
        if FLAGS.freeze_binder:
            x_intercept = jax.lax.stop_gradient(x_intercept)
            slope = jax.lax.stop_gradient(slope)

        assert x_intercept.shape == binder_class_1hot.shape
        x_intercept = jnp.sum(x_intercept * binder_class_1hot, axis=-1)

        binder_logits = (plddt - x_intercept)*slope
        nonbinder_logits = jnp.zeros(binder_logits.shape, binder_logits.dtype)
        logits = jnp.concatenate([nonbinder_logits, binder_logits], axis=-1)

        return logits, plddt


class PaeBinderClassifier(hk.Module):
    """
    Avg PAE between peptide and MHC.

    Assumes we have, defined in the input_features:
    * partner1_mask
    * partner2_mask
    * binder_class_1hot
    * predicted_aligned_error dict

    eg partner1_mask might be the MHC and partner2_mask might be the peptide

    symmetrizes: we sum [partner1_mask][partner2_mask] and also
    [partner2_mask][partner1_mask]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name="PaeBinderClassifier")

    def __call__(self, input_features):
        mask1 = input_features['partner1_mask']
        mask2 = input_features['partner2_mask']
        binder_class_1hot = input_features['binder_class_1hot']

        paes = train_utils.compute_predicted_aligned_error_jax(
            input_features['predicted_aligned_error']['logits'],
            input_features['predicted_aligned_error']['breaks'].ravel(),
        )['predicted_aligned_error']

        paes12 = jnp.sum((paes * mask1[...,None])*mask2[...,None,:], keepdims=True)
        paes21 = jnp.sum((paes * mask2[...,None])*mask1[...,None,:], keepdims=True)
        # multiply by 0.1 to reduce range
        pae = 0.1 * (paes12 + paes21)[0]/(2*jnp.sum(mask1)*jnp.sum(mask2))
        print('pae:', pae)

        if FLAGS.only_fit_binder:
            pae = jax.lax.stop_gradient(pae)

        x_intercept_init_values = jnp.array(FLAGS.binder_intercepts, dtype=pae.dtype)
        num_binder_classes = len(FLAGS.binder_intercepts)
        slope_init_value = FLAGS.pae_binder_slope

        x_intercept_init = lambda s,d:jnp.stack([x_intercept_init_values]*s[0])
        x_intercept = hk.get_parameter(
            "x_intercept", shape=[pae.shape[-1], num_binder_classes],
            dtype=pae.dtype, init = x_intercept_init)

        slope_init = lambda s,d:jnp.full(s, slope_init_value, dtype=d)
        slope = hk.get_parameter(
            "slope", shape=[pae.shape[-1]], dtype=pae.dtype, init=slope_init)
        if FLAGS.freeze_binder:
            x_intercept = jax.lax.stop_gradient(x_intercept)
            slope = jax.lax.stop_gradient(slope)

        assert x_intercept.shape == binder_class_1hot.shape
        x_intercept = jnp.sum(x_intercept * binder_class_1hot, axis=-1)

        binder_logits = (pae - x_intercept)*slope
        nonbinder_logits = jnp.zeros(binder_logits.shape, binder_logits.dtype)
        logits = jnp.concatenate([nonbinder_logits, binder_logits], axis=-1)

        return logits, pae


########### end of binder stuff (phbradley/alphafold_finetune) ##############


def softmax_cross_entropy(logits, labels):
    """Softmax cross entropy from logits and one-hot labels."""
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    return jnp.asarray(loss), jax.nn.softmax(logits)


def get_loss_fn_only_fit_base_af2(af2_model_params,
        key, processed_feature_dict, structure_flag, binder_loss_weight):
    """
    Loss for base AF2 params only.
    """
    classifier_dict = {}
    binder_labels = jnp.array(processed_feature_dict['labels'], dtype=jnp.float32)
    classifier_dict['binder_class_1hot'] = processed_feature_dict['binder_class_1hot']
    classifier_dict['peptide_mask'] = processed_feature_dict['peptide_mask']
    classifier_dict['partner1_mask'] = processed_feature_dict['partner1_mask']
    classifier_dict['partner2_mask'] = processed_feature_dict['partner2_mask']
    del processed_feature_dict['binder_class_1hot']
    del processed_feature_dict['peptide_mask']
    del processed_feature_dict['partner1_mask']
    del processed_feature_dict['partner2_mask']
    del processed_feature_dict['labels']

    predicted_dict, loss = model_runner.apply(
        af2_model_params, key, processed_feature_dict)

    classifier_dict.update(predicted_dict)

    if FLAGS.only_fit_binder:
        loss = jax.lax.stop_gradient(loss)

    fake_native_weight = jnp.array(FLAGS.fake_native_weight, jnp.float32)
    sf = jnp.array(structure_flag, jnp.float32)
    not_sf = jnp.ones(sf.shape, dtype=sf.dtype) - sf

    loss = (sf + not_sf*fake_native_weight)*loss[0]

    return loss, (predicted_dict)


def get_loss_fn_pEC50only(pEC50_head_params, base_af2_model_params,
        key, processed_feature_dict, structure_flag, binder_loss_weight):
    """
    pEC50-only loss with separated gradients for pEC50 head params.
    """
    af2_model_params = hk.data_structures.merge(base_af2_model_params, pEC50_head_params)

    classifier_dict = {}
    binder_labels = jnp.array(processed_feature_dict['labels'], dtype=jnp.float32)
    classifier_dict['binder_class_1hot'] = processed_feature_dict['binder_class_1hot']
    classifier_dict['peptide_mask'] = processed_feature_dict['peptide_mask']
    classifier_dict['partner1_mask'] = processed_feature_dict['partner1_mask']
    classifier_dict['partner2_mask'] = processed_feature_dict['partner2_mask']
    del processed_feature_dict['binder_class_1hot']
    del processed_feature_dict['peptide_mask']
    del processed_feature_dict['partner1_mask']
    del processed_feature_dict['partner2_mask']
    del processed_feature_dict['labels']

    predicted_dict, loss, ec50_loss = model_runner.apply(
        af2_model_params, key, processed_feature_dict)

    classifier_dict.update(predicted_dict)

    if FLAGS.only_fit_binder:
        loss = jax.lax.stop_gradient(loss)
    fake_native_weight = jnp.array(FLAGS.fake_native_weight, jnp.float32)
    sf = jnp.array(structure_flag, jnp.float32)
    not_sf = jnp.ones(sf.shape, dtype=sf.dtype) - sf

    loss = (sf + not_sf*fake_native_weight)*loss[0]
    loss += not_sf * ec50_loss[0]

    return loss, (predicted_dict)


def train_step(pEC50_head_params, base_af2_model_params, key, batch, structure_flag, binder_loss_weight):
    """
    Execute one training step for pEC50 head params only.
    """
    (loss, (predicted_dict)), grads = jax.value_and_grad(
        get_loss_fn_pEC50only, has_aux=True)(
            pEC50_head_params, base_af2_model_params, key, batch, structure_flag, binder_loss_weight)

    grads = jax.lax.pmean(grads, axis_name='model_ax')
    loss = jax.lax.pmean(loss, axis_name='model_ax')
    return loss, grads, predicted_dict


def get_loss_fn_with_af2(trainable_params, frozen_params, key, processed_feature_dict, structure_flag, binder_loss_weight):
    """
    Loss function for training both pEC50 and AF2 params.
    """
    af2_model_params = hk.data_structures.merge(frozen_params, trainable_params)
    
    classifier_dict = {}
    binder_labels = jnp.array(processed_feature_dict['labels'], dtype=jnp.float32)
    classifier_dict['binder_class_1hot'] = processed_feature_dict['binder_class_1hot']
    classifier_dict['peptide_mask'] = processed_feature_dict['peptide_mask']
    classifier_dict['partner1_mask'] = processed_feature_dict['partner1_mask']
    classifier_dict['partner2_mask'] = processed_feature_dict['partner2_mask']
    del processed_feature_dict['binder_class_1hot']
    del processed_feature_dict['peptide_mask']
    del processed_feature_dict['partner1_mask']
    del processed_feature_dict['partner2_mask']
    del processed_feature_dict['labels']

    predicted_dict, loss, ec50_loss = model_runner.apply(
        af2_model_params, key, processed_feature_dict)

    classifier_dict.update(predicted_dict)

    if FLAGS.only_fit_binder:
        loss = jax.lax.stop_gradient(loss)
    fake_native_weight = jnp.array(FLAGS.fake_native_weight, jnp.float32)
    sf = jnp.array(structure_flag, jnp.float32)
    not_sf = jnp.ones(sf.shape, dtype=sf.dtype) - sf

    loss = (sf + not_sf*fake_native_weight)*loss[0]
    loss += not_sf * ec50_loss[0]

    return loss, (predicted_dict)


def train_step_with_af2(trainable_params, frozen_params, key, batch, structure_flag, binder_loss_weight):
    """
    Execute one training step for both pEC50 and AF2 params.
    """
    (loss, (predicted_dict)), grads = jax.value_and_grad(
        get_loss_fn_with_af2, has_aux=True)(
            trainable_params, frozen_params, key, batch, structure_flag, binder_loss_weight)

    grads = jax.lax.pmean(grads, axis_name='model_ax')
    loss = jax.lax.pmean(loss, axis_name='model_ax')
    return loss, grads, predicted_dict


def norm_grads_per_example(grads, l2_norm_clip=0.1):
    """Normalize gradients by L2 norm clipping."""
    nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
    ### Changing neg to a jnp.ndarray
    total_grad_norm = jnp.linalg.norm(jnp.array([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads]))
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    grads = jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)
    return grads


def get_pec50_layer_names(params):
    """
    Extract and return all pEC50 head layer names from the parameter tree.
    """
    layer_names = []

    expected_layers = ['input_layer_norm', 'act_0', 'act_1', 'logits', 'act_2', 'act_3']

    def collect_ec50_layers(module_path, name, value):
        if 'EC50' in module_path:
            layer_name = name.split('/')[0] if '/' in name else name
            if layer_name in expected_layers:
                full_path = (module_path, layer_name)
                if full_path not in layer_names:
                    layer_names.append(full_path)

    for module_name, module_params in params.items():
        if isinstance(module_params, dict):
            for param_name in module_params.keys():
                collect_ec50_layers(module_name, param_name, module_params[param_name])

    def sort_key(item):
        module_path, layer_name = item
        module_priority = 0 if 'chym' in module_path.lower() else 1
        layer_priority = expected_layers.index(layer_name) if layer_name in expected_layers else 999
        return (module_priority, layer_priority)

    layer_names.sort(key=sort_key)
    return layer_names


def split_pec50_params_by_layers(pEC50_head_params, num_trainable_layers):
    """
    Split pEC50 head parameters into trainable (last N layers) and frozen parts.
    """
    all_layers = get_pec50_layer_names(pEC50_head_params)

    if num_trainable_layers <= 0:
        frozen_params = pEC50_head_params
        trainable_params = {}
        print(f"All pEC50 layers frozen (train_last_n_layers={num_trainable_layers})")
        return frozen_params, trainable_params

    if num_trainable_layers >= len(all_layers):
        frozen_params = {}
        trainable_params = pEC50_head_params
        print(f"All {len(all_layers)} pEC50 layers trainable (train_last_n_layers={num_trainable_layers} >= total layers)")
        return frozen_params, trainable_params

    trainable_layer_set = set(all_layers[-num_trainable_layers:])

    print(f"Training last {num_trainable_layers} of {len(all_layers)} pEC50 layers:")
    print(f"  Trainable layers: {[f'{m}/{n}' for m, n in trainable_layer_set]}")

    def is_frozen(module_path, param_name, param_value):
        layer_name = param_name.split('/')[0] if '/' in param_name else param_name
        return (module_path, layer_name) not in trainable_layer_set

    frozen_params, trainable_params = hk.data_structures.partition(
        is_frozen, pEC50_head_params)

    return frozen_params, trainable_params


def create_batch_from_dataset_row(row):
    """
    Create batch from pandas DataFrame row.
    """
    debug = FLAGS.debug
    verbose = FLAGS.verbose
    if True:
        print('create_batch_from_dataset_row:',
              getattr(row,'targetid','unk'),
              getattr(row,'pdbid','unk'))

    nres = len(row.target_chainseq.replace('/',''))
    if hasattr(row, 'target_trim_positions'):
        target_trim_positions = [int(x) for x in row.target_trim_positions.split(';')]
        if target_trim_positions != list(range(nres)):
            print('WARNING: target_trimming:',
                  [x for x in range(nres) if x not in target_trim_positions])
    else:
        target_trim_positions = list(range(nres))

    native_align = {int(x.split(':')[0]):int(x.split(':')[1])
                    for x in row.native_alignstring.split(';')}

    native_identities = getattr(row, 'native_identities', None)
    native_identities = None if pd.isna(native_identities) else native_identities
    native_len = getattr(row, 'native_len', None)
    native_len = None if pd.isna(native_len) else native_len

    random_seed = 0 if FLAGS.no_random else None
    batch = predict_utils.create_batch_for_training(
        row.target_chainseq, target_trim_positions, row.templates_alignfile,
        row.native_pdbfile, native_align, FLAGS.crop_size, model_runner,
        native_identities = native_identities,
        native_len = native_len,
        debug=debug,
        verbose=verbose,
        random_seed=random_seed,
    )

    num_binder_classes = len(FLAGS.binder_intercepts)
    peptide_positions = [int(x) for x in row.peptide_positions.split(';')]
    nonpeptide_positions = [x for x in range(nres) if x not in peptide_positions]
    assert all(0 <= x < nres for x in peptide_positions)
    assert row.binder in [0,1]
    if num_binder_classes>1:
        binder_class_int = int(row.binder_class)
    else:
        binder_class_int = 0
    assert 0 <= binder_class_int < num_binder_classes
    labels = np.ones(1, np.int32) if row.binder else np.zeros(1, np.int32)
    binder_class = np.full(1, binder_class_int, np.int32)
    batch['labels'] = np.eye(2)[labels]
    batch['binder_class_1hot'] = np.eye(num_binder_classes)[binder_class]
    assert batch['binder_class_1hot'].shape == (1, num_binder_classes)

    batch['peptide_mask'] = np.zeros((FLAGS.crop_size,))
    batch['peptide_mask'][peptide_positions] = 1.

    batch['partner1_mask'] = np.zeros((FLAGS.crop_size,))
    batch['partner1_mask'][nonpeptide_positions] = 1.

    batch['partner2_mask'] = np.zeros((FLAGS.crop_size,))
    batch['partner2_mask'][peptide_positions] = 1.

    batch['native_exists'] = bool(row.native_exists)

    # Adding chym_ec50, tryp_ec50, and ec50_class to the batch.
    batch['chym_ec50'] = float(row.chym_ec50)
    batch['tryp_ec50'] = float(row.tryp_ec50)
    batch['ec50_class'] = float(row.ec50_class)

    # Adding chym_ec50_class and tryp_ec50_class to the batch
    batch['chym_ec50_class'] = float(row.chym_ec50_class)
    batch['tryp_ec50_class'] = float(row.tryp_ec50_class)

    # Include targetid of current sample(s)
    batch['targetid'] = row.targetid

    print([k for k in batch])

    return batch


class CustomPandasDataset(torch.utils.data.Dataset):
    """Custom PyTorch dataset for loading from pandas DataFrame."""

    def __init__(self, df, loader):
        self.df = df.copy()
        self.loader = loader

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if FLAGS.debug:
            print('CustomPandasDataset:get_item: index=', index)
        out = self.loader(self.df.iloc[index])
        return out


def collate(samples):
    """Collate for DataLoader."""
    out_dict = {}
    for name in train_utils.list_a + train_utils.list_a_templates:
        values = [item[name][0,] for item in samples]
        out_dict[name] = np.stack(values, axis=0)
    for name in train_utils.list_b + train_utils.list_b_templates:
        values = [item[name] for item in samples]
        out_dict[name] = np.stack(values, axis=0)
    aatype_ = [item['aatype'][0,] for item in samples]
    out_dict['aatype_'] = np.stack(aatype_, axis=0)

    out_dict['binder_class_1hot' ] = [item['binder_class_1hot' ] for item in samples]
    out_dict['peptide_mask' ] = [item['peptide_mask' ] for item in samples]
    out_dict['partner1_mask'] = [item['partner1_mask'] for item in samples]
    out_dict['partner2_mask'] = [item['partner2_mask'] for item in samples]
    out_dict['labels'] = [item['labels'] for item in samples]
    out_dict['native_exists'] = [item['native_exists'] for item in samples]

    out_dict['chym_ec50'] = [item['chym_ec50'] for item in samples]
    out_dict['tryp_ec50'] = [item['tryp_ec50'] for item in samples]
    out_dict['ec50_class'] = [item['ec50_class'] for item in samples]
    out_dict['chym_ec50_class'] = [item['chym_ec50_class'] for item in samples]
    out_dict['tryp_ec50_class'] = [item['tryp_ec50_class'] for item in samples]
    out_dict['targetid'] = [item['targetid'] for item in samples]

    return out_dict


def prep_batch_for_step(batch, training):
    """Prep batch for training/validation step."""
    torsion_dict = atom37_to_torsion_angles(
        jnp.array(batch['aatype_']),
        jnp.array(batch['all_atom_positions']),
        jnp.array(batch['all_atom_mask']))
    batch['chi_mask'] = torsion_dict['torsion_angles_mask'][:,:,3:]
    sin_chi = torsion_dict['torsion_angles_sin_cos'][:,:,3:,0]
    cos_chi = torsion_dict['torsion_angles_sin_cos'][:,:,3:,1]
    batch['chi_angles'] = jnp.arctan2(sin_chi, cos_chi)
    rigidgroups_dict = atom37_to_frames(
        jnp.array(batch['aatype_']),
        jnp.array(batch['all_atom_positions']),
        jnp.array(batch['all_atom_mask']))
    batch.update(rigidgroups_dict)
    for key_, value_ in batch.items():
        if key_ in train_utils.list_a + train_utils.list_a_templates:
            batch[key_] = value_[:,None,]
        if key_ in train_utils.list_c:
            batch[key_] = value_[:,None,]
    for item_ in train_utils.pdb_key_list_int:
        batch[item_] = jnp.array(batch[item_], jnp.int32)
    if training:
        if FLAGS.random_recycling:
            batch['num_iter_recycling'] = jnp.array(np.tile(
                np.random.randint(0, model_config.model.num_recycle+1, 1)[None,],
                (batch_size, model_config.model.num_recycle)), jnp.int32)
            print('num_iter_recycling:', batch['num_iter_recycling'])
        else:
            print('not setting num_iter_recycling!!! will do',
                  model_config.model.num_recycle,'recycles')

    batch['binder_class_1hot'] = jnp.array(batch['binder_class_1hot'],
                                           dtype=jnp.float32)
    batch['peptide_mask' ] = jnp.array(batch['peptide_mask' ], dtype=jnp.float32)
    batch['partner1_mask'] = jnp.array(batch['partner1_mask'], dtype=jnp.float32)
    batch['partner2_mask'] = jnp.array(batch['partner2_mask'], dtype=jnp.float32)

    batch['chym_ec50'] = jnp.array([batch['chym_ec50']], dtype=jnp.float32)
    batch['tryp_ec50'] = jnp.array([batch['tryp_ec50']], dtype=jnp.float32)
    batch['ec50_class'] = jnp.array([batch['ec50_class']], dtype=jnp.float32)
    batch['chym_ec50_class'] = jnp.array([batch['chym_ec50_class']], dtype=jnp.float32)
    batch['tryp_ec50_class'] = jnp.array([batch['tryp_ec50_class']], dtype=jnp.float32)

    del batch['native_exists']

    ### avoid passing string dtype to jax pmap. I save this var 
    ### outside the function, in the training loop.
    del batch['targetid']

    if FLAGS.verbose:
        print('prep_batch_for_step_final_features:', ' '.join(batch.keys()))


def protein_from_prediction(batch, predicted_dict, b_factors=None):
    """Create Protein from prediction."""
    fold_output = predicted_dict['structure_module']
    if b_factors is None:
        b_factors = np.zeros_like(fold_output['final_atom_mask'][0,])

    return Protein(
        aatype=batch['aatype'][0,0,:],
        atom_positions=fold_output['final_atom_positions'][0,],
        atom_mask=fold_output['final_atom_mask'][0,],
        residue_index=batch['residue_index'][0,0,:] + 1,
        b_factors=b_factors)


def compute_valid_stats(valid_loader, replicated_params, jax_key, binder_loss_weight=1.0):
    """
    Compute validation statistics.
    """
    if FLAGS.no_valid:
        print('no valid stats')
        return

    temp_train_loss = []
    temp_binder_loss = []
    temp_lddt_ca = []
    temp_distogram = []
    temp_masked_msa = []
    temp_exper_res = []
    temp_pred_lddt = []
    temp_chi_loss = []
    temp_fape = []
    temp_sidechain_fape = []

    ### Saving raw FAPE values, as well as weigthed
    ### by structure_flag FAPE values.
    temp_raw_fape = []
    temp_raw_sidechain_fape = []

    all_mean_loss, all_mean_binder_loss, all_mean_fape = [], [], []

    ### Keep track of the mean Protease EC50 losses for the validation set.
    all_mean_chym_ec50_loss, all_mean_tryp_ec50_loss = [], []

    for n, batch in enumerate(valid_loader):
        print('test_epoch:', e, 'batch:', n)
        structure_flag = batch['native_exists'][0]
        targetid = batch['targetid'][0]
        binder_labels = batch['labels']
        prep_batch_for_step(batch, True)

        if FLAGS.no_random:
            subkey = jax.random.PRNGKey(0)
        else:
            jax_key, subkey = jax.random.split(jax_key)

        binder_model_params, af2_model_params = hk.data_structures.partition(
            lambda m, n, p: m[:9] != "alphafold", replicated_params)
        base_af2_model_params, pEC50_head_params = hk.data_structures.partition(
            lambda m, n, p: "EC50" not in m, af2_model_params)

        loss, grads, predicted_dict = jax.pmap(
            train_step, in_axes=(0,0,None,0,None,None), axis_name='model_ax')(
                pEC50_head_params, base_af2_model_params, subkey, batch, structure_flag, binder_loss_weight)

        binder_loss = [-1]
        binder_probs = -1
        binder_features = -1

        print('test_epoch_n=', e, n, 'loss=', loss[0],
              'lddt_ca=', np.mean(predicted_dict['predicted_lddt']['lddt_ca']),
              'fape=', np.mean(predicted_dict['structure_module']['fape']),
              'binder_probs=', binder_probs,
              'binder_loss=', binder_loss[0],
              'binder_features=', binder_features,
              'binder_labels=', binder_labels)

        temp_train_loss.append(np.mean(loss[0]))
        temp_binder_loss.append(np.mean(binder_loss[0]))
        temp_lddt_ca.append(np.mean(predicted_dict['predicted_lddt']['lddt_ca']))
        temp_distogram.append(np.mean(predicted_dict['distogram']['loss']))
        temp_masked_msa.append(np.mean(predicted_dict['masked_msa']['loss']))
        temp_pred_lddt.append(np.mean(predicted_dict['predicted_lddt']['loss']))
        temp_chi_loss.append(np.mean(predicted_dict['structure_module']['chi_loss']))

        ### Multiplying the fape by the native_exists (structure_flag), so
        ### that it is zero for unstable examples.
        temp_fape.append(np.mean(predicted_dict['structure_module']['fape']) * structure_flag)
        temp_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']) * structure_flag)
        ### Saving raw fape losses, too
        temp_raw_fape.append(np.mean(predicted_dict['structure_module']['fape']))
        temp_raw_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']))

        ## Making a critical change to these loss values. Saving these variables as the
        ## Loss for the current sample, and not the mean over the entire epoch. Doing this by changing
        ## mean_loss = round(float(np.mean(temp_train_loss)),4) to
        ## mean_loss = round(float(np.mean(temp_train_loss[-1])),4)
        mean_loss = round(float(np.mean(temp_train_loss[-1])),4)
        mean_binder_loss = round(float(np.mean(temp_binder_loss[-1])),4)
        lddt_ca = round(float(np.mean(temp_lddt_ca[-1])),4)
        distogram = round(float(np.mean(temp_distogram[-1])),4)
        masked_msa = round(float(np.mean(temp_masked_msa[-1])),4)
        fape = round(float(np.mean(temp_fape[-1])),4)
        sidechain_fape = round(float(np.mean(temp_sidechain_fape[-1])),4)
        chi_loss = round(float(np.mean(temp_chi_loss[-1])),4)

        ### Save the validation chym_ec50 and tryp_ec50 losses
        chym_ec50_loss = round(float(predicted_dict['predicted_chym_ec50']['loss']),4)
        tryp_ec50_loss = round(float(predicted_dict['predicted_tryp_ec50']['loss']),4)

        ### Save the validation predicted chym_ec50 and tryp_ec50 classes
        chym_ec50_pred = round(float(predicted_dict['predicted_chym_ec50']['pred_ec50_class']),4)
        tryp_ec50_pred = round(float(predicted_dict['predicted_tryp_ec50']['pred_ec50_class']),4)

        ### Print out and save the weighted binder and Protease EC50 losses
        weighted_chym_ec50_loss = round(float(chym_ec50_loss * FLAGS.chymEC50_loss_weight), 4)
        weighted_tryp_ec50_loss = round(float(tryp_ec50_loss * FLAGS.trypEC50_loss_weight), 4)
        weighted_binder_loss = 0

        raw_fape = round(float(np.mean(temp_raw_fape[-1])),4)
        raw_sidechain_fape = round(float(np.mean(temp_raw_sidechain_fape[-1])),4)

        print(f'test_Step: {global_step} {n} loss: {mean_loss} binder_loss: {mean_binder_loss} \
lddt: {lddt_ca} fape: {fape} sc_fape: {sidechain_fape} chi: {chi_loss} disto: {distogram} \
msa: {masked_msa} chym_ec50_loss: {chym_ec50_loss} tryp_ec50_loss: {tryp_ec50_loss} \
weighted_binder_loss: {weighted_binder_loss} weighted_chym_ec50_loss: {weighted_chym_ec50_loss} \
weighted_tryp_ec50_loss: {weighted_tryp_ec50_loss} chym_ec50_pred: {chym_ec50_pred} tryp_ec50_pred: {tryp_ec50_pred} \
true_chym_ec50: {batch["chym_ec50"][0]} true_tryp_ec50: {batch["tryp_ec50"][0]} \
true_chym_ec50_class: {batch["chym_ec50_class"][0]} true_tryp_ec50_class: {batch["tryp_ec50_class"][0]} \
sample: {targetid} raw_fape: {raw_fape} raw_sc_fape: {raw_sidechain_fape}', flush=True)

        all_mean_loss.append(mean_loss)
        all_mean_binder_loss.append(mean_binder_loss)
        all_mean_fape.append(fape)

        ### Keep track of chym_ec50 and tryp_ec50 losses
        all_mean_chym_ec50_loss.append(predicted_dict['predicted_chym_ec50']['loss'])
        all_mean_tryp_ec50_loss.append(predicted_dict['predicted_tryp_ec50']['loss'])

        loss_string = f'test_Step: {global_step} {n} loss: {mean_loss} binder_loss: {mean_binder_loss} \
lddt: {lddt_ca} fape: {fape} sc_fape: {sidechain_fape} chi: {chi_loss} disto: {distogram} msa: {masked_msa} \
chym_ec50_loss: {chym_ec50_loss} tryp_ec50_loss: {tryp_ec50_loss} weighted_binder_loss: {weighted_binder_loss} \
weighted_chym_ec50_loss: {weighted_chym_ec50_loss} weighted_tryp_ec50_loss: {weighted_tryp_ec50_loss} \
chym_ec50_pred: {chym_ec50_pred} tryp_ec50_pred: {tryp_ec50_pred} true_chym_ec50: {batch["chym_ec50"][0]} \
true_tryp_ec50: {batch["tryp_ec50"][0]} true_chym_ec50_class: {batch["chym_ec50_class"][0]} \
true_tryp_ec50_class: {batch["tryp_ec50_class"][0]} \
sample: {targetid} raw_fape: {raw_fape} raw_sc_fape: {raw_sidechain_fape}\n'
        prefix = FLAGS.outprefix
        loss_fname = f'{prefix}_validation_loss.txt'
        with open(loss_fname, 'a') as f:
            f.write(loss_string)

        sys.stdout.flush()

    return np.mean(all_mean_loss), np.mean(all_mean_binder_loss), np.mean(all_mean_fape), \
        np.mean(all_mean_chym_ec50_loss), np.mean(all_mean_tryp_ec50_loss)


def save_model(prefix, global_step, replicated_params, opt_state, model_config):
    """Save fine-tuned model params, state, config, and loss."""
    step_fname = f'{prefix}_global_step.npy'
    np.save(step_fname, global_step)

    param_fname = f'{prefix}_params_{global_step}.pkl'
    save_params = jax.tree_map(lambda x: x[0,], replicated_params)
    with open(param_fname, 'wb') as f:
        pickle.dump(save_params, f)

    state_fname = f'{prefix}_state_{global_step}.pkl'
    with open(state_fname, 'wb') as f:
        pickle.dump(opt_state, f)

    config_fname = f'{prefix}_config.pkl'
    with open(config_fname, 'wb') as f:
        pickle.dump(model_config, f)

    print(f'\nSaved model at {global_step} training steps.\n')


def load_finetuned_params_and_loss(FLAGS):
    """Load previous fine-tuned model params and loss."""
    with open(FLAGS.resume_params, 'rb') as f:
        model_params = pickle.load(f)

    prefix = FLAGS.outprefix
    train_loss_fname = f'{prefix}_training_loss.txt'
    new_file_string = ''
    with open(FLAGS.resume_train_loss, 'r') as f:
        for line in f.readlines():
            line_split = line.replace(',','').split()
            curr_step = int(line_split[1].replace(',',''))
            if curr_step > FLAGS.resume_train_step:
                continue
            else:
                new_file_string += line.replace(',','')
    with open(train_loss_fname, 'w') as f:
        f.write(new_file_string)

    valid_new_file_string = ''
    valid_loss_fname = f'{prefix}_validation_loss.txt'
    with open(FLAGS.resume_valid_loss, 'r') as f:
        for line in f.readlines():
            line_split = line.replace(',','').split()
            curr_step = int(line_split[1].replace(',',''))
            if curr_step > FLAGS.resume_train_step:
                continue
            else:
                valid_new_file_string += line.replace(',','')

    with open(valid_loss_fname, 'w') as f:
        f.write(valid_new_file_string)

    global_step = FLAGS.resume_train_step + 1

    curr_step, prev_step = 0, 0
    all_valid_loss, all_valid_binder_loss, all_valid_fape = [], [], []
    temp_valid_loss, temp_valid_binder_loss, temp_valid_fape = [], [], []
    with open(FLAGS.resume_valid_loss, 'r') as f:
        for line in f.readlines():
            if line.startswith('test_Step') == False:
                continue
            line_split = line.replace(',','').split()
            curr_step = int(line_split[1].replace(',',''))
            if curr_step > FLAGS.resume_train_step:
                continue
            elif prev_step == 0 or curr_step == prev_step:
                temp_valid_loss.append(float(line_split[4]))
                temp_valid_binder_loss.append(float(line_split[6]))
                temp_valid_fape.append(float(line_split[10]))
            else:
                all_valid_loss.append(np.mean(temp_valid_loss))
                all_valid_binder_loss.append(np.mean(temp_valid_binder_loss))
                all_valid_fape.append(np.mean(temp_valid_fape))
                temp_valid_loss, temp_valid_binder_loss, temp_valid_fape = [], [], []
            prev_step = curr_step

    print('\n\nLoaded Previous Validation Loss Values:')
    print(all_valid_loss)
    print('\n\n')

    return model_params, global_step, all_valid_loss, all_valid_binder_loss, all_valid_fape


def parse_arguments():
    """
    Parse command-line arguments for AF2 fine-tuning.
    """
    parser = argparse.ArgumentParser(
        description='Fine-tune AlphaFold2 on a custom dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--outprefix', type=str, default='testrun',
                        help='Prefix for output filenames')
    parser.add_argument('--model_name', type=str, default='model_2_ptm',
                        help='AlphaFold model name')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Location of AlphaFold params/ folder')

    parser.add_argument('--train_dataset', type=str, required=True,
                        help='tsv file listing the dataset training samples. '
                        'See phbradley/alphafold_finetune github README for format.')
    parser.add_argument('--valid_dataset', type=str, required=True,
                        help='tsv file listing the dataset validation samples')

    parser.add_argument('--plddt_binder', action='store_true', default=False,
                        help='Use pLDDT binder classifier (default: False)')
    parser.add_argument('--plddt_binder_slope', type=float, default=16.69,
                        help='Slope for plddt-based binder head')
    parser.add_argument('--pae_binder_slope', type=float, default=-7.90,
                        help='Slope for pae-based binder head')
    parser.add_argument('--binder_loss_weight', type=float, default=1.0,
                        help='Weight for binder classification loss')
    parser.add_argument('--binder_intercepts', type=float, nargs='+', default=[0.74],
                        help='Initial values for the switchpoint parameter in '
                             'the logistic regression binder layer. This is the point '
                             'where the prediction switches from binder to non-binder. '
                             'Should be in values of 0.01*pLDDT or 0.1*PAE, where '
                             'pLDDT ranges 0-100 and PAE has values in the 2-20 ish range.'
                             'See example command lines on the phbradley/alphafold_finetune '
                             'github README.')
    parser.add_argument('--freeze_binder', action='store_true',
                        help='Freeze binder model params')
    parser.add_argument('--only_fit_binder', action='store_true',
                        help='Freeze alphafold model params')
    parser.add_argument('--freeze_everything', action='store_true',
                        help='Dont apply the accumulated gradients')

    parser.add_argument('--crop_size', type=int, default=190,
                        help='Max size of training example; set this '
                             'as low as possible for memory and speed')
    parser.add_argument('--extra_msa', type=int, default=1,
                        help='Number of extra MSA sequences')
    parser.add_argument('--msa_clusters', type=int, default=5,
                        help='Number of MSA cluster sequences')
    parser.add_argument('--num_evo_blocks', type=int, default=48,
                        help='Number of evoformer blocks')
    parser.add_argument('--struc_viol_weight', type=float, default=0.0,
                        help='Structural violation weight')

    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (only batch_size=1 tested)')
    parser.add_argument('--num_cpus', type=int, default=1,
                        help='Number of CPUs for multiprocessing')
    parser.add_argument('--fake_native_weight', type=float, default=0,
                        help='Weight to apply to structure loss when using a predicted '
                        'structure as the ground truth')
    parser.add_argument('--random_recycling', action='store_true',
                        help='if True, set num_iter_recycling '
                        'randomly during training')

    parser.add_argument('--grad_norm_clip', type=float, default=10.0,
                        help='Clip gradient L2 norm to this value per update')
    parser.add_argument('--apply_every', type=int, default=1,
                        help='Accumulate gradients from this many batches before applying')
    parser.add_argument('--lr_coef', type=float, default=0.025,
                        help='Coefficient to multiply the learning rate')
    parser.add_argument('--no_ramp', action='store_true',
                        help='Turn off the learning rate ramp-up')

    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Number of global_steps between model checkpoints')
    parser.add_argument('--print_steps', type=int, default=20,
                        help='Number of global_steps between printing averaged results')
    parser.add_argument('--valid_steps', type=int, default=10000,
                        help='Number of global_steps between validation predictions')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of validation steps before early stopping. '
                        'Training will terminate if validation loss '
                        'does not decrease in <patience> validation epochs')

    parser.add_argument('--resume_training', action='store_true',
                        help='Resume fine-tuning of a fine-tuned model')
    parser.add_argument('--resume_params', type=str, default='',
                        help='Path to params.pkl file from which to continue model training')
    parser.add_argument('--resume_train_step', type=int, default=30000,
                        help='Global_step at which to resume fine-tuning. Use previous loss values '
                        'before this step for early stopping.')
    parser.add_argument('--resume_train_loss', type=str, default='',
                        help='txt file containing training loss of the previously fine-tuned model')
    parser.add_argument('--resume_valid_loss', type=str, default='',
                        help='txt file containing validation loss of the previously fine-tuned model')

    parser.add_argument('--test_load', action='store_true',
                        help='Loop through datasets to test loading')
    parser.add_argument('--notrain', action='store_true',
                        help='Dont do training (debugging)')
    parser.add_argument('--no_valid', action='store_true',
                        help='Skip validation steps')
    parser.add_argument('--dump_pdbs', action='store_true',
                        help='Dump pdb files of modeled structures during training')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose logging for debugging')
    parser.add_argument('--debug', action='store_true',
                        help='Debug logging for debugging')
    parser.add_argument('--no_random', action='store_true',
                        help='Set all random seeds to 0')

    parser.add_argument('--chymEC50_loss_weight', type=float, default=1.0,
                        help='Set ChymotrypsinEC50 Loss weight.')
    parser.add_argument('--trypEC50_loss_weight', type=float, default=1.0,
                        help='Set TrypsinEC50 Loss weight.')

    parser.add_argument('--train_last_n_layers', type=int, default=None,
                        help='Number of pEC50 head layers to unfreeze (counting from the end). '
                        'If not set or None, all pEC50 layers are trainable by default. '
                        'If set to 0, all pEC50 layers are frozen. '
                        'If set to a positive integer N, only the last N layers of each pEC50 head are trainable. '
                        'Each pEC50 head (chym and tryp) has 6 layers: input_layer_norm, act_0, act_1, logits, act_2, act_3. '
                        'Example: --train_last_n_layers=2 unfreezes only act_2 and act_3 (the final 2 layers). '
                        'This flag is overridden by --only_fit_pec50.')
    parser.add_argument('--only_fit_pec50', action='store_true',
                        help='If True, unfreeze all pEC50 head layers and freeze base AF2 model params. '
                        'This flag overrides --train_last_n_layers.')

    parser.add_argument('--unfreeze_af2_model', action='store_true',
                        help='If True, unfreeze the entire base AlphaFold2 model for training '
                        'in addition to the pEC50 head. This allows joint training of both '
                        'the base model and the pEC50 prediction heads.')
    parser.add_argument('--af2_params_to_train', type=str, default=None,
                        help='Path to a txt file containing a newline-separated list of '
                        'AlphaFold2 parameter names to selectively unfreeze for training. '
                        'Parameter names should match the keys in the Haiku parameter tree. '
                        'If provided, only the specified AF2 parameters will be trainable. '
                        'This flag is ignored if --unfreeze_af2_model is True.')

    return parser.parse_args()


def configure_model(args, model_name):
    """
    Configure AF model settings.
    """
    model_config = config.model_config(model_name)
    model_config.data.common.resample_msa_in_recycling = True
    model_config.model.resample_msa_in_recycling = True
    model_config.data.common.max_extra_msa = args.extra_msa
    model_config.data.eval.max_msa_clusters = args.msa_clusters
    model_config.data.eval.crop_size = args.crop_size
    model_config.model.heads.structure_module.structural_violation_loss_weight = args.struc_viol_weight
    model_config.model.embeddings_and_evoformer.evoformer_num_block = args.num_evo_blocks

    model_config.model.heads.predicted_chym_ec50.weight = args.chymEC50_loss_weight
    model_config.model.heads.predicted_tryp_ec50.weight = args.trypEC50_loss_weight

    return model_config


def initialize_binder_classifier(args):
    """
    Initialize binder classifier (PAE or pLDDT based).
    """
    if not args.plddt_binder:
        assert all(x < 2.0 for x in args.binder_intercepts), \
            'binder model PAE values are multiplied by 0.1'

        def binder_classification_fn(input_features):
            model = PaeBinderClassifier()(input_features)
            return model

        binder_classifier = hk.transform(binder_classification_fn, apply_rng=True)

        rng = jax.random.PRNGKey(42)
        classifier_dict = {}
        crop_size = 200
        num_bins = 64
        num_binder_classes = len(args.binder_intercepts)

        classifier_dict['partner1_mask'] = jnp.ones([crop_size], dtype=jnp.float32)
        classifier_dict['partner2_mask'] = jnp.ones([crop_size], dtype=jnp.float32)
        classifier_dict['predicted_aligned_error'] = {
            'logits': jnp.zeros([1, crop_size, crop_size, num_bins], dtype=jnp.float32),
            'breaks': jnp.zeros([1, num_bins-1], dtype=jnp.float32),
        }
        classifier_dict['binder_class_1hot'] = jnp.ones([1, num_binder_classes],
                                                        dtype=jnp.float32)

        binder_model_params = binder_classifier.init(rng, input_features=classifier_dict)
    else:
        assert all(x < 1.0 for x in args.binder_intercepts), \
            'binder model pLDDT values are multiplied by 0.01 (ie, run from 0-1)'

        def binder_classification_fn(input_features):
            model = PlddtBinderClassifier()(input_features)
            return model

        binder_classifier = hk.transform(binder_classification_fn, apply_rng=True)

        rng = jax.random.PRNGKey(42)
        classifier_dict = {}
        crop_size = 200
        num_bins = 50
        num_binder_classes = len(args.binder_intercepts)

        classifier_dict['peptide_mask'] = jnp.ones([1, crop_size], dtype=jnp.float32)
        classifier_dict['binder_class_1hot'] = jnp.ones([1, num_binder_classes],
                                                        dtype=jnp.float32)
        classifier_dict['predicted_lddt'] = {
            'logits': jnp.zeros([crop_size, num_bins], dtype=jnp.float32)
        }

        binder_model_params = binder_classifier.init(rng, input_features=classifier_dict)

    return binder_classifier, binder_model_params


def initialize_model_params(args, model_name, binder_model_params):
    """
    Initialize or resume model params and training state.
    """
    if args.resume_training:
        model_params, global_step, all_valid_loss, \
            all_valid_binder_loss, all_valid_fape = load_finetuned_params_and_loss(args)

        binder_model_params, af2_model_params = hk.data_structures.partition(
            lambda m, n, p: m[:9] != "alphafold", model_params)
    else:
        af2_model_params = data.get_model_haiku_params(
            model_name=model_name, data_dir=args.data_dir)

        print('initial binder params:', binder_model_params)

        model_params = hk.data_structures.merge(af2_model_params, binder_model_params)
        global_step = 0
        all_valid_loss, all_valid_binder_loss, all_valid_fape = [], [], []

    all_valid_chym_ec50_loss = []
    all_valid_tryp_ec50_loss = []

    return model_params, global_step, all_valid_loss, all_valid_binder_loss, all_valid_fape, \
        all_valid_chym_ec50_loss, all_valid_tryp_ec50_loss, af2_model_params


def load_datasets(args, batch_size):
    """
    Load training and validation datasets.
    """
    train_df = pd.read_table(args.train_dataset)
    valid_df = pd.read_table(args.valid_dataset)

    training_set = CustomPandasDataset(train_df, create_batch_from_dataset_row)
    valid_set = CustomPandasDataset(valid_df, create_batch_from_dataset_row)

    params_loader = {
        'num_workers': args.num_cpus,
        'pin_memory': False,
        'batch_size': batch_size,
        'collate_fn': collate,
    }

    class_counts = train_df['ec50_class'].value_counts()
    sample_weights = [1/class_counts[i] for i in train_df.ec50_class.values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_df), replacement=True)

    train_loader = torch.utils.data.DataLoader(training_set, sampler=sampler, **params_loader)
    valid_loader = torch.utils.data.DataLoader(valid_set, **params_loader)

    return train_loader, valid_loader


def setup_optimizer(args, model_params):
    """
    Setup optimizer and initialize optimizer state.
    """
    if args.no_ramp:
        scheduler = optax.linear_schedule(1e-3, 1e-3, 1000, 0)
    else:
        scheduler = optax.linear_schedule(0.0, 1e-3, 1000, 0)

    chain_me = [
        optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-6),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0 * args.lr_coef),
    ]

    gradient_transform = optax.chain(*chain_me)

    n_devices = jax.local_device_count()
    print('Number of devices:', n_devices, '\nSetting n_devices=1')
    n_devices = 1
    replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), model_params)

    binder_model_params, af2_model_params = hk.data_structures.partition(
        lambda m, n, p: m[:9] != "alphafold", replicated_params)
    base_af2_model_params, pEC50_head_params = hk.data_structures.partition(
        lambda m, n, p: "EC50" not in m, af2_model_params)

    if args.only_fit_pec50:
        print("\n--only_fit_pec50 is True, unfreezing all pEC50 head layers")
        print("      Base AF2 model remains frozen\n")
        trainable_pEC50_params = pEC50_head_params
        frozen_pEC50_params = {}
    elif args.train_last_n_layers is not None:
        print(f"\n--train_last_n_layers={args.train_last_n_layers}")
        print("      Unfreezing only the last N layers of pEC50 heads")
        print("      Base AF2 model remains frozen\n")
        frozen_pEC50_params, trainable_pEC50_params = split_pec50_params_by_layers(
            pEC50_head_params, args.train_last_n_layers)
    else:
        print("\nNo train_last_n_layers specified, training all pEC50 head layers")
        print("      Base AF2 model remains frozen\n")
        trainable_pEC50_params = pEC50_head_params
        frozen_pEC50_params = {}

    # Determine which base AF2 params to train based on flags
    if args.unfreeze_af2_model:
        print("\n--unfreeze_af2_model is True, unfreezing entire base AF2 model")
        print("      Training both AF2 base model and pEC50 heads\n")
        trainable_af2_params = base_af2_model_params
        frozen_af2_params = {}
    elif args.af2_params_to_train is not None:
        print(f"\n--af2_params_to_train specified: {args.af2_params_to_train}")
        print("      Selectively unfreezing specified AF2 parameters\n")
        
        # Load parameter names from file
        with open(args.af2_params_to_train, 'r') as f:
            params_to_train = set(line.strip() for line in f if line.strip())
        
        print(f"      Loaded {len(params_to_train)} parameter names to train:")
        for pname in list(params_to_train)[:5]:  # Print first 5 as examples
            print(f"        - {pname}")
        if len(params_to_train) > 5:
            print(f"        ... and {len(params_to_train) - 5} more")
        print()
        
        # Partition based on whether module name is in the list
        def is_trainable(module_name, param_name, param_value):
            return module_name in params_to_train
        
        frozen_af2_params, trainable_af2_params = hk.data_structures.partition(
            is_trainable, base_af2_model_params)
        
        # Count how many params are trainable
        trainable_count = len(jax.tree_util.tree_leaves(trainable_af2_params))
        frozen_count = len(jax.tree_util.tree_leaves(frozen_af2_params))
        print(f"      Result: {trainable_count} trainable AF2 params, {frozen_count} frozen AF2 params\n")
    else:
        print("\nBase AF2 model remains frozen (default behavior)\n")
        trainable_af2_params = {}
        frozen_af2_params = base_af2_model_params

    # Initialize optimizer states for trainable parameters
    if trainable_af2_params:
        # Training both pEC50 and AF2 params
        trainable_params = hk.data_structures.merge(trainable_pEC50_params, trainable_af2_params)
        opt_state = gradient_transform.init(trainable_params)
        print(f"Optimizer initialized for {len(jax.tree_util.tree_leaves(trainable_params))} trainable parameters")
        print(f"      (pEC50: {len(jax.tree_util.tree_leaves(trainable_pEC50_params))}, "
              f"AF2: {len(jax.tree_util.tree_leaves(trainable_af2_params))})\n")
    else:
        # Training only pEC50 params (default behavior)
        opt_state = gradient_transform.init(trainable_pEC50_params)
        print(f"Optimizer initialized for {len(jax.tree_util.tree_leaves(trainable_pEC50_params))} trainable pEC50 parameters\n")

    return gradient_transform, replicated_params, opt_state, base_af2_model_params, \
        trainable_pEC50_params, frozen_pEC50_params, binder_model_params, \
        trainable_af2_params, frozen_af2_params


def train_one_epoch(epoch, train_loader, replicated_params, base_af2_model_params,
                    trainable_pEC50_params, frozen_pEC50_params, binder_model_params,
                    trainable_af2_params, frozen_af2_params,
                    opt_state, gradient_transform, model_config, args, global_step,
                    all_valid_loss, all_valid_binder_loss, all_valid_fape,
                    all_valid_chym_ec50_loss, all_valid_tryp_ec50_loss,
                    valid_loader, jax_key, binder_loss_weight):
    """
    Train for one epoch.
    """
    temp_train_loss = []
    temp_binder_loss = []
    temp_lddt_ca = []
    temp_distogram = []
    temp_masked_msa = []
    temp_exper_res = []
    temp_pred_lddt = []
    temp_chi_loss = []
    temp_fape = []
    temp_sidechain_fape = []
    temp_chym_ec50_loss = []
    temp_tryp_ec50_loss = []
    temp_raw_fape = []
    temp_raw_sidechain_fape = []
    grads_sum, grads_sum_count = None, 0

    early_stop = False

    for n, batch in enumerate(train_loader):
        if args.notrain:
            break

        structure_flag = batch['native_exists'][0]
        targetid = batch['targetid'][0]

        print('train_epoch:', epoch, 'batch:', n)
        if 2:
            tmp_params = jax.tree_map(lambda x: x[0,], replicated_params)
            b_params, _ = hk.data_structures.partition(
                lambda m, x, p: m[:9] != "alphafold", tmp_params)
            print('binder_params:', b_params)

        binder_labels = batch['labels']
        prep_batch_for_step(batch, True)
        peptide_mask = batch['peptide_mask']

        if args.verbose:
            for key_, value_ in batch.items():
                print('before_train_step:', key_, type(value_),
                      getattr(value_, 'shape', 'unk_shape'),
                      getattr(value_, 'dtype', 'unk_dtype'))
        sys.stdout.flush()

        if args.no_random:
            subkey = jax.random.PRNGKey(0)
        else:
            jax_key, subkey = jax.random.split(jax_key)

        # Conditional training based on whether AF2 params are trainable
        if trainable_af2_params:
            # Training both AF2 and pEC50 params
            trainable_params = hk.data_structures.merge(trainable_pEC50_params, trainable_af2_params)
            frozen_params = hk.data_structures.merge(frozen_af2_params, frozen_pEC50_params)
            
            loss, grads, predicted_dict = jax.pmap(
                train_step_with_af2, in_axes=(0,0,None,0,None,None), axis_name='model_ax')(
                    trainable_params, frozen_params, subkey, batch, structure_flag, binder_loss_weight)
        else:
            # Training only pEC50 params (original behavior)
            base_af2_with_frozen = hk.data_structures.merge(base_af2_model_params, frozen_pEC50_params)
            
            loss, grads, predicted_dict = jax.pmap(
                train_step, in_axes=(0,0,None,0,None,None), axis_name='model_ax')(
                    trainable_pEC50_params, base_af2_with_frozen, subkey, batch, structure_flag, binder_loss_weight)

        binder_loss = [-1]
        binder_probs = -1
        binder_features = -1

        plddts = confidence.compute_plddt(predicted_dict['predicted_lddt']['logits'])
        print('train_epoch_n=', epoch, n, 'loss=', loss[0],
              'structure_flag:', structure_flag,
              'lddt_ca=', np.mean(predicted_dict['predicted_lddt']['lddt_ca']),
              'fape=', np.mean(predicted_dict['structure_module']['fape']),
              'binder_probs=', binder_probs,
              'binder_loss=', binder_loss,
              'peptide_plddt=', np.sum(peptide_mask*plddts)/9,
              'binder_features=', binder_features,
              'binder_labels=', binder_labels,
              'binder_params=', b_params,
              'chym_ec50_loss=', predicted_dict['predicted_chym_ec50']['loss'],
              'tryp_ec50_loss=', predicted_dict['predicted_tryp_ec50']['loss'],
        )

        if args.dump_pdbs:
            unrelaxed_protein = protein_from_prediction(batch, predicted_dict)
            outfile = f'{args.outprefix}train_result_{epoch:02d}_{n:04d}.pdb'
            with open(outfile, 'w') as f:
                f.write(protein.to_pdb(unrelaxed_protein))
                print('made:', outfile)

        temp_train_loss.append(np.mean(loss[0]))
        temp_binder_loss.append(np.mean(binder_loss[0]))
        temp_lddt_ca.append(np.mean(predicted_dict['predicted_lddt']['lddt_ca']))
        temp_distogram.append(np.mean(predicted_dict['distogram']['loss']))
        temp_masked_msa.append(np.mean(predicted_dict['masked_msa']['loss']))
        temp_pred_lddt.append(np.mean(predicted_dict['predicted_lddt']['loss']))
        temp_chi_loss.append(np.mean(predicted_dict['structure_module']['chi_loss']))

        ### Multiplying the fape by the native_exists (structure_flag), so
        ### that it is zero for unstable examples.
        temp_fape.append(np.mean(predicted_dict['structure_module']['fape']) * structure_flag)
        temp_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']) * structure_flag)
        ### Saving raw fape losses, too
        temp_raw_fape.append(np.mean(predicted_dict['structure_module']['fape']))
        temp_raw_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']))
        temp_chym_ec50_loss.append(predicted_dict['predicted_chym_ec50']['loss'])
        temp_tryp_ec50_loss.append(predicted_dict['predicted_tryp_ec50']['loss'])

        ## Making a critical change to these loss values. Saving these variables as the
        ## Loss for the current sample, and not the mean over the entire epoch. Doing this by changing
        ## mean_loss = round(float(np.mean(temp_train_loss)),4) to
        ## mean_loss = round(float(np.mean(temp_train_loss[-1])),4)
        mean_loss = round(float(np.mean(temp_train_loss[-1])), 4)
        mean_binder_loss = round(float(np.mean(temp_binder_loss[-1])), 4)
        lddt_ca = round(float(np.mean(temp_lddt_ca[-1])), 4)
        distogram = round(float(np.mean(temp_distogram[-1])), 4)
        masked_msa = round(float(np.mean(temp_masked_msa[-1])), 4)
        fape = round(float(np.mean(temp_fape[-1])), 4)
        sidechain_fape = round(float(np.mean(temp_sidechain_fape[-1])), 4)
        chi_loss = round(float(np.mean(temp_chi_loss[-1])), 4)
        raw_fape = round(float(np.mean(temp_raw_fape[-1])), 4)
        raw_sidechain_fape = round(float(np.mean(temp_raw_sidechain_fape[-1])), 4)

        ### EC50 loss values
        chym_ec50_loss = round(float(predicted_dict['predicted_chym_ec50']['loss']), 4)
        tryp_ec50_loss = round(float(predicted_dict['predicted_tryp_ec50']['loss']), 4)

        ### Predicted chym_ec50 and tryp_ec50 classes
        chym_ec50_pred = round(float(predicted_dict['predicted_chym_ec50']['pred_ec50_class']), 4)
        tryp_ec50_pred = round(float(predicted_dict['predicted_tryp_ec50']['pred_ec50_class']), 4)

        ### Weighted Protease EC50 losses
        weighted_chym_ec50_loss = round(float(chym_ec50_loss * args.chymEC50_loss_weight), 4)
        weighted_tryp_ec50_loss = round(float(tryp_ec50_loss * args.trypEC50_loss_weight), 4)
        weighted_binder_loss = 0

        loss_string = f'Step: {global_step} \
loss: {mean_loss} binder_loss: {mean_binder_loss} lddt: {lddt_ca} fape: {fape} sc_fape: {sidechain_fape} \
chi: {chi_loss} chym_ec50_loss: {chym_ec50_loss} tryp_ec50_loss: {tryp_ec50_loss} \
weighted_binder_loss: {weighted_binder_loss} weighted_chym_ec50_loss: {weighted_chym_ec50_loss} \
weighted_tryp_ec50_loss: {weighted_tryp_ec50_loss} chym_ec50_pred: {chym_ec50_pred} tryp_ec50_pred: {tryp_ec50_pred} \
true_chym_ec50: {batch["chym_ec50"][0]} true_tryp_ec50: {batch["tryp_ec50"][0]} true_chym_ec50_class: \
{batch["chym_ec50_class"][0]} true_tryp_ec50_class: {batch["tryp_ec50_class"][0]} \
sample: {targetid} raw_fape: {raw_fape} raw_sc_fape: {raw_sidechain_fape}\n'

        prefix = args.outprefix
        loss_fname = f'{prefix}_training_loss.txt'
        with open(loss_fname, 'a') as f:
            f.write(loss_string)

        global_step += 1

        print('grad accumulate:', global_step, grads_sum_count)
        if grads_sum_count == 0:
            grads_sum = grads
        else:
            grads_sum = jax.tree_map(lambda x, y: x+y, grads_sum, grads)
        grads_sum_count += 1

        if (grads_sum_count >= args.apply_every and not args.freeze_everything):
            print('grad update!', global_step, grads_sum_count)

            grads_sum = jax.tree_map(lambda x: x/grads_sum_count, grads_sum)
            grads_sum = norm_grads_per_example(grads_sum, l2_norm_clip=args.grad_norm_clip)

            if trainable_af2_params:
                # Training both pEC50 and AF2 - apply gradients to combined trainable params
                trainable_params = hk.data_structures.merge(trainable_pEC50_params, trainable_af2_params)
                updates, opt_state = gradient_transform.update(grads_sum, opt_state)
                trainable_params = optax.apply_updates(trainable_params, updates)
                
                # Split back into pEC50 and AF2
                trainable_pEC50_params, trainable_af2_params = hk.data_structures.partition(
                    lambda m, n, p: "EC50" in m, trainable_params)
                
                # Recombine all parameter groups
                frozen_params = hk.data_structures.merge(frozen_af2_params, frozen_pEC50_params)
                af2_model_params = hk.data_structures.merge(trainable_pEC50_params, trainable_af2_params, frozen_params)
                replicated_params = hk.data_structures.merge(binder_model_params, af2_model_params)
            else:
                # Original behavior - only update pEC50 params
                updates, opt_state = gradient_transform.update(grads_sum, opt_state)
                trainable_pEC50_params = optax.apply_updates(trainable_pEC50_params, updates)
                
                # Recombine all parameter groups
                pEC50_head_params_updated = hk.data_structures.merge(frozen_pEC50_params, trainable_pEC50_params)
                af2_model_params = hk.data_structures.merge(base_af2_model_params, pEC50_head_params_updated)
                replicated_params = hk.data_structures.merge(af2_model_params, binder_model_params)

            grads_sum, grads_sum_count = None, 0

        if global_step % args.print_steps == 0:
            mean_loss = round(float(np.mean(temp_train_loss[-1])), 4)
            mean_binder_loss = round(float(np.mean(temp_binder_loss[-1])), 4)
            lddt_ca = round(float(np.mean(temp_lddt_ca[-1])), 4)
            distogram = round(float(np.mean(temp_distogram[-1])), 4)
            masked_msa = round(float(np.mean(temp_masked_msa[-1])), 4)
            fape = round(float(np.mean(temp_fape[-1])), 4)
            sidechain_fape = round(float(np.mean(temp_sidechain_fape[-1])), 4)
            chi_loss = round(float(np.mean(temp_chi_loss[-1])), 4)
            raw_fape = round(float(np.mean(temp_raw_fape[-1])), 4)
            raw_sidechain_fape = round(float(np.mean(temp_raw_sidechain_fape[-1])), 4)

            chym_ec50_loss = round(float(predicted_dict['predicted_chym_ec50']['loss']), 4)
            tryp_ec50_loss = round(float(predicted_dict['predicted_tryp_ec50']['loss']), 4)
            chym_ec50_pred = round(float(predicted_dict['predicted_chym_ec50']['pred_ec50_class']), 4)
            tryp_ec50_pred = round(float(predicted_dict['predicted_tryp_ec50']['pred_ec50_class']), 4)
            weighted_chym_ec50_loss = round(float(chym_ec50_loss * args.chymEC50_loss_weight))
            weighted_tryp_ec50_loss = round(float(tryp_ec50_loss * args.trypEC50_loss_weight))
            weighted_binder_loss = 0

            print(f'Step: {global_step} \
loss: {mean_loss} binder_loss: {mean_binder_loss} lddt: {lddt_ca} fape: {fape} sc_fape: {sidechain_fape} \
chi: {chi_loss} disto: {distogram} msa: {masked_msa} chym_ec50_loss: {chym_ec50_loss} tryp_ec50_loss: {tryp_ec50_loss} \
weighted_binder_loss: {weighted_binder_loss} weighted_chym_ec50_loss: {weighted_chym_ec50_loss} \
weighted_tryp_ec50_loss: {weighted_tryp_ec50_loss} chym_ec50_pred: {chym_ec50_pred} tryp_ec50_pred: {tryp_ec50_pred} \
true_chym_ec50: {batch["chym_ec50"][0]} true_tryp_ec50: {batch["tryp_ec50"][0]} \
true_chym_ec50_class: {batch["chym_ec50_class"][0]} true_tryp_ec50_class: {batch["tryp_ec50_class"][0]} \
sample: {targetid} raw_fape: {raw_fape} raw_sc_fape: {raw_sidechain_fape}', flush=True)

            temp_train_loss = []
            temp_binder_loss = []
            temp_lddt_ca = []
            temp_distogram = []
            temp_masked_msa = []
            temp_exper_res = []
            temp_pred_lddt = []
            temp_chi_loss = []
            temp_fape = []
            temp_sidechain_fape = []
            temp_chym_ec50_loss = []
            temp_tryp_ec50_loss = []

        if global_step % args.save_steps == 0:
            save_model(prefix, global_step, replicated_params, opt_state, model_config)

        if global_step % args.valid_steps == 0:
            valid_loss, valid_binder_loss, valid_fape, valid_chym_ec50_loss, valid_tryp_ec50_loss = \
                compute_valid_stats(valid_loader, replicated_params, jax_key, binder_loss_weight)

            print('\n\nCurrent validation loss:')
            print(valid_loss)
            print('\n\nPrevious Validation Loss values:')
            print(all_valid_loss)
            print('\n\n')

            if all(valid_loss < val for val in all_valid_loss):
                prefix = args.outprefix + '_best_model'

                for f in os.listdir('.'):
                    if prefix in f:
                        os.remove(f)

                save_model(prefix, global_step, replicated_params, opt_state, model_config)

            all_valid_loss.append(valid_loss)
            all_valid_binder_loss.append(valid_binder_loss)
            all_valid_fape.append(valid_fape)
            all_valid_chym_ec50_loss.append(valid_chym_ec50_loss)
            all_valid_tryp_ec50_loss.append(valid_tryp_ec50_loss)

            if len(all_valid_loss) > args.patience+1 and \
               all(all_valid_loss[-1*args.patience-1] < val for val in all_valid_loss[-1*args.patience:]):
                prefix = args.outprefix
                save_model(prefix, global_step, replicated_params, opt_state, model_config)

                print('\n\nEarly Stopping; Model Training Stopped because Validation Loss has not decreased in %d Validation Steps\n\n' % args.patience)
                print('Previous Validation Steps:')
                print(all_valid_loss[-1*args.patience-1:])
                print('Minimum Previous Validation Loss:')
                print(all_valid_loss[-1*args.patience-1])
                early_stop = True
                break

        sys.stdout.flush()

    return replicated_params, trainable_pEC50_params, trainable_af2_params, opt_state, global_step, early_stop, jax_key, \
           all_valid_loss, all_valid_binder_loss, all_valid_fape, all_valid_chym_ec50_loss, all_valid_tryp_ec50_loss


def main():
    """Main execution for AF2 fine-tuning."""
    global FLAGS, model_runner, model_config, binder_classifier, batch_size, e, global_step

    args = parse_arguments()
    FLAGS = args

    print('FLAGS:', FLAGS)
    sys.stdout.flush()

    model_name = args.model_name
    batch_size = args.batch_size

    jax_key = jax.random.PRNGKey(0)

    print('start:', popen('hostname').read()[:-1])
    sys.stdout.flush()

    model_config = configure_model(args, model_name)

    binder_classifier, binder_model_params = initialize_binder_classifier(args)

    model_params, global_step, all_valid_loss, all_valid_binder_loss, all_valid_fape, \
        all_valid_chym_ec50_loss, all_valid_tryp_ec50_loss, af2_model_params = initialize_model_params(
            args, model_name, binder_model_params)

    model_runner = model.RunModel(model_config, af2_model_params)

    train_loader, valid_loader = load_datasets(args, batch_size)

    gradient_transform, replicated_params, opt_state, base_af2_model_params, \
        trainable_pEC50_params, frozen_pEC50_params, binder_model_params, \
        trainable_af2_params, frozen_af2_params = setup_optimizer(args, model_params)

    if args.test_load:
        for n, batch in enumerate(train_loader):
            print('trainload:', n)
            sys.stdout.flush()
        for n, batch in enumerate(valid_loader):
            print('validload:', n)
            sys.stdout.flush()

    if args.notrain:
        num_epochs = 1
    else:
        num_epochs = args.num_epochs

    binder_loss_weight = args.binder_loss_weight

    early_stop = False

    for e in range(num_epochs):
        replicated_params, trainable_pEC50_params, trainable_af2_params, opt_state, global_step, early_stop, jax_key, \
            all_valid_loss, all_valid_binder_loss, all_valid_fape, all_valid_chym_ec50_loss, all_valid_tryp_ec50_loss = train_one_epoch(
                e, train_loader, replicated_params, base_af2_model_params,
                trainable_pEC50_params, frozen_pEC50_params, binder_model_params,
                trainable_af2_params, frozen_af2_params,
                opt_state, gradient_transform, model_config, args, global_step,
                all_valid_loss, all_valid_binder_loss, all_valid_fape,
                all_valid_chym_ec50_loss, all_valid_tryp_ec50_loss,
                valid_loader, jax_key, binder_loss_weight
            )

        if early_stop:
            break

    valid_loss, valid_binder_loss, valid_fape, valid_chym_ec50_loss, valid_tryp_ec50_loss = compute_valid_stats(
        valid_loader, replicated_params, jax_key, binder_loss_weight)
    print('\nFinished final validation stats calculation.\n')


if __name__ == '__main__':
    main()