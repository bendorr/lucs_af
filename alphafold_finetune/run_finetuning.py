"""
run_finetuning.py

Author: Ben Orr
Date: 10.7.24
Adapted from https://github.com/phbradley/alphafold_finetune

Fine-tune Alphafold2 on a custom dataset.
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

import warnings
warnings.filterwarnings("ignore")
hk.vmap.require_split_rng = False

import train_utils
import predict_utils

print('done importing') ; sys.stdout.flush()


### Binder stuff #######################################################################

class PlddtBinderClassifier(hk.Module):
    """
    Uses average pLDDT value over the peptide.

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
    Uses average PAE value between peptide and MHC.

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


############################ end of binder stuff ############################


def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    return jnp.asarray(loss), jax.nn.softmax(logits)


def get_loss_fn_only_fit_base_af2(af2_model_params,
        key, processed_feature_dict, structure_flag, binder_loss_weight):
    """
    Calculate loss for base AlphaFold2 parameters only.

    Separates the gradients of the af2_model_params and binder_model_params,
    so that the binder_model_params gradients are not backpropagated to the
    rest of the AF2 model.
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


def train_step(af2_model_params, key, batch, structure_flag, binder_loss_weight):
    """
    Execute one training step.

    Returns the gradients of the base AF2 params only.
    The gradients for the binder_model_params are not returned.
    """
    (loss, (predicted_dict)), grads = jax.value_and_grad(
        get_loss_fn_only_fit_base_af2, has_aux=True)(
            af2_model_params, key, batch, structure_flag, binder_loss_weight)

    grads = jax.lax.pmean(grads, axis_name='model_ax')
    loss = jax.lax.pmean(loss, axis_name='model_ax')
    return loss, grads, predicted_dict


def norm_grads_per_example(grads, l2_norm_clip=0.1):
    """Normalize gradients by clipping to L2 norm."""
    nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm(jnp.array([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads]))
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    grads = jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)
    return grads


def create_batch_from_dataset_row(row):
    """
    Create a batch from a pandas DataFrame row.

    Required fields:
      target_chainseq  ('/'-separated chain amino acid sequences)
      templates_alignfile
      native_pdbfile
      native_alignstring (';'-separated intpairs eg '0:0;1:1;3:2;4:3')
      binder  (0 or 1)
      peptide_positions (';'-separated intlist eg '201;202;203;204;205')
      native_exists (True or False or 0 or 1)

    Optional fields:
      binder_class (needed if len(FLAGS.binder_intercepts)>1 )
      target_trim_positions (used to subset the target for memory reasons)
      native_identities (for debugging PDB IO)
      native_len (for debugging PDB IO)
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

    # binder stuff here:
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

    # peptide_mask is used by PlddtBinderClassifier
    batch['peptide_mask'] = np.zeros((FLAGS.crop_size,))
    batch['peptide_mask'][peptide_positions] = 1.

    # partner1/partner2_mask is used by PaeBinderClassifier
    batch['partner1_mask'] = np.zeros((FLAGS.crop_size,))
    batch['partner1_mask'][nonpeptide_positions] = 1.

    batch['partner2_mask'] = np.zeros((FLAGS.crop_size,))
    batch['partner2_mask'][peptide_positions] = 1.

    batch['native_exists'] = bool(row.native_exists)
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
    """Collate function for DataLoader."""
    out_dict = {}
    for name in train_utils.list_a + train_utils.list_a_templates:
        values = [item[name][0,] for item in samples]
        out_dict[name] = np.stack(values, axis=0)
    for name in train_utils.list_b + train_utils.list_b_templates:
        values = [item[name] for item in samples]
        out_dict[name] = np.stack(values, axis=0)
    aatype_ = [item['aatype'][0,] for item in samples]
    out_dict['aatype_'] = np.stack(aatype_, axis=0)

    # binder stuff:
    out_dict['binder_class_1hot' ] = [item['binder_class_1hot' ] for item in samples]
    out_dict['peptide_mask' ] = [item['peptide_mask' ] for item in samples]
    out_dict['partner1_mask'] = [item['partner1_mask'] for item in samples]
    out_dict['partner2_mask'] = [item['partner2_mask'] for item in samples]
    out_dict['labels'] = [item['labels'] for item in samples]
    out_dict['native_exists'] = [item['native_exists'] for item in samples]
    out_dict['targetid'] = [item['targetid'] for item in samples]

    return out_dict


def prep_batch_for_step(batch, training):
    """Prepare batch for training/validation step."""
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

    # binder stuff:
    batch['binder_class_1hot'] = jnp.array(batch['binder_class_1hot'], dtype=jnp.float32)
    batch['peptide_mask' ] = jnp.array(batch['peptide_mask' ], dtype=jnp.float32)
    batch['partner1_mask'] = jnp.array(batch['partner1_mask'], dtype=jnp.float32)
    batch['partner2_mask'] = jnp.array(batch['partner2_mask'], dtype=jnp.float32)

    del batch['native_exists']
    del batch['targetid']

    if FLAGS.verbose:
        print('prep_batch_for_step_final_features:', ' '.join(batch.keys()))


def protein_from_prediction(batch, predicted_dict, b_factors=None):
    """Create Protein object from prediction."""
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

    Parameters
    ----------
    valid_loader : torch.utils.data.DataLoader
        Validation data loader.
    replicated_params : dict
        Replicated model parameters.
    jax_key : jax.random.PRNGKey
        JAX random key.
    binder_loss_weight : float
        Weight for binder loss.

    Returns
    -------
    tuple
        (mean_valid_loss, mean_valid_binder_loss, mean_valid_fape)
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
    temp_raw_fape = []
    temp_raw_sidechain_fape = []

    all_mean_loss, all_mean_binder_loss, all_mean_fape = [], [], []

    for n, batch in enumerate(valid_loader):
        print('test_epoch:', e, 'batch:', n)
        structure_flag = batch['native_exists'][0]
        targetid = batch['targetid'][0]
        binder_labels = batch['labels']
        peptide_mask = batch['peptide_mask']
        prep_batch_for_step(batch, True)

        if FLAGS.no_random:
            subkey = jax.random.PRNGKey(0)
        else:
            jax_key, subkey = jax.random.split(jax_key)

        binder_model_params, af2_model_params = hk.data_structures.partition(
            lambda m, n, p: m[:9] != "alphafold", replicated_params)

        loss, grads, predicted_dict = jax.pmap(
            train_step, in_axes=(0,None,0,None,None), axis_name='model_ax')(
                af2_model_params, subkey, batch, structure_flag, binder_loss_weight)

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
        temp_fape.append(np.mean(predicted_dict['structure_module']['fape']) * structure_flag)
        temp_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']) * structure_flag)
        temp_raw_fape.append(np.mean(predicted_dict['structure_module']['fape']))
        temp_raw_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']))

        mean_loss = round(float(np.mean(temp_train_loss[-1])),4)
        mean_binder_loss = round(float(np.mean(temp_binder_loss[-1])),4)
        lddt_ca = round(float(np.mean(temp_lddt_ca[-1])),4)
        distogram = round(float(np.mean(temp_distogram[-1])),4)
        masked_msa = round(float(np.mean(temp_masked_msa[-1])),4)
        fape = round(float(np.mean(temp_fape[-1])),4)
        sidechain_fape = round(float(np.mean(temp_sidechain_fape[-1])),4)
        chi_loss = round(float(np.mean(temp_chi_loss[-1])),4)
        weighted_binder_loss = 0

        print(f'test_Step: {global_step} {n} loss: {mean_loss} binder_loss: {mean_binder_loss} \
lddt: {lddt_ca} fape: {fape} sc_fape: {sidechain_fape} chi: {chi_loss} disto: {distogram} msa: {masked_msa} \
weighted_binder_loss: {weighted_binder_loss} \
sample: {targetid}', flush=True)

        all_mean_loss.append(mean_loss)
        all_mean_binder_loss.append(mean_binder_loss)
        all_mean_fape.append(fape)

        loss_string = f'test_Step: {global_step} {n} loss: {mean_loss} binder_loss: {mean_binder_loss} \
lddt: {lddt_ca} fape: {fape} sc_fape: {sidechain_fape} chi: {chi_loss} disto: {distogram} msa: {masked_msa} \
weighted_binder_loss: {weighted_binder_loss} \
sample: {targetid}\n'
        prefix = FLAGS.outprefix
        loss_fname = f'{prefix}_validation_loss.txt'
        with open(loss_fname, 'a') as f:
            f.write(loss_string)

        sys.stdout.flush()

    return np.mean(all_mean_loss), np.mean(all_mean_binder_loss), np.mean(all_mean_fape)


def save_model(prefix, global_step, replicated_params, opt_state, model_config):
    """Save the fine-tuned model's params, state, configuration, and loss values."""
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
    """Load a previous fine-tuned model's params and loss values."""
    with open(FLAGS.resume_params, 'rb') as f:
        model_params = pickle.load(f)

    # Copy the previous training and validation loss files
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

    # Set the current global step
    global_step = FLAGS.resume_train_step + 1

    # Load the previous validation loss values
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


######################################################################################
# Refactor 12.18.25 - Added main() function and organized code into logical sub-functions.
######################################################################################

def parse_arguments():
    """
    Parse command-line arguments for AlphaFold fine-tuning.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Fine-tune AlphaFold2 on a custom dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Output and model configuration
    parser.add_argument('--outprefix', type=str, default='testrun',
                        help='Prefix for output filenames')
    parser.add_argument('--model_name', type=str, default='model_2_ptm',
                        help='AlphaFold model name')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Location of AlphaFold params/ folder')

    # Dataset configuration
    parser.add_argument('--train_dataset', type=str, required=True,
                        help='File listing the dataset training samples')
    parser.add_argument('--valid_dataset', type=str, required=True,
                        help='File listing the dataset validation samples')

    # Binder classifier configuration
    parser.add_argument('--plddt_binder', action='store_true', default=True,
                        help='Use pLDDT binder classifier (default: True)')
    parser.add_argument('--plddt_binder_slope', type=float, default=40.0,
                        help='Slope for plddt-based binder head')
    parser.add_argument('--pae_binder_slope', type=float, default=-20.0,
                        help='Slope for pae-based binder head')
    parser.add_argument('--binder_loss_weight', type=float, default=1.0,
                        help='Weight for binder classification loss')
    parser.add_argument('--binder_intercepts', type=float, nargs='+', default=[0.74],
                        help='Binder model intercept values for different binder classes')
    parser.add_argument('--freeze_binder', action='store_true',
                        help='Freeze binder model params')
    parser.add_argument('--only_fit_binder', action='store_true',
                        help='Freeze alphafold model params')
    parser.add_argument('--freeze_everything', action='store_true',
                        help='Dont apply the accumulated gradients')

    # Model architecture configuration
    parser.add_argument('--crop_size', type=int, default=200,
                        help='Training crop size')
    parser.add_argument('--extra_msa', type=int, default=512,
                        help='Number of extra MSA sequences')
    parser.add_argument('--msa_clusters', type=int, default=256,
                        help='Number of MSA clusters')
    parser.add_argument('--num_evo_blocks', type=int, default=48,
                        help='Number of evoformer blocks')
    parser.add_argument('--struc_viol_weight', type=float, default=0.0,
                        help='Structural violation weight')

    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (NOTE: Only BS=1 tested)')
    parser.add_argument('--num_cpus', type=int, default=1,
                        help='Number of CPUs for multiprocessing')
    parser.add_argument('--fake_native_weight', type=float, default=0.25,
                        help='Weight for AF confidence terms when no real native structure')
    parser.add_argument('--random_recycling', action='store_true',
                        help='Use random recycling iterations')

    # Optimizer configuration
    parser.add_argument('--grad_norm_clip', type=float, default=0.1,
                        help='Clip gradient L2 norm to this value')
    parser.add_argument('--apply_every', type=int, default=1,
                        help='Accumulate gradients from this many batches before applying')
    parser.add_argument('--lr_coef', type=float, default=1.0,
                        help='Coefficient to multiply the learning rate')
    parser.add_argument('--no_ramp', action='store_true',
                        help='Turn off the learning rate ramp-up')

    # Checkpointing and logging
    parser.add_argument('--save_steps', type=int, default=5000,
                        help='Number of global_steps between model checkpoints')
    parser.add_argument('--print_steps', type=int, default=20,
                        help='Number of global_steps between printouts')
    parser.add_argument('--valid_steps', type=int, default=1000,
                        help='Number of global_steps between validation predictions')
    parser.add_argument('--patience', type=int, default=2,
                        help='Number of validation steps before early stopping')

    # Resume training
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume fine-tuning of an already fine-tuned model')
    parser.add_argument('--resume_params', type=str, default='',
                        help='Path to the params.pkl file of a previously fine-tuned model')
    parser.add_argument('--resume_train_step', type=int, default=0,
                        help='Global_step at which to resume fine-tuning')
    parser.add_argument('--resume_train_loss', type=str, default='',
                        help='Path to the training loss file of the previously fine-tuned model')
    parser.add_argument('--resume_valid_loss', type=str, default='',
                        help='Path to the validation loss file of the previously fine-tuned model')

    # Debugging and testing
    parser.add_argument('--test_load', action='store_true',
                        help='Test the dataset loading')
    parser.add_argument('--notrain', action='store_true',
                        help='Dont do training (debugging)')
    parser.add_argument('--no_valid', action='store_true',
                        help='Skip the validation pass')
    parser.add_argument('--dump_pdbs', action='store_true',
                        help='Dump PDB files during training')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose logging for debugging')
    parser.add_argument('--debug', action='store_true',
                        help='Debug logging for debugging')
    parser.add_argument('--no_random', action='store_true',
                        help='Set all random seeds to 0')

    return parser.parse_args()


def configure_model(args, model_name):
    """
    Configure the AlphaFold model settings.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    model_name : str
        Name of the AlphaFold model to use.

    Returns
    -------
    model_config
        Configured AlphaFold model configuration.
    """
    model_config = config.model_config(model_name)
    model_config.data.common.resample_msa_in_recycling = True
    model_config.model.resample_msa_in_recycling = True
    model_config.data.common.max_extra_msa = args.extra_msa
    model_config.data.eval.max_msa_clusters = args.msa_clusters
    model_config.data.eval.crop_size = args.crop_size
    model_config.model.heads.structure_module.structural_violation_loss_weight = args.struc_viol_weight
    model_config.model.embeddings_and_evoformer.evoformer_num_block = args.num_evo_blocks

    return model_config


def initialize_binder_classifier(args):
    """
    Initialize the binder classifier model (PAE or pLDDT based).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    tuple
        (binder_classifier, binder_model_params) - The classifier function and its initial parameters.
    """
    if not args.plddt_binder:
        # PAE-based binder classifier
        assert all(x < 2.0 for x in args.binder_intercepts), \
            'binder model PAE values are multiplied by 0.1'

        def binder_classification_fn(input_features):
            model = PaeBinderClassifier()(input_features)
            return model

        binder_classifier = hk.transform(binder_classification_fn, apply_rng=True)

        # Initialize PAE model parameters
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
        # pLDDT-based binder classifier
        assert all(x < 1.0 for x in args.binder_intercepts), \
            'binder model pLDDT values are multiplied by 0.01 (ie, run from 0-1)'

        def binder_classification_fn(input_features):
            model = PlddtBinderClassifier()(input_features)
            return model

        binder_classifier = hk.transform(binder_classification_fn, apply_rng=True)

        # Initialize pLDDT model parameters
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
    Initialize or resume model parameters and training state.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    model_name : str
        Name of the AlphaFold model.
    binder_model_params : dict
        Initial binder classifier parameters.

    Returns
    -------
    tuple
        (model_params, global_step, all_valid_loss, all_valid_binder_loss, all_valid_fape,
         af2_model_params) - Model parameters and training state.
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

    return model_params, global_step, all_valid_loss, all_valid_binder_loss, all_valid_fape, af2_model_params


def load_datasets(args, batch_size):
    """
    Load training and validation datasets.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    batch_size : int
        Batch size for dataloaders.

    Returns
    -------
    tuple
        (train_loader, valid_loader) - PyTorch data loaders for training and validation.
    """
    train_df = pd.read_table(args.train_dataset)
    valid_df = pd.read_table(args.valid_dataset)

    training_set = CustomPandasDataset(train_df, create_batch_from_dataset_row)
    valid_set = CustomPandasDataset(valid_df, create_batch_from_dataset_row)

    params_loader = {
        'shuffle': True,
        'num_workers': args.num_cpus,
        'pin_memory': False,
        'batch_size': batch_size,
        'collate_fn': collate,
    }

    train_loader = torch.utils.data.DataLoader(training_set, **params_loader)
    valid_loader = torch.utils.data.DataLoader(valid_set, **params_loader)

    return train_loader, valid_loader


def setup_optimizer(args, model_params):
    """
    Setup the optimizer and initialize optimizer state.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    model_params : dict
        Model parameters.

    Returns
    -------
    tuple
        (gradient_transform, replicated_params, af2_opt_state) - Optimizer components.
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
    af2_opt_state = gradient_transform.init(af2_model_params)

    return gradient_transform, replicated_params, af2_opt_state


def train_one_epoch(epoch, train_loader, replicated_params, af2_opt_state, gradient_transform,
                    model_config, args, global_step, all_valid_loss, all_valid_binder_loss,
                    all_valid_fape, valid_loader, jax_key, binder_loss_weight):
    """
    Train for one epoch.

    Parameters
    ----------
    epoch : int
        Current epoch number.
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    replicated_params : dict
        Replicated model parameters.
    af2_opt_state : optax state
        Optimizer state for AlphaFold2 parameters.
    gradient_transform : optax.GradientTransformation
        Gradient transformation chain.
    model_config : model config
        AlphaFold model configuration.
    args : argparse.Namespace
        Command-line arguments.
    global_step : int
        Current global training step.
    all_valid_loss : list
        History of validation losses.
    all_valid_binder_loss : list
        History of validation binder losses.
    all_valid_fape : list
        History of validation FAPE values.
    valid_loader : torch.utils.data.DataLoader
        Validation data loader.
    jax_key : jax.random.PRNGKey
        JAX random key.
    binder_loss_weight : float
        Weight for binder loss.

    Returns
    -------
    tuple
        (replicated_params, af2_opt_state, global_step, early_stop, jax_key,
         all_valid_loss, all_valid_binder_loss, all_valid_fape) - Updated training state.
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
    grads_sum, grads_sum_count = None, 0

    temp_raw_fape = []
    temp_raw_sidechain_fape = []

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

        binder_model_params, af2_model_params = hk.data_structures.partition(
            lambda m, n, p: m[:9] != "alphafold", replicated_params)

        loss, grads, predicted_dict = jax.pmap(
            train_step, in_axes=(0, None, 0, None, None), axis_name='model_ax')(
                af2_model_params, subkey, batch, structure_flag, binder_loss_weight)

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

        temp_fape.append(np.mean(predicted_dict['structure_module']['fape']) * structure_flag)
        temp_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']) * structure_flag)
        temp_raw_fape.append(np.mean(predicted_dict['structure_module']['fape']))
        temp_raw_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']))

        mean_loss = round(float(np.mean(temp_train_loss[-1])), 4)
        mean_binder_loss = round(float(np.mean(temp_binder_loss[-1])), 4)
        lddt_ca = round(float(np.mean(temp_lddt_ca[-1])), 4)
        distogram = round(float(np.mean(temp_distogram[-1])), 4)
        masked_msa = round(float(np.mean(temp_masked_msa[-1])), 4)
        fape = round(float(np.mean(temp_fape[-1])), 4)
        sidechain_fape = round(float(np.mean(temp_sidechain_fape[-1])), 4)
        chi_loss = round(float(np.mean(temp_chi_loss[-1])), 4)
        weighted_binder_loss = 0

        loss_string = f'Step: {global_step} loss: {mean_loss} binder_loss: {binder_loss} \
lddt: {lddt_ca} fape: {fape} sc_fape: {sidechain_fape} chi: {chi_loss} weighted_binder_loss: {weighted_binder_loss} \
sample: {targetid}\n'

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

            updates, af2_opt_state = gradient_transform.update(grads_sum, af2_opt_state)

            binder_model_params, af2_model_params = hk.data_structures.partition(
                lambda m, n, p: m[:9] != "alphafold", replicated_params)

            af2_model_params = optax.apply_updates(af2_model_params, updates)
            replicated_params = hk.data_structures.merge(af2_model_params, binder_model_params)

            grads_sum, grads_sum_count = None, 0

        if global_step % args.print_steps == 0:
            mean_loss = round(float(np.mean(temp_train_loss)), 4)
            binder_loss = round(float(np.mean(temp_binder_loss)), 4)
            lddt_ca = round(float(np.mean(temp_lddt_ca)), 4)
            distogram = round(float(np.mean(temp_distogram)), 4)
            masked_msa = round(float(np.mean(temp_masked_msa)), 4)
            fape = round(float(np.mean(temp_fape)), 4)
            sidechain_fape = round(float(np.mean(temp_sidechain_fape)), 4)
            chi_loss = round(float(np.mean(temp_chi_loss)), 4)
            weighted_binder_loss = 0

            print(f'Step: {global_step} loss: {mean_loss} binder_loss: {binder_loss} \
lddt: {lddt_ca} fape: {fape} sc_fape: {sidechain_fape} chi: {chi_loss} disto: {distogram} \
msa: {masked_msa} weighted_binder_loss: {weighted_binder_loss} \
sample: {targetid}', flush=True)

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

        if global_step % args.save_steps == 0:
            save_model(prefix, global_step, replicated_params, af2_opt_state, model_config)

        if global_step % args.valid_steps == 0:
            valid_loss, valid_binder_loss, valid_fape = \
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

                save_model(prefix, global_step, replicated_params, af2_opt_state, model_config)

            all_valid_loss.append(valid_loss)
            all_valid_binder_loss.append(valid_binder_loss)
            all_valid_fape.append(valid_fape)

            if len(all_valid_loss) > args.patience+1 and \
               all(all_valid_loss[-1*args.patience-1] < val for val in all_valid_loss[-1*args.patience:]):
                prefix = args.outprefix
                save_model(prefix, global_step, replicated_params, af2_opt_state, model_config)

                print('\n\nEarly Stopping; Model Training Stopped because Validation Loss has not decreased in %d Validation Steps\n\n' % args.patience)
                print('Previous Validation Steps:')
                print(all_valid_loss[-1*args.patience-1:])
                print('Minimum Previous Validation Loss:')
                print(all_valid_loss[-1*args.patience-1])
                early_stop = True
                break

        sys.stdout.flush()

    return replicated_params, af2_opt_state, global_step, early_stop, jax_key, \
           all_valid_loss, all_valid_binder_loss, all_valid_fape


def main():
    """Main execution function for AlphaFold fine-tuning."""
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
        af2_model_params = initialize_model_params(args, model_name, binder_model_params)

    model_runner = model.RunModel(model_config, af2_model_params)

    train_loader, valid_loader = load_datasets(args, batch_size)

    gradient_transform, replicated_params, af2_opt_state = setup_optimizer(args, model_params)

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
        replicated_params, af2_opt_state, global_step, early_stop, jax_key, \
            all_valid_loss, all_valid_binder_loss, all_valid_fape = train_one_epoch(
                e, train_loader, replicated_params, af2_opt_state, gradient_transform,
                model_config, args, global_step, all_valid_loss, all_valid_binder_loss,
                all_valid_fape, valid_loader, jax_key, binder_loss_weight
            )

        if early_stop:
            break

    # Compute validation stats one last time
    valid_loss, valid_binder_loss, valid_fape = compute_valid_stats(
        valid_loader, replicated_params, jax_key, binder_loss_weight)
    print('\nFinished final validation stats calculation.\n')


if __name__ == '__main__':
    main()
