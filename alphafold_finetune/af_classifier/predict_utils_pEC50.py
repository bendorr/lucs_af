"""
predict_utils.py

Author: Ben Orr
Date: 7.10.23
Adapted from https://github.com/phbradley/alphafold_finetune

AF2 prediction utilities
"""

import sys
import os
from os.path import exists
import pickle
from collections import OrderedDict
from sys import exit
from typing import Dict, List, Tuple, Optional, Any, Union
from timeit import default_timer as timer
import random

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
import haiku as hk

import train_utils_ec50 as train_utils
from alphafold.common import residue_constants, protein
from alphafold.data import pipeline, templates
from alphafold.model import data, config, model


def mk_mock_template(
    query_sequence: Union[str, List[str]],
    num_temp: int = 1
) -> Dict[str, Any]:
    """
    Empty template features dict.

    From Sergey Ovchinnikov's colabfold/batch.py.
    Uses residue_constants vs templates.residue_constants.

    Parameters
    ----------
    query_sequence : str or list of str
        Query sequence(s)
    num_temp : int
        Number of templates, default 1

    Returns
    -------
    dict
        Template features with empty/mock values
    """
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )

    # Using '-' instead of 'A'
    output_templates_sequence = "-" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, residue_constants.atom_type_num))
    templates_aatype = residue_constants.sequence_to_onehot(
        output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID
    )

    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        )[0],
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        )[0],
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_confidence_scores": np.tile(
            output_confidence_scores[None], [num_temp, 1]
        ),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features


def load_pdb_coords(
    pdbfile: str,
    allow_chainbreaks: bool = False,
    allow_skipped_lines: bool = False,
    verbose: bool = False,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Dict[str, Dict[str, npt.NDArray]]], Dict[str, Dict[str, str]]]:
    """
    Load pdb coords. Simple, stripped-down reader - not robust for messy pdbs.

    Parameters
    ----------
    pdbfile : str
        pdb file path
    allow_chainbreaks : bool
        Allow chain breaks, default False
    allow_skipped_lines : bool
        Allow skipped lines, default False
    verbose : bool
        Verbose output, default False

    Returns
    -------
    tuple
        (chains, all_resids, all_coords, all_name1s)
        chains: list of chain IDs
        all_resids: dict[chain] = list of residue IDs
        all_coords: dict[chain][resid][atom] = xyz coords
        all_name1s: dict[chain][resid] = single letter AA
    """
    chains = []
    all_resids: Dict[str, List[str]] = {}
    all_coords: Dict[str, Dict[str, Dict[str, npt.NDArray]]] = {}
    all_name1s: Dict[str, Dict[str, str]] = {}

    if verbose:
        print('reading:', pdbfile)

    skipped_lines = False
    with open(pdbfile, 'r') as data:
        for line in data:
            if (line[:6] in ['ATOM  ', 'HETATM'] and
                line[17:20] != 'HOH' and
                line[16] in ' A1'):

                if (line[17:20] in residue_constants.restype_3to1 or
                    line[17:20] == 'MSE'):  # 2022-03-31 change to include MSE
                    name1 = ('M' if line[17:20] == 'MSE'
                             else residue_constants.restype_3to1[line[17:20]])
                    resid = line[22:27]
                    chain = line[21]

                    if chain not in all_resids:
                        all_resids[chain] = []
                        all_coords[chain] = {}
                        all_name1s[chain] = {}
                        chains.append(chain)

                    if line.startswith('HETATM'):
                        print('WARNING: HETATM', pdbfile, line[:-1])

                    atom = line[12:16].split()[0]
                    if resid not in all_resids[chain]:
                        all_resids[chain].append(resid)
                        all_coords[chain][resid] = {}
                        all_name1s[chain][resid] = name1

                    all_coords[chain][resid][atom] = np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    )
                else:
                    print('skip ATOM line:', line[:-1], pdbfile)
                    skipped_lines = True

    # Check for chainbreaks
    maxdis = 1.75
    for chain in chains:
        for res1, res2 in zip(all_resids[chain][:-1], all_resids[chain][1:]):
            coords1 = all_coords[chain][res1]
            coords2 = all_coords[chain][res2]
            if 'C' in coords1 and 'N' in coords2:
                dis = np.sqrt(np.sum(np.square(coords1['C'] - coords2['N'])))
                if dis > maxdis:
                    print('WARNING chainbreak:', chain, res1, res2, dis, pdbfile)
                    if not allow_chainbreaks:
                        print('STOP: chainbreaks', pdbfile)
                        print('DONE')
                        exit()

    if skipped_lines and not allow_skipped_lines:
        print('STOP: skipped lines:', pdbfile)
        print('DONE')
        exit()

    return chains, all_resids, all_coords, all_name1s


def fill_afold_coords(
    chain_order: List[str],
    all_resids: Dict[str, List[str]],
    all_coords: Dict[str, Dict[str, Dict[str, npt.NDArray]]],
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Fill AlphaFold atom37 coords.

    Parameters
    ----------
    chain_order : list of str
        Ordered chain IDs
    all_resids : dict
        dict[chain] = list of residue IDs
    all_coords : dict
        dict[chain][resid][atom] = xyz coords

    Returns
    -------
    tuple
        (all_positions, all_positions_mask)
        all_positions: [num_res, 37, 3] atom37 coords
        all_positions_mask: [num_res, 37] atom37 mask
    """
    assert residue_constants.atom_type_num == 37  # HACK/SANITY
    crs = [(chain, resid) for chain in chain_order for resid in all_resids[chain]]
    num_res = len(crs)

    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                   dtype=np.int64)

    for res_index, (chain, resid) in enumerate(crs):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)

        for atom_name, xyz in all_coords[chain][resid].items():
            x, y, z = xyz
            if atom_name in residue_constants.atom_order.keys():
                pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                mask[residue_constants.atom_order[atom_name]] = 1.0
            elif atom_name != 'NV':  # PRO NV OK to skip
                # Debugging/verbose output
                name = atom_name[:]
                while name[0] in '123':
                    name = name[1:]
                if name[0] != 'H':
                    print('unrecognized atom:', atom_name, chain, resid)

        all_positions[res_index] = pos
        all_positions_mask[res_index] = mask

    return all_positions, all_positions_mask


def run_alphafold_prediction(
    query_sequence: str,
    msa: List[str],
    deletion_matrix: List[List[int]],
    chainbreak_sequence: str,
    template_features: Dict[str, Any],
    model_runners: Dict[str, Any],
    out_prefix: str,
    crop_size: Optional[int] = None,
    dump_pdbs: bool = True,
    dump_metrics: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run AF prediction with template features.

    Parameters
    ----------
    query_sequence : str
        Query AA sequence
    msa : list of str
        MSA (single seq = list of one str)
    deletion_matrix : list of list of int
        Deletion matrix for MSA
    chainbreak_sequence : str
        Sequence with '/' chain breaks
    template_features : dict
        Template features dict
    model_runners : dict
        AF model runners
    out_prefix : str
        Output file prefix
    crop_size : int, optional
        Crop size
    dump_pdbs : bool
        Dump pdbs, default True
    dump_metrics : bool
        Dump metrics, default True

    Returns
    -------
    dict
        dict[model_name] = dict of metrics
    """
    # Gather features for running with only template information
    feature_dict = {
        **pipeline.make_sequence_features(
            sequence=query_sequence,
            description="none",
            num_res=len(query_sequence)
        ),
        **pipeline.make_msa_features(
            msas=[msa],
            deletion_matrices=[deletion_matrix]
        ),
        **template_features
    }

    # Add big enough number to residue index to indicate chain breaks
    # Ls: number of residues in each chain
    Ls = [len(split) for split in chainbreak_sequence.split('/')]
    idx_res = feature_dict['residue_index']
    L_prev = 0
    for L_i in Ls[:-1]:
        idx_res[L_prev + L_i:] += 200
        L_prev += L_i
    feature_dict['residue_index'] = idx_res

    all_metrics = predict_structure(
        out_prefix, feature_dict, model_runners,
        crop_size=crop_size,
        dump_pdbs=dump_pdbs,
        dump_metrics=dump_metrics,
    )

    return all_metrics


def predict_structure(
    prefix: str,
    feature_dict: Dict[str, Any],
    model_runners: Dict[str, Any],
    random_seed: int = 0,
    crop_size: Optional[int] = None,
    dump_pdbs: bool = True,
    dump_metrics: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Predict structure using AF.

    Parameters
    ----------
    prefix : str
        Output file prefix
    feature_dict : dict
        Feature dict for AF
    model_runners : dict
        AF model runners
    random_seed : int
        Random seed, default 0
    crop_size : int, optional
        Crop size
    dump_pdbs : bool
        Dump pdbs, default True
    dump_metrics : bool
        Dump metrics, default True

    Returns
    -------
    dict
        dict[model_name] = dict of metrics
    """
    unrelaxed_pdb_lines = []
    model_names = []

    metric_tags = 'plddt ptm predicted_aligned_error predicted_chym_ec50 predicted_tryp_ec50'.split()

    all_metrics: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, List[Any]] = {}

    # Run the models
    for model_name, model_runner in model_runners.items():
        start = timer()
        print(f"running {model_name}")

        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=random_seed
        )

        prediction_result = model_runner.predict(processed_feature_dict)

        unrelaxed_protein = protein.from_prediction(
            processed_feature_dict, prediction_result
        )
        unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
        model_names.append(model_name)

        all_metrics[model_name] = {}
        for tag in metric_tags:
            result = prediction_result.get(tag, None)
            metrics.setdefault(tag, []).append(result)

            if result is not None:
                
                ### Ben Orr 1.17.24. The PredictedChymEC50Head and PredictedTrypEC50Head
                ### return a dict containing logits, but loss is calculated on
                ### jnp.mean(jnp.array(value['predicted_chym_ec50']['logits'], dtype=jnp.float32))
                if tag in 'predicted_chym_ec50 predicted_tryp_ec50'.split():

                    ### Ben Orr 2.1.24: With the classifier pEC50 heads, the activations
                    ### stored after jax.nn.sigmoid are stored in the 'act' key.
                    ec50_result = np.mean(result['act'])
                    all_metrics[model_name][tag] = ec50_result

                else:
                    all_metrics[model_name][tag] = result

        print(f"{model_name} pLDDT: {np.mean(prediction_result['plddt'])} "
              f"Time: {timer() - start}")

    # Rerank models based on predicted LDDT
    plddts = metrics['plddt']
    lddt_rank = np.mean(plddts, -1).argsort()[::-1]

    for n, r in enumerate(lddt_rank):
        print(f"model_{n+1} {np.mean(plddts[r])}")

        if dump_pdbs:
            unrelaxed_pdb_path = f'{prefix}_model_{n+1}_{model_names[r]}.pdb'
            with open(unrelaxed_pdb_path, 'w') as f:
                f.write(unrelaxed_pdb_lines[r])

        if dump_metrics:
            metrics_prefix = f'{prefix}_model_{n+1}_{model_names[r]}'
            for tag in metric_tags:
                m = metrics[tag][r]
                if m is not None:
                    np.save(f'{metrics_prefix}_{tag}.npy', m)

    return all_metrics


def load_model_runners(
    model_names: List[str],
    crop_size: int,
    data_dir: str,
    num_recycle: int = 3,
    num_ensemble: int = 1,
    model_params_files: Optional[List[Optional[str]]] = None,
    resample_msa_in_recycling: bool = True,
    small_msas: bool = True,
) -> OrderedDict:
    """
    Load AF model runners.

    Parameters
    ----------
    model_names : list of str
        Model names
    crop_size : int
        Crop size
    data_dir : str
        AF params directory
    num_recycle : int
        Recycling iterations, default 3
    num_ensemble : int
        Ensemble iterations, default 1
    model_params_files : list of str or None
        Custom param files (None = default)
    resample_msa_in_recycling : bool
        Resample MSA, default True
    small_msas : bool
        Use small MSAs, default True

    Returns
    -------
    OrderedDict
        Model runners by name
    """
    if model_params_files is None:
        model_params_files = [None] * len(model_names)

    assert len(model_names) == len(model_params_files)

    model_runners: OrderedDict = OrderedDict()

    for model_name, model_params_file in zip(model_names, model_params_files):
        print('config:', model_name)
        af_model_name = (model_name[:model_name.index('_ft')]
                         if '_ft' in model_name else model_name)
        model_config = config.model_config(af_model_name)

        # Set crop size (not set automatically based on nres in this version)
        model_config.data.eval.crop_size = crop_size
        model_config.data.eval.num_ensemble = num_ensemble
        model_config.data.common.num_recycle = num_recycle
        model_config.model.num_recycle = num_recycle

        if small_msas:
            print('load_model_runners:: small_msas==True setting small '
                  'max_extra_msa and max_msa_clusters')
            model_config.data.common.max_extra_msa = 1
            model_config.data.eval.max_msa_clusters = 5

        if not resample_msa_in_recycling:
            model_config.data.common.resample_msa_in_recycling = False
            model_config.model.resample_msa_in_recycling = False

        if model_params_file != 'classic' and model_params_file is not None:
            print('loading', model_name, 'params from file:', model_params_file)
            with open(model_params_file, 'rb') as f:
                model_params = pickle.load(f)

            model_params, other_params = hk.data_structures.partition(
                lambda m, n, p: m[:9] == "alphafold", model_params
            )
            print('ignoring other_params:', other_params)
        else:
            assert '_ft' not in model_name
            model_params = data.get_model_haiku_params(
                model_name=model_name, data_dir=data_dir
            )

        model_runners[model_name] = model.RunModel(model_config, model_params)

    return model_runners


def create_single_template_features(
    target_sequence: str,
    template_pdbfile: str,
    target_to_template_alignment: Dict[int, int],
    template_name: str,
    allow_chainbreaks: bool = True,
    allow_skipped_lines: bool = True,
    expected_identities: Optional[int] = None,
    expected_template_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create template features from single template pdb.

    Parameters
    ----------
    target_sequence : str
        Target AA sequence
    template_pdbfile : str
        Template pdb file path
    target_to_template_alignment : dict
        Map target positions to template positions (0-indexed)
    template_name : str
        Template name (goes into template_domain_names, encoded)
    allow_chainbreaks : bool
        Allow chain breaks in template, default True
    allow_skipped_lines : bool
        Allow skipped lines when reading pdb, default True
    expected_identities : int, optional
        Expected identities for sanity check
    expected_template_len : int, optional
        Expected template length for sanity check

    Returns
    -------
    dict
        Template features dict
    """
    num_res = len(target_sequence)
    chains_tmp, all_resids_tmp, all_coords_tmp, all_name1s_tmp = load_pdb_coords(
        template_pdbfile,
        allow_chainbreaks=allow_chainbreaks,
        allow_skipped_lines=allow_skipped_lines,
    )

    crs_tmp = [(c, r) for c in chains_tmp for r in all_resids_tmp[c]]
    num_res_tmp = len(crs_tmp)
    template_full_sequence = ''.join(all_name1s_tmp[c][r] for c, r in crs_tmp)

    if expected_template_len:
        assert len(template_full_sequence) == expected_template_len

    all_positions_tmp, all_positions_mask_tmp = fill_afold_coords(
        chains_tmp, all_resids_tmp, all_coords_tmp
    )

    identities = sum(
        target_sequence[i] == template_full_sequence[j]
        for i, j in target_to_template_alignment.items()
    )
    if expected_identities:
        assert identities == expected_identities

    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                   dtype=np.int64)

    template_alseq = ['-'] * num_res
    for i, j in target_to_template_alignment.items():  # i=target, j=template
        template_alseq[i] = template_full_sequence[j]
        all_positions[i] = all_positions_tmp[j]
        all_positions_mask[i] = all_positions_mask_tmp[j]

    template_sequence = ''.join(template_alseq)
    assert len(template_sequence) == len(target_sequence)
    assert identities == sum(a == b for a, b in zip(template_sequence, target_sequence))

    template_aatype = residue_constants.sequence_to_onehot(
        template_sequence, residue_constants.HHBLITS_AA_TO_ID
    )

    template_features = {
        'template_all_atom_positions': all_positions,
        'template_all_atom_masks': all_positions_mask,
        'template_sequence': template_sequence.encode(),
        'template_aatype': template_aatype,
        'template_domain_names': template_name.encode(),
        'template_sum_probs': [identities],
    }

    return template_features


def compile_template_features(
    template_features_list: List[Dict[str, Any]]
) -> Dict[str, npt.NDArray]:
    """
    Compile multiple template features into single dict.

    Parameters
    ----------
    template_features_list : list of dict
        List of template feature dicts

    Returns
    -------
    dict
        Compiled template features with stacked arrays
    """
    all_template_features = {}
    for name, dtype in templates.TEMPLATE_FEATURES.items():
        all_template_features[name] = np.stack(
            [f[name] for f in template_features_list], axis=0
        ).astype(dtype)
    return all_template_features


def create_batch_for_training(
    target_chainseq: str,
    target_trim_positions: List[int],
    templates_alignfile: str,
    native_pdbfile: str,
    native_align: Dict[int, int],
    crop_size: int,
    model_runner: Any,
    native_identities: Optional[int] = None,
    native_len: Optional[int] = None,
    debug: bool = False,
    verbose: bool = False,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create training batch with native structure info.

    Parameters
    ----------
    target_chainseq : str
        Target sequence with '/' between chains
    target_trim_positions : list of int
        0-indexed positions to keep from full target sequence
    templates_alignfile : str
        tsv with template alignments (cols: template_pdbfile,
        target_to_template_alignstring, identities, target_len, template_len)
    native_pdbfile : str
        Native structure pdb path
    native_align : dict
        Map (target_pos, native_pos), 0-indexed wrt full target seq
    crop_size : int
        Crop size for model
    model_runner : RunModel
        AF model runner for feature processing
    native_identities : int, optional
        Expected native identities for sanity check
    native_len : int, optional
        Expected native length for sanity check
    debug : bool
        Debug mode, default False
    verbose : bool
        Verbose output, default False
    random_seed : int, optional
        Random seed (None = randomize), default None

    Returns
    -------
    dict
        Processed feature dict for training
    """
    assert len(target_trim_positions) <= crop_size
    assert None not in target_trim_positions

    if verbose:
        print('create_batch_for_training:', target_chainseq, target_trim_positions,
              templates_alignfile, native_pdbfile, native_align, crop_size,
              native_identities, native_len)

    target_trim_positions = sorted(set(target_trim_positions))  # sanity
    full_pos_to_trim_pos = {pos: i for i, pos in enumerate(target_trim_positions)}

    target_full_sequence = target_chainseq.replace('/', '')
    target_sequence = ''.join(target_full_sequence[x] for x in target_trim_positions)
    target_cs = target_chainseq.split('/')

    # Set up residue indices with chain breaks
    residue_index = np.arange(len(target_full_sequence))
    for ch in range(1, len(target_cs)):
        chain_begin = sum(len(x) for x in target_cs[:ch])
        residue_index[chain_begin:] += 200
    trim_residue_index = residue_index[target_trim_positions]

    # Load template features
    template_features_list = []
    templates_df = pd.read_table(templates_alignfile)

    for l in templates_df.itertuples():
        align_full = {
            int(x.split(':')[0]): int(x.split(':')[1])
            for x in l.target_to_template_alignstring.split(';')
        }
        align = {
            full_pos_to_trim_pos[x]: y for x, y in align_full.items()
            if x in target_trim_positions
        }

        # Return mock template features (empty template feature dicts)
        # as generated in Sergey Ovchinnikov's colabfold
        features = mk_mock_template(target_sequence)
        template_features_list.append(features)

    all_template_features = compile_template_features(template_features_list)

    # Create MSA features
    msa = [target_sequence]
    deletions = [[0] * len(target_sequence)]

    feature_dict = {
        **pipeline.make_sequence_features(
            sequence=target_sequence, description="none",
            num_res=len(target_sequence)
        ),
        **pipeline.make_msa_features(msas=[msa], deletion_matrices=[deletions]),
        **all_template_features,
    }

    if verbose:
        print('features_after_creation:', ' '.join(feature_dict.keys()))

    old_ri = feature_dict['residue_index']
    feature_dict['residue_index'] = trim_residue_index.astype(old_ri.dtype)

    if random_seed is None:
        random_seed = np.random.randint(0, 999999)

    with tf.device('cpu:0'):
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=random_seed
        )

    if verbose:
        print('features_after_initial_processing:',
              ' '.join(processed_feature_dict.keys()))

    # Get native coordinates
    if debug:
        native_features = create_single_template_features(
            target_full_sequence, native_pdbfile, native_align, 'dummy',
            expected_identities=native_identities,
            expected_template_len=native_len
        )

    native_align_trimmed = {
        full_pos_to_trim_pos[x]: y for x, y in native_align.items()
        if x in target_trim_positions
    }
    native_features = create_single_template_features(
        target_sequence, native_pdbfile, native_align_trimmed, 'dummy',
        expected_template_len=native_len
    )

    L = processed_feature_dict['aatype'].shape[1]
    assert L == crop_size
    L0 = len(target_trim_positions)

    atom_positions = np.concatenate([
        native_features['template_all_atom_positions'],
        np.zeros([L - L0, 37, 3])
    ], 0)
    atom_mask = np.concatenate([
        native_features['template_all_atom_masks'],
        np.zeros([L - L0, 37])
    ], 0)
    aatype = processed_feature_dict['aatype'][0]
    aatype = np.concatenate([
        processed_feature_dict['aatype'][0][:L0],
        20 * np.ones([L - L0])
    ], 0).astype(np.int32)

    # Calculate pseudo-beta positions
    pseudo_beta, pseudo_beta_mask = train_utils.pseudo_beta_fn_np(
        aatype, atom_positions, atom_mask
    )

    protein_dict = {
        'aatype': aatype,
        'all_atom_positions': atom_positions,
        'all_atom_mask': atom_mask
    }
    protein_dict = train_utils.make_atom14_positions(protein_dict)
    del protein_dict['aatype']

    for key_, value_ in protein_dict.items():
        protein_dict[key_] = np.array(value_)[None,]

    processed_feature_dict['pseudo_beta'] = np.array(pseudo_beta)[None,]
    processed_feature_dict['pseudo_beta_mask'] = np.array(pseudo_beta_mask)[None,]
    processed_feature_dict['all_atom_mask'] = np.array(atom_mask)[None,]
    processed_feature_dict['resolution'] = np.array(1.0)[None,]
    processed_feature_dict.update(protein_dict)

    # Calculate backbone transformations
    n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
    rot, trans = train_utils.make_transform_from_reference_np(
        n_xyz=processed_feature_dict['all_atom_positions'][0, :, n, :],
        ca_xyz=processed_feature_dict['all_atom_positions'][0, :, ca, :],
        c_xyz=processed_feature_dict['all_atom_positions'][0, :, c, :]
    )
    processed_feature_dict['backbone_translation'] = trans[None,]
    processed_feature_dict['backbone_rotation'] = rot[None,]

    # Backbone affine mask: whether the residue has C, CA, N
    # (code borrowed from modules.py:2013)
    processed_feature_dict['backbone_affine_mask'] = (
        processed_feature_dict['all_atom_mask'][0, :, n] *
        processed_feature_dict['all_atom_mask'][0, :, ca] *
        processed_feature_dict['all_atom_mask'][0, :, c]
    )[None,]

    if verbose:
        print('features_at_end:', ' '.join(processed_feature_dict.keys()))

    return processed_feature_dict
