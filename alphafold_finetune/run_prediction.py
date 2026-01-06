"""
run_prediction.py

Author: Ben Orr
Date: 10.7.24
Adapted from https://github.com/phbradley/alphafold_finetune

Run template-based AlphaFold inference on target sequences.
"""

import argparse
import os
import sys
from os.path import exists
import itertools
import numpy as np
import pandas as pd

import predict_utils
from alphafold.common import residue_constants


def mk_mock_template(query_sequence, num_temp=1):
    """
    Create empty template features dict.

    From Sergey Ovchinnikov's colabfold/batch.py.
    Edited to use residue_constants instead of templates.residue_constants.

    Parameters
    ----------
    query_sequence : str or list
        Query sequence(s).
    num_temp : int, optional
        Number of templates (default: 1).

    Returns
    -------
    dict
        Template features dictionary with empty/mock values.
    """
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
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
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_confidence_scores": np.tile(
            output_confidence_scores[None], [num_temp, 1]
        ),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_release_date": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features


def remove_completed_targets(targets, completed_pred_dir, prefix):
    """
    Remove completed predictions from the targets dataframe.

    Parameters
    ----------
    targets : pd.DataFrame
        Targets dataframe.
    completed_pred_dir : str
        Directory containing completed prediction PDB files.
    prefix : str
        Output file prefix.

    Returns
    -------
    pd.DataFrame
        Filtered targets dataframe without completed predictions.
    """
    completed_ids = []
    for f in os.listdir(completed_pred_dir):
        if f.endswith('.pdb'):
            header = prefix.split('/')[-1]
            targetid = f.replace(header+'_', '').split('_model')[0]
            completed_ids.append(targetid)

    targets = targets[~targets['targetid'].isin(completed_ids)]
    targets = targets.reset_index(drop=True)

    return targets


def parse_arguments():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run simple template-based alphafold inference",
        epilog='''
Examples:

python3 run_prediction.py \\
    --targets examples/targets.tsv \\
    --data_dir alphafold/ \\
    --outfile_prefix test1 \\
    --model_names model_2_ptm \\
    --ignore_identities

        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument('--targets', required=True,
                        help='File listing the targets to be modeled')

    # Output configuration
    parser.add_argument('--outfile_prefix',
                        help='Prefix prepended to output filenames')
    parser.add_argument('--final_outfile_prefix',
                        help='Prefix prepended to final output TSV filename')

    # Model configuration
    parser.add_argument('--data_dir',
                        help='Location of AlphaFold params/ folder')
    parser.add_argument('--model_names', type=str, nargs='*', default=['model_2_ptm'],
                        help='AlphaFold model names to use')
    parser.add_argument('--model_params_files', type=str, nargs='*',
                        help='Custom model parameter files')
    parser.add_argument('--num_recycle', type=int, default=3,
                        help='Number of recycling steps (default: 3)')

    # Prediction options
    parser.add_argument('--ignore_identities', action='store_true',
                        help='Ignore sequence identities in template alignment files')
    parser.add_argument('--no_resample_msa', action='store_true',
                        help='Dont randomly resample from MSA during recycling')
    parser.add_argument('--completed_pred_dir', type=str, default=None,
                        help='Directory with completed predictions to skip')

    # Output options
    parser.add_argument('--no_pdbs', action='store_true',
                        help='Dont write out PDB files')
    parser.add_argument('--terse', action='store_true',
                        help='Dont write out PDBs or confidence matrices')

    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def load_targets(args):
    """
    Load and filter target sequences.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    pd.DataFrame
        Targets dataframe.
    """
    targets = pd.read_table(args.targets)

    if args.completed_pred_dir is not None:
        targets = remove_completed_targets(
            targets, args.completed_pred_dir, args.outfile_prefix
        )

    return targets


def calculate_crop_size(targets):
    """
    Calculate the maximum crop size needed for all targets.

    Parameters
    ----------
    targets : pd.DataFrame
        Targets dataframe.

    Returns
    -------
    int
        Maximum crop size.
    """
    lens = [len(x.target_chainseq.replace('/',''))
            for x in targets.itertuples()]
    crop_size = max(lens)

    return crop_size


def print_verbose_info(args, targets, crop_size):
    """
    Print verbose logging information.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    targets : pd.DataFrame
        Targets dataframe.
    crop_size : int
        Crop size.
    """
    if args.verbose:
        import jax
        from os import popen

        platform = jax.local_devices()[0].platform
        hostname = popen('hostname').readlines()[0].strip()

        print('cmd:', ' '.join(sys.argv))
        print('local_device:', platform, 'hostname:', hostname,
              'num_targets:', targets.shape[0], 'max_len=', crop_size)

    sys.stdout.flush()


def load_model_runners(args, crop_size):
    """
    Load AlphaFold model runners.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    crop_size : int
        Crop size for models.

    Returns
    -------
    dict
        Dictionary of model runners.
    """
    model_runners = predict_utils.load_model_runners(
        args.model_names,
        crop_size,
        args.data_dir,
        num_recycle=args.num_recycle,
        model_params_files=args.model_params_files,
        resample_msa_in_recycling=not args.no_resample_msa,
    )

    return model_runners


def load_template_features(alignfile, query_sequence, args):
    """
    Load template features from alignment file.

    Parameters
    ----------
    alignfile : str
        Path to template alignment file.
    query_sequence : str
        Query sequence.
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    dict
        Compiled template features.
    """
    data = pd.read_table(alignfile)
    template_features_list = []

    for tnum, row in data.iterrows():
        target_to_template_alignment = {
            int(x.split(':')[0]): int(x.split(':')[1])
            for x in row.target_to_template_alignstring.split(';')
        }

        template_name = f'T{tnum:03d}'
        template_features = mk_mock_template(query_sequence)
        template_features_list.append(template_features)

    all_template_features = predict_utils.compile_template_features(
        template_features_list)

    print('All Template Features:')
    print(all_template_features)

    return all_template_features


def run_prediction_for_target(target, model_runners, args, crop_size):
    """
    Run AlphaFold prediction for a single target.

    Parameters
    ----------
    target : pd.Series
        Target row from dataframe.
    model_runners : dict
        Dictionary of model runners.
    args : argparse.Namespace
        Command-line arguments.
    crop_size : int
        Crop size.

    Returns
    -------
    dict
        Prediction metrics for all models.
    """
    query_chainseq = target.target_chainseq
    query_sequence = query_chainseq.replace('/','')

    alignfile = target.templates_alignfile
    assert exists(alignfile), f"Alignment file not found: {alignfile}"

    all_template_features = load_template_features(alignfile, query_sequence, args)

    msa = [query_sequence]
    deletion_matrix = [[0] * len(query_sequence)]

    all_metrics = predict_utils.run_alphafold_prediction(
        query_sequence=query_sequence,
        msa=msa,
        deletion_matrix=deletion_matrix,
        chainbreak_sequence=query_chainseq,
        template_features=all_template_features,
        model_runners=model_runners,
        out_prefix=None,  # Will be set in process_single_target
        crop_size=crop_size,
        dump_pdbs=not (args.no_pdbs or args.terse),
        dump_metrics=not args.terse,
    )

    return all_metrics


def calculate_confidence_metrics(target, all_metrics, query_chainseq):
    """
    Calculate pLDDT and PAE metrics from prediction results.

    Parameters
    ----------
    target : pd.Series
        Target row from dataframe.
    all_metrics : dict
        Prediction metrics from AlphaFold.
    query_chainseq : str
        Query chain sequence (with '/' separators).

    Returns
    -------
    pd.Series
        Target row with added confidence metrics.
    """
    outl = target.copy()
    query_sequence = query_chainseq.replace('/','')
    num_res = len(query_sequence)

    # Parse chain information
    cs = query_chainseq.split('/')
    chain_stops = list(itertools.accumulate(len(x) for x in cs))
    chain_starts = [0] + chain_stops[:-1]
    nres = chain_stops[-1]
    assert nres == num_res

    # Calculate metrics for each model
    for model_name, metrics in all_metrics.items():
        plddts = metrics['plddt']
        paes = metrics.get('predicted_aligned_error', None)

        # Overall metrics
        outl[model_name+'_plddt'] = np.mean(plddts[:nres])
        if paes is not None:
            outl[model_name+'_pae'] = np.mean(paes[:nres,:nres])

        # Per-chain metrics
        for chain1, (start1, stop1) in enumerate(zip(chain_starts, chain_stops)):
            outl[f'{model_name}_plddt_{chain1}'] = np.mean(plddts[start1:stop1])

            if paes is not None:
                for chain2 in range(len(cs)):
                    start2, stop2 = chain_starts[chain2], chain_stops[chain2]
                    pae = np.mean(paes[start1:stop1, start2:stop2])
                    outl[f'{model_name}_pae_{chain1}_{chain2}'] = pae

    return outl


def process_single_target(target, counter, total_targets, model_runners, args, crop_size):
    """
    Process a single target sequence.

    Parameters
    ----------
    target : pd.Series
        Target row from dataframe.
    counter : int
        Current target index.
    total_targets : int
        Total number of targets.
    model_runners : dict
        Dictionary of model runners.
    args : argparse.Namespace
        Command-line arguments.
    crop_size : int
        Crop size.

    Returns
    -------
    pd.Series
        Target with calculated metrics.
    """
    print('START:', counter, 'of', total_targets)

    # Determine output prefix
    if 'outfile_prefix' in target:
        outfile_prefix = target.outfile_prefix
    else:
        assert args.outfile_prefix is not None, \
            "Must provide --outfile_prefix or outfile_prefix column"
        if 'targetid' in target:
            outfile_prefix = args.outfile_prefix + '_' + str(target.targetid)
        else:
            outfile_prefix = f'{args.outfile_prefix}_T{counter}'

    # Run prediction
    query_chainseq = target.target_chainseq
    query_sequence = query_chainseq.replace('/','')

    alignfile = target.templates_alignfile
    assert exists(alignfile), f"Alignment file not found: {alignfile}"

    all_template_features = load_template_features(alignfile, query_sequence, args)

    msa = [query_sequence]
    deletion_matrix = [[0] * len(query_sequence)]

    all_metrics = predict_utils.run_alphafold_prediction(
        query_sequence=query_sequence,
        msa=msa,
        deletion_matrix=deletion_matrix,
        chainbreak_sequence=query_chainseq,
        template_features=all_template_features,
        model_runners=model_runners,
        out_prefix=outfile_prefix,
        crop_size=crop_size,
        dump_pdbs=not (args.no_pdbs or args.terse),
        dump_metrics=not args.terse,
    )

    # Calculate confidence metrics
    result = calculate_confidence_metrics(target, all_metrics, query_chainseq)

    return result


def process_all_targets(targets, model_runners, args, crop_size):
    """
    Process all target sequences.

    Parameters
    ----------
    targets : pd.DataFrame
        Targets dataframe.
    model_runners : dict
        Dictionary of model runners.
    args : argparse.Namespace
        Command-line arguments.
    crop_size : int
        Crop size.

    Returns
    -------
    list
        List of results for all targets.
    """
    final_results = []
    total_targets = targets.shape[0]

    for counter, targetl in targets.iterrows():
        result = process_single_target(
            targetl, counter, total_targets, model_runners, args, crop_size
        )
        final_results.append(result)

    return final_results


def save_final_results(args, targets, final_results):
    """
    Save final prediction results to TSV file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    targets : pd.DataFrame
        Original targets dataframe.
    final_results : list
        List of results for all targets.
    """
    # Determine output prefix
    if args.final_outfile_prefix:
        outfile_prefix = args.final_outfile_prefix
    elif args.outfile_prefix:
        outfile_prefix = args.outfile_prefix
    elif 'outfile_prefix' in targets.columns:
        outfile_prefix = targets.outfile_prefix.iloc[0]
    else:
        outfile_prefix = None

    # Save results
    if outfile_prefix:
        if args.completed_pred_dir is not None:
            outfile = f'{outfile_prefix}_final_{len(final_results)}.tsv'
        else:
            outfile = f'{outfile_prefix}_final.tsv'

        pd.DataFrame(final_results).to_csv(outfile, sep='\t', index=False)
        print('made:', outfile)


def main():
    """Main execution function for AlphaFold prediction."""
    # Parse arguments
    args = parse_arguments()

    # Load targets
    targets = load_targets(args)

    # Calculate crop size
    crop_size = calculate_crop_size(targets)

    # Print verbose info
    print_verbose_info(args, targets, crop_size)

    # Load model runners
    model_runners = load_model_runners(args, crop_size)

    # Process all targets
    final_results = process_all_targets(targets, model_runners, args, crop_size)

    # Save final results
    save_final_results(args, targets, final_results)


if __name__ == '__main__':
    main()
