######################################################################################88

FREDHUTCH_HACKS = False # silly stuff Phil added for running on Hutch servers
if FREDHUTCH_HACKS:
    import os
    from shutil import which
    os.environ['XLA_FLAGS']='--xla_gpu_force_compilation_parallelism=1'
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = '1'
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '2.0'
    assert which('ptxas') is not None


import argparse

parser = argparse.ArgumentParser(
    description="Run simple template-based alphafold inference",
    epilog = f'''
Examples:

# this command will build models and compute confidence scores for
# all 10mer peptides in HCV_POLG77 bound to HLA-A*02:01, using the default
# alphafold model_2_ptm parameters. You would need to change the --data_dir
# argument to point to the location of the folder containing the alphafold
# params/ subfolder.

python3 run_prediction.py --targets examples/pmhc_hcv_polg_10mers/targets.tsv --data_dir /home/pbradley/csdat/alphafold/data/ --outfile_prefix polg_test1 --model_names model_2_ptm --ignore_identities


    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--outfile_prefix',
                    help='Prefix that will be prepended to the output '
                    'filenames')
parser.add_argument('--final_outfile_prefix',
                    help='Prefix that will be prepended to the final output '
                    'tsv filename')
parser.add_argument('--targets', required=True, help='File listing the targets to '
                    'be modeled. See description of file format in the github '
                    'README and also examples in the examples/*/*tsv')
parser.add_argument('--data_dir', help='Location of AlphaFold params/ folder')

parser.add_argument('--model_names', type=str, nargs='*', default=['model_2_ptm'])
parser.add_argument('--model_params_files', type=str, nargs='*')

parser.add_argument('--verbose', action='store_true')
parser.add_argument('--ignore_identities', action='store_true',
                    help='Ignore the sequence identities column in the templates '
                    'alignment files. Useful when modeling many different peptides '
                    'using the same alignment file.')
parser.add_argument('--no_pdbs', action='store_true', help='Dont write out pdbs')
parser.add_argument('--terse', action='store_true', help='Dont write out pdbs or '
                    'matrices with alphafold confidence values')
parser.add_argument('--no_resample_msa', action='store_true', help='Dont randomly '
                    'resample from the MSA during recycling. Perhaps useful for '
                    'testing...')

args = parser.parse_args()

import os
import sys
from os.path import exists
import itertools
import numpy as np
import pandas as pd

### Ben Orr 1.17.24: Import the version of predict_utils that also returns predicted Protease EC50 values.
import predict_utils_pEC50 as predict_utils

### Ben 10.22.22: mk_mock_template() uses AF2s residue_constants
from alphafold.common import residue_constants

### Ben 10.21.22: Adding colabfold/batch.py::mk_mock_template() to this
### script to make empty template_features dicts
def mk_mock_template(query_sequence, num_temp=1):
    """
    From Sergey Ovchinnikovs colabfold/batch.py
    Returns empty template_features dict
    Edited to use residue_constants instead of templates.residue_constants
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


targets = pd.read_table(args.targets)

lens = [len(x.target_chainseq.replace('/',''))
        for x in targets.itertuples()]
# crop_size = max(lens)
### Ben Orr 1.22.24: Set crop_size to its default size used in fine-tuning, 
### as this crop_size was used to initialize the ProteaseEC50 heads parameters.
crop_size = 190

if args.verbose:
    import jax
    from os import popen # just to get hostname for logging, not necessary
    # print some logging info
    platform = jax.local_devices()[0].platform
    hostname = popen('hostname').readlines()[0].strip()

    print('cmd:', ' '.join(sys.argv))
    print('local_device:', platform, 'hostname:', hostname, 'num_targets:',
          targets.shape[0], 'max_len=', crop_size)

sys.stdout.flush()

model_runners = predict_utils.load_model_runners(
    args.model_names,
    crop_size,
    args.data_dir,
    model_params_files=args.model_params_files,
    resample_msa_in_recycling = not args.no_resample_msa,
)

final_dfl = []
for counter, targetl in targets.iterrows():
    print('START:', counter, 'of', targets.shape[0])

    alignfile = targetl.templates_alignfile
    assert exists(alignfile)

    query_chainseq = targetl.target_chainseq
    if 'outfile_prefix' in targetl:
        outfile_prefix = targetl.outfile_prefix
    else:
        assert args.outfile_prefix is not None
        if 'targetid' in targetl:
            outfile_prefix = args.outfile_prefix+'_'+str(targetl.targetid)
        else:
            outfile_prefix = f'{args.outfile_prefix}_T{counter}'

    query_sequence = query_chainseq.replace('/','')
    num_res = len(query_sequence)

    data = pd.read_table(alignfile)
    cols = ('template_pdbfile target_to_template_alignstring identities '
            'target_len template_len'.split())
    template_features_list = []
    for tnum, row in data.iterrows():
        #(template_pdbfile, target_to_template_alignstring,
        # identities, target_len, template_len) = line[cols]

        ### Ben 3.2.23: Commenting this assertion out -- hopefully no error follows.
        # assert row.target_len == len(query_sequence)
        target_to_template_alignment = {
            int(x.split(':')[0]) : int(x.split(':')[1]) # 0-indexed
            for x in row.target_to_template_alignstring.split(';')
        }

        template_name = f'T{tnum:03d}' # dont think this matters
        ### Ben 10.21.22: Using colabfold/batch.py::mk_mock_template() to 
        ### make empty template_features dicts
        ## For empty template features:
        template_features = mk_mock_template(query_sequence)
        ## For templates with GAP sequence:
        ## First param can be query sequence (for sequence in template) or '-'*len(query_sequence) for 'GAP' sequence
        # template_features = predict_utils.create_single_template_features(
        #     '-'*len(query_sequence), row.template_pdbfile, target_to_template_alignment,
        #     template_name, allow_chainbreaks=True, allow_skipped_lines=True,
        #     expected_identities = None if args.ignore_identities else row.identities,
        #     expected_template_len = row.template_len,
        # )
        template_features_list.append(template_features)

    all_template_features = predict_utils.compile_template_features(
        template_features_list)

    ### Ben 2.28.23: Print all_template_features
    # print('All Template Features:')
    # print(all_template_features)

    msa=[query_sequence]
    deletion_matrix=[[0]*len(query_sequence)]

    all_metrics = predict_utils.run_alphafold_prediction(
        query_sequence=query_sequence,
        msa=msa,
        deletion_matrix=deletion_matrix,
        chainbreak_sequence=query_chainseq,
        template_features=all_template_features,
        model_runners=model_runners,
        out_prefix=outfile_prefix,
        crop_size=crop_size,
        dump_pdbs = not (args.no_pdbs or args.terse),
        dump_metrics = not args.terse,
    )


    outl = targetl.copy()
    for model_name, metrics in all_metrics.items():
        plddts = metrics['plddt']
        paes = metrics.get('predicted_aligned_error', None)

        ### Ben Orr 1.17.24: Adding the predicted_chym_ec50 and predicted_tryp_ec50 values
        ### to the final df
        outl[model_name+'_predicted_chym_ec50'] = metrics['predicted_chym_ec50']
        outl[model_name+'_predicted_tryp_ec50'] = metrics['predicted_tryp_ec50']

        cs = query_chainseq.split('/')
        chain_stops = list(itertools.accumulate(len(x) for x in cs))
        chain_starts = [0]+chain_stops[:-1]
        nres = chain_stops[-1]
        assert nres == num_res
        outl[model_name+'_plddt'] = np.mean(plddts[:nres])
        if paes is not None:
            outl[model_name+'_pae'] = np.mean(paes[:nres,:nres])
        for chain1,(start1,stop1) in enumerate(zip(chain_starts, chain_stops)):
            outl[f'{model_name}_plddt_{chain1}'] = np.mean(plddts[start1:stop1])

            if paes is not None:
                for chain2 in range(len(cs)):
                    start2, stop2 = chain_starts[chain2], chain_stops[chain2]
                    pae = np.mean(paes[start1:stop1,start2:stop2])
                    outl[f'{model_name}_pae_{chain1}_{chain2}'] = pae
    final_dfl.append(outl)

if args.final_outfile_prefix:
    outfile_prefix = args.final_outfile_prefix
elif args.outfile_prefix:
    outfile_prefix = args.outfile_prefix
elif 'outfile_prefix' in targets.columns:
    outfile_prefix = targets.outfile_prefix.iloc[0]
else:
    outfile_prefix = None

if outfile_prefix:
    outfile = f'{outfile_prefix}_final.tsv'
    pd.DataFrame(final_dfl).to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

