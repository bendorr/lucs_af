"""
Analyzes structural metrics and confidence scores for LUCS designs.

Author: Ben Orr
Date: 8.21.23

This module analyzes LUCS (Loop-Helix-Loop Unit Combinatorial Sampling) designs
by calculating structural metrics and confidence scores. It processes Rosetta models
and AlphaFold2/3 predictions to evaluate the quality and accuracy of reshaped regions.

The script performs the following analyses:
    - RMSD calculations between designs and reference structures
    - Helix RMSD calculations
    - TM-score calculations for global structural alignment
    - AlphaFold confidence metrics (pLDDT and pAE)
    - Helix displacement and angle calculations
    - Secondary structure analysis using DSSP

Uses PyRosetta for structural alignment and RMSD calculations, and supports
both AlphaFold2 and AlphaFold3 prediction formats.

Dependencies:
    - PyRosetta: For alignment and RMSD calculations
    - Biopython: For PDB parsing and DSSP analysis
    - tmtools: For TM-score calculations

Example:
    Basic usage with LUCS designs and AlphaFold predictions::

        $ python degree_reshaped.py \\
            --lucs_data /path/to/lucs/designs \\
            --af_data_dirs /path/to/af/predictions \\
            --starting_structure_path /path/to/reference.pdb \\
            --starting_structure_insertion_points /path/to/insertion_points.json \\
            --outfolder /path/to/output \\
            --outfile results.csv
"""

import argparse
import os
import sys
import json
import gzip
import shutil
import glob
import pickle as pkl
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

# For Reshaped Helix Angle and Displacement
from Bio.PDB.DSSP import DSSP
from Bio.PDB import *

# For conversion between CIF and PDB format
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO

# For helix RMSD calculations
import pyrosetta
from pyrosetta import rosetta

# For alignment and RMSD calculations
from pyrosetta import init
import pyrosetta.distributed.dask
from pyrosetta import pose_from_file
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector, OrResidueSelector
from pyrosetta.rosetta.protocols.fold_from_loops.movers import AlignByResidueSelectorMover
# init pyrosetta
init(options='-mute all')

# tmtools for calculating TM score
from tmtools.io import get_structure, get_residue_data
from tmtools.testing import get_pdb_path
from tmtools import tm_align

# Import utility functions for degree_reshaped.py
import utils_degree_reshaped as utils


@dataclass
class StructureData:
    """
    Container for structure data and residue information.

    This dataclass consolidates all structure-related data needed for
    calculating metrics between design models and starting structures.
    It replaces passing many individual parameters to functions.

    Attributes
    ----------
    args : argparse.Namespace
        Command-line arguments and configuration.
    design_struct : Any
        PyRosetta pose of design structure.
    start_struct : Any
        PyRosetta pose of starting structure.
    design_path : str
        Path to design structure file.

    Design residues (pose numbering):
    design_helix_residues : List[List[int]]
        Helix residues in design.
    design_lhl_residues : List[List[int]]
        Loop-Helix-Loop residues in design.
    design_non_bb_remodeled_residues : List[int]
        Non-backbone-remodeled residues in design.
    design_bb_remodeled_residues_string_list : List[str]
        String-formatted backbone-remodeled residues for PyRosetta selectors.

    Starting structure residues (pose numbering):
    ss_helix_residues : List[List[int]]
        Helix residues in starting structure.
    ss_lhl_residues : List[List[int]]
        Loop-Helix-Loop residues in starting structure.
    ss_non_bb_remodeled_residues : List[int]
        Non-backbone-remodeled residues in starting structure.

    PDB numbering variants:
    pdb_design_helix_residues : List[List[int]]
        Design helix residues in PDB numbering.
    pdb_ss_helix_residues : List[List[int]]
        Starting structure helix residues in PDB numbering.
    pdb_design_sheet_residues : List[List[int]]
        Design sheet residues in PDB numbering.

    Alignment data:
    align_residues : Optional[List[List[List[int]]]]
        Optional custom alignment residues.
    trimmed_design_helix_residues : List[List[int]]
        Trimmed helix residues in design after processing.
    trimmed_ss_helix_residues : List[List[int]]
        Trimmed helix residues in starting structure after processing.
    design_non_bb_rem_align_residues : List[List[int]]
        Non-backbone-remodeled alignment residues in design.
    ss_non_bb_rem_align_residues : List[List[int]]
        Non-backbone-remodeled alignment residues in starting structure.

    String-formatted residue lists for PyRosetta selectors:
    design_align_res_str_list : List[str]
        Design alignment residue strings.
    ss_align_res_str_list : List[str]
        Starting structure alignment residue strings.
    design_helix_residues_str_list : List[str]
        Design helix residue strings.
    ss_helix_residues_str_list : List[str]
        Starting structure helix residue strings.

    Analysis parameters:
    helix_idxs : List[int]
        Indices of helices to analyze.
    num_lhls : int
        Number of Loop-Helix-Loop units.
    chain_id : str
        Chain ID for the structure.
    """
    # Configuration
    args: argparse.Namespace

    # Structures
    design_struct: Any
    start_struct: Any

    # Paths
    design_path: str

    # Design residues (pose numbering)
    design_helix_residues: List[List[int]]
    design_lhl_residues: List[List[int]]
    design_non_bb_remodeled_residues: List[int]
    design_bb_remodeled_residues_string_list: List[str]

    # Starting structure residues (pose numbering)
    ss_helix_residues: List[List[int]]
    ss_lhl_residues: List[List[int]]
    ss_non_bb_remodeled_residues: List[int]

    # PDB numbering variants
    pdb_design_helix_residues: List[List[int]]
    pdb_ss_helix_residues: List[List[int]]
    pdb_design_sheet_residues: List[List[int]]

    # Alignment data
    align_residues: Optional[List[List[List[int]]]]
    trimmed_design_helix_residues: List[List[int]]
    trimmed_ss_helix_residues: List[List[int]]
    design_non_bb_rem_align_residues: List[List[int]]
    ss_non_bb_rem_align_residues: List[List[int]]

    # String-formatted residue lists
    design_align_res_str_list: List[str]
    ss_align_res_str_list: List[str]
    design_helix_residues_str_list: List[str]
    ss_helix_residues_str_list: List[str]

    # Analysis parameters
    helix_idxs: List[int]
    num_lhls: int
    chain_id: str


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Calculate Degree of Reshaping metrics on LUCS designs\
        and their AF2/3 predictions.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # LUCS inputs
    parser.add_argument('--lucs_data', nargs='*', type=str, required=True,
            help='Path to a directory or directories containing LUCS designs.')
    parser.add_argument('--lucs_data_subdir_split', action='store_true',
            help='Indicate whether the LUCS designs are split into subdirectories of the\
            lucs_data directory.')
    parser.add_argument('--lucs_af_design_ids_match', action='store_true',
            help="Indicate whether the LUCS design IDs match the AF design IDs exactly. The alternative \
            is that the LUCS design ID is af_design_id.split('_')[0].")
    parser.add_argument('--original_lucs_data', type=str, required=False, default='',
            help='Path to a directory containing the backbones output by LUCS. This differs from the lucs_data \
            folder in that the lucs_data folder may contain Rosetta models of threaded design sequences onto Rosetta\
            backbones.')
    parser.add_argument('--original_lucs_data_subdir_split', action='store_true',
            help='Indicate whether the Original LUCS designs are split into subdirectories of the\
            lucs_data directory.')
    parser.add_argument('--original_LUCS_data_compatible_LHLs', action='store_true',
            help='Indicate whether the Original LUCS designs are contained in a \
            _screen_compatible_loop_helix_loop_units directory (whose filenames are model_<lucs_design_id>.pdb).')
    parser.add_argument('--starting_structure_path', type=str, required=False,
            default='/wynton/home/kortemme/borr/thesis/local_protein_sequence_design/test_inputs/2lv8_inputs/2lv8_cleaned.pdb',
            help='pdb file of the LUCS starting structure.')
    parser.add_argument('--starting_structure_insertion_points', type=str, required=False,
            default='/wynton/home/kortemme/borr/thesis/local_protein_sequence_design/test_inputs/2lv8_inputs/2lv8_insertion_points.json',
            help='json file containing the LUCS starting structures insertion points.')
    parser.add_argument('--start_struct_res_start_idx', type=int, required=False, default=1,
            help='Index on which the starting structures residue numbers begin.')
    parser.add_argument('--filter_by_subsequence', type=str, required=False, default='',
            help='If a designs sequence does not contain this subsequence, then skip the design.')
    parser.add_argument('--lucs_id_suffix', type=str, required=False, default='',
            help='Suffix added to <design_id>/<design_id><suffix>.pdb file names.')
    parser.add_argument('--include_designs_df', type=str, required=False, default=None,
            help='csv file with a design_id column. If provided, only these design IDs '
            'will be analyzed.')

    # RMSD inputs
    parser.add_argument('--use_dssp', action='store_true',
            help='Use DSSP secondary structure calculations to identify helical residues\
            in the reshaped regions.')
    parser.add_argument('--assume_helix', action='store_true',
            help='If fewer than 4 helical residues are found with DSSP, then assume res_list[5:-5] is helical.')
    parser.add_argument('--bb_rem_all_helical', action='store_true',
            help="Assume that all designs' bb_remodeled_residues (in the design_info.json files) are helical.")
    parser.add_argument('--starting_structure_helix_residues_file', type=str, required=False, default=None,
            help='json file containing (a list of lists of) helix residues in the Starting Structure. \
            If not provided, then use DSSP to find reshaped helix residues in the Starting Structure.')

    # Alignment inputs
    parser.add_argument('--use_align_residues_file', action='store_true',
            help='Align the design structures on the residues contained in their <lucs_dir>/<design_id>_align_residues.json \
            files, which contain lists of lists of residues.')
    parser.add_argument('--starting_structure_align_residues_file', type=str, required=False, default=None,
            help='json file containing align_residues (list of lists of residue indices) for the starting structure. \
            Required if using --use_align_residues_file to select residues on which to align for each design.')
    parser.add_argument('--aligned_pdb_outdir', type=str, required=False, default=None,
            help='If provided, then save aligned pdb files when calculating Helix RMSD to the starting structure \
            to the specified directory.')

    # AlphaFold inputs
    parser.add_argument('--af_data_dirs', nargs='*', type=str, required=False, default=None,
            help='Path to a directory or directories containing AF predictions for LUCS designs.\
            If not provided, then skip analyis of AF predictions.')
    parser.add_argument('--af_data_parents', nargs='*', type=str, required=False, default=None,
            help='Path to a parent directory or parent directories containing subdirectories with AF predictions for LUCS designs.')
    parser.add_argument('--is_colabfold_format', action='store_true',
            help='Indicate whether the AF data dir is in Colabfold output format (all PDB and json\
            files for all designs in the same directory).')
    parser.add_argument('--use_relaxed_af', action='store_true',
            help='Use relaxed AF predictions.')
    parser.add_argument('--use_af_sequence', action='store_true',
            help='Store design sequences based off their AF predictions sequences.')

    # AlphaFold3 inputs
    parser.add_argument('--is_af3', action='store_true',
            help='Indicate whether the AlphaFold predictions are from AlphaFold3, and in the format output by the AlphaFold Server.')
    parser.add_argument('--is_af3_linux', action='store_true',
            help='Indicate whether the AlphaFold predictions are from AlphaFold3, and in the format output by AF3 linux installations.')
    parser.add_argument('--af_prefix', type=str, required=False, default='',
            help='Prefix in front of design_id in AF2/3 predictions, such as fold_ for AF3 server outputs.')
    parser.add_argument('--af_is_cif', action='store_true',
            help='Indicate that the AF predictions are .cif files.')

    # ProteinMPNN designs inputs
    parser.add_argument('--pmpnn_suffixes', nargs='*', type=str, required=False, default=[''],
            help='A list of suffixes for multiple ProteinMPNN sequences for a single LUCS backbone.')

    # Parallelization arguments
    parser.add_argument('--num_tasks', type=int, required=False, default=1,
            help='Number of tasks by which to divide the calculations.')
    parser.add_argument('--task_id', type=int, required=False, default=1,
            help='ID of the current task (1-indexed, e.g. $SGE_TASK_ID for SGE job distributor).')

    # Output arguments
    parser.add_argument('--outfolder', type=str, required=True,
            help='Path to the output folder for figures and files containing clustered data.')
    parser.add_argument('--outfile', type=str, required=True,
            help='Name for the output dataframe.')
    parser.add_argument('--skip_completed_df_dir', type=str, required=False, default=None,
            help='If provided, search for the df that should be written out by the current task. If this dataframe \
            exists, then finish the current task.')

    # Debugging arguments
    parser.add_argument('--verbose', action='store_true',
            help='Print extra print statements.')
    parser.add_argument('--test_stop', type=int, required=False, default=None,
            help='Number of test calculations to perform.')

    args = parser.parse_args()

    # Check if the current task's df has already been written out.
    # If it has, then end the current task.
    if args.skip_completed_df_dir != None:
        outfile = args.outfile.replace('.csv', f'_{args.task_id}.csv')
        if os.path.exists(os.path.join(args.skip_completed_df_dir, outfile)):
            print(f'DataFrame {os.path.join(args.skip_completed_df_dir, outfile)} already exists. Skipping this task.')
            exit()

    return args


def load_starting_structure_data(args: argparse.Namespace) -> Tuple[Any, List[int], List[int], List[List[int]], List[List[int]], List[List[int]], List[List[int]], List[str], List[str]]:
    """
    Load starting structure and extract residue information.

    Args:
        args: Parsed command line arguments.

    Returns:
        Tuple containing:
            - start_struct: PyRosetta pose of starting structure
            - ss_bb_remodeled_residues: Backbone remodeled residues (pose numbering)
            - ss_non_bb_remodeled_residues: Non-backbone remodeled residues (pose numbering)
            - ss_lhl_residues: LHL residues (pose numbering)
            - ss_helix_residues: Helix residues (pose numbering)
            - pdb_ss_helix_residues: Helix residues (PDB numbering)
            - pdb_ss_lhl_residues: LHL residues (PDB numbering)
            - ss_non_bb_remodeled_residues_string_list: String list for PyRosetta
            - ss_bb_remodeled_residues_string_list: String list for PyRosetta
    """
    # Load the Starting Structure
    start_struct = pose_from_file(args.starting_structure_path)

    # Select the non-backbone_remodeled residues in the Starting
    # Structure so they can be aligned to the non-backbone_remodeled
    # residues of the LUCS structures
    pdb_ss_bb_remodeled_residues, pdb_ss_non_bb_remodeled_residues, ss_start_stop_idxs = \
        utils.get_remodeled_residues_from_insertion_points(
            args.starting_structure_insertion_points,
            args.start_struct_res_start_idx,
            len(start_struct.sequence()))

    if args.verbose:
        print('Starting Structure bb_remodeled_residues (PDB-numbered):')
        print(pdb_ss_bb_remodeled_residues)
        print()

    # Load a list of lists of Starting Structure LHL residues
    if args.starting_structure_helix_residues_file:
        pdb_ss_lhl_residues, pdb_ss_helix_residues = utils.load_ss_lhl_res_from_file(
                                                            args.starting_structure_helix_residues_file,
                                                            pdb_ss_bb_remodeled_residues)
    else:
        pdb_ss_lhl_residues, pdb_ss_helix_residues = utils.load_ss_lhl_res(start_struct, pdb_ss_bb_remodeled_residues)

    if args.verbose:
        print('Starting Structure LHL residues:')
        for res_list in pdb_ss_lhl_residues:
            print(res_list)
        print('Starting Structure Helix residues:')
        for res_list in pdb_ss_helix_residues:
            print(res_list)

    ### Convert between PDB and PyRosetta residue numberings for Starting Structure
    ss_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(start_struct, pdb_ss_bb_remodeled_residues, verbose=args.verbose)
    ss_non_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(start_struct, pdb_ss_non_bb_remodeled_residues, verbose=args.verbose)
    ss_lhl_residues = []
    for i in range(len(pdb_ss_lhl_residues)):
        ss_lhl_residues.append([])
        ss_lhl_residues[i] = utils.convert_pdb_to_pose_resnums(start_struct, pdb_ss_lhl_residues[i], verbose=args.verbose)
    ss_helix_residues = []
    for i in range(len(pdb_ss_helix_residues)):
        ss_helix_residues.append([])
        ss_helix_residues[i] = utils.convert_pdb_to_pose_resnums(start_struct, pdb_ss_helix_residues[i], verbose=args.verbose)

    # Load strings containing residue ranges for PyRosetta
    ss_non_bb_remodeled_residues_string_list, ss_bb_remodeled_residues_string_list = \
        utils.load_pyrosetta_strings(args.starting_structure_path, ss_lhl_residues)

    return (start_struct, ss_bb_remodeled_residues, ss_non_bb_remodeled_residues,
            ss_lhl_residues, ss_helix_residues, pdb_ss_helix_residues, pdb_ss_lhl_residues,
            ss_non_bb_remodeled_residues_string_list, ss_bb_remodeled_residues_string_list)


def load_design_structure(args: argparse.Namespace, lucs_design_id: str, af_design_id: str,
                          lucs_dir: str, original_lucs_dir: str) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """
    Load the LUCS design structure.

    Args:
        args: Parsed command line arguments.
        lucs_design_id: LUCS design ID.
        af_design_id: AlphaFold design ID.
        lucs_dir: Directory containing LUCS structures.
        original_lucs_dir: Directory containing original LUCS backbones.

    Returns:
        Tuple containing:
            - design_struct: PyRosetta pose of design structure
            - lucs_struct_path: Path to LUCS structure file
            - design_path: Path to Design Model structure file
    """
    # If the original LUCS backbone is provided, then use this structure as the Design Model reference structure:
    if args.original_lucs_data != '':
        lucs_struct_dir = original_lucs_dir
    else:
        lucs_struct_dir = lucs_dir

    # Search for multiple possible LUCS structure filenames (some LUCS structures are
    # named unzipped_design.pdb, while others are named {af_design_id}.pdb)
    lucs_struct_filenames = [f'{af_design_id}{args.lucs_id_suffix}.pdb', 'unzipped_design.pdb']
    lucs_struct_paths = []
    for lucs_struct_file in lucs_struct_filenames:
        if args.verbose:
            print(f'Looking for LUCS file: {os.path.join(lucs_struct_dir, lucs_struct_file)}')
            print(f'Found LUCS Files: {glob.glob(os.path.join(lucs_struct_dir, lucs_struct_file))}')
        for found_file in glob.glob(os.path.join(lucs_struct_dir, lucs_struct_file)):
            lucs_struct_paths.append(found_file)
    lucs_struct_path = lucs_struct_paths[0]

    if os.path.isfile(lucs_struct_path):
        pass
    else:
        with gzip.open(os.path.join(lucs_struct_dir, "design.pdb.gz"), 'rb') as f_in:
            with open(os.path.join(lucs_struct_dir, f"{af_design_id}.pdb"), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    design_path = os.path.join(lucs_struct_path)

    try:
        design_struct = pose_from_file(design_path)
        if args.verbose:
            print('Design Model Sequence and Sequence Length:')
            print(design_struct.sequence())
            print(len(design_struct.sequence()))
    except:
        print('\n\nSkipping Design %s -- Failed to Load Design Model Structure for: %s\n\n' % \
            (lucs_design_id, design_path))
        return None, None, None

    return design_struct, lucs_struct_path, design_path


def load_alphafold_structures(args: argparse.Namespace, af_dir: str, af_design_id: str) -> Tuple[Optional[List[Any]], Optional[List[str]], Optional[Dict], Optional[List[str]]]:
    """
    Load AlphaFold prediction structures and confidence metrics.

    Args:
        args: Parsed command line arguments.
        af_dir: Directory containing AlphaFold predictions.
        af_design_id: AlphaFold design ID.

    Returns:
        Tuple containing:
            - af_structs: List of 5 AlphaFold structure poses
            - af_paths: List of paths to AlphaFold structure files
            - results_plddts: Dictionary of pLDDT values
            - af_model_names: List of AlphaFold model names
    """
    if args.af_data_dirs is None and args.af_data_parents is None:
        return None, None, None, None

    # Load the PyRosetta pose objects for the 5 AF structures
    if args.is_colabfold_format:
        af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_paths = \
            utils.load_colabfold_structs(args, af_dir, af_design_id)
        results_plddts = utils.load_colabfold_plddts(af_dir, af_design_id)
        af_model_names = ['af_model1', 'af_model2', 'af_model3', 'af_model4', 'af_model5']
    else:
        if args.is_af3:
            af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_paths = \
                utils.load_af3_structs(af_dir, design_id=af_design_id)
        elif args.is_af3_linux:
            af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_paths = \
                utils.load_af3_linux_structs(args, af_dir, design_id=af_design_id)
        else:
            if args.use_relaxed_af:
                af_struct_header = 'relaxed'
                suffix = '_ptm.pdb'
            else:
                af_struct_header = 'unrelaxed'
                suffix = '_ptm.pdb'
            af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5, af_paths = \
                utils.load_af_structs(af_dir, header=af_struct_header, suffix=suffix)

        results_plddts = utils.load_af_plddts(af_dir, args, design_id=af_struct_header)
        af_model_names = ['af_model1', 'af_model2', 'af_model3', 'af_model4', 'af_model5']

        if args.verbose:
            print('Loaded results_plddts.')

    af_structs = [af_struct_1, af_struct_2, af_struct_3, af_struct_4, af_struct_5]
    return af_structs, af_paths, results_plddts, af_model_names


def get_design_sequence(args: argparse.Namespace, design_path: str, af_dir: Optional[str],
                        af_design_id: str, af_paths: Optional[List[str]]) -> Tuple[str, str]:
    """
    Get the sequence of the current design.

    Args:
        args: Parsed command line arguments.
        design_path: Path to Design Model structure file.
        af_dir: Directory containing AlphaFold predictions.
        af_design_id: AlphaFold design ID.
        af_paths: List of paths to AlphaFold structure files.

    Returns:
        Tuple containing:
            - sequence: Design sequence
            - chain_id: Chain ID
    """
    # Load the sequence of the current design.
    if args.use_af_sequence and (args.af_data_dirs != None or args.af_data_parents != None):
        # Use the current designs' AF predictions to gather the design sequence.
        if args.is_colabfold_format:
            af_paths = utils.get_colabfold_struct_paths(args, af_dir, af_design_id)
            af_path = af_paths[0]
        else:
            results_fnames = af_paths

            # Add 'relaxed' or 'unrelaxed' to the starts of the filenames.
            if args.is_af3:
                results_files = [f'{af_design_id}{fname}' for fname in results_fnames]
            elif args.use_relaxed_af:
                results_files = ['relaxed' + fname for fname in results_fnames]
            else:
                results_files = ['unrelaxed' + fname for fname in results_fnames]

            af_path = os.path.join(af_dir, results_files[0])

        sequence, chain_id = utils.pdb_to_seq(af_path)

    else:
        # Use the Design Model structure to gather the design sequence.
        sequence, chain_id = utils.pdb_to_seq(design_path)

    return sequence, chain_id


def calc_design_to_ss_metrics(data: StructureData) -> Dict[str, Any]:
    """
    Calculate RMSD and TM-score metrics between Design Model structure and Starting Structure.

    Args:
        data: StructureData object containing all structure and residue information.

    Returns:
        Dictionary containing calculated metrics.
    """
    metrics = {}

    # Calculate the TM score between Design Model and Starting Structure
    design_ss_tm_score = utils.calc_TM_score(data.design_path, data.args.starting_structure_path)
    metrics['TM_score_fd_to_ss'] = design_ss_tm_score

    # Get local references for readability
    design_struct = data.design_struct
    start_struct = data.start_struct
    align_residues = data.align_residues
    design_non_bb_remodeled_residues = data.design_non_bb_remodeled_residues
    ss_non_bb_remodeled_residues = data.ss_non_bb_remodeled_residues

    # Align the design and starting structures on their align_residues, if defined
    if align_residues != None:
        # superimpose_poses_by_residues expects a single list of residues on which to align.
        design_struct, start_struct = utils.superimpose_poses_by_residues(
            design_struct,
            [i for lst in align_residues[1] for i in lst],
            start_struct,
            [i for lst in align_residues[0] for i in lst],
            atom_types=['N', 'CA', 'C'])

    # If align_residues is not defined, align structures on their non-remodeled residues.
    else:
        design_struct, start_struct = utils.superimpose_poses_by_residues(
            design_struct,
            design_non_bb_remodeled_residues,
            start_struct,
            design_non_bb_remodeled_residues,
            atom_types=['N', 'CA', 'C'])

    all_rmsd_types = [
        pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_heavy_including_O,
        pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_ca]
    all_atom_types_names = ['all_heavy_bb', 'CA']
    helix_rmsd_alignment_atom_types = [['N', 'CA', 'C'], ['CA']]

    for rmsd_type, atom_types_name, helix_rmsd_atom_types in zip(all_rmsd_types, all_atom_types_names, helix_rmsd_alignment_atom_types):

        design_ss_helix_rmsd, trimmed_ss_helix_residues, trimmed_design_helix_residues = utils.calculate_bb_remodeled_region_rmsd(data.args,
                data.args.starting_structure_path,
                data.design_path,
                helix_idxs=data.helix_idxs,
                is_af_design=False,
                atom_types=helix_rmsd_atom_types,
                ref_lhl_residues=data.ss_lhl_residues,
                pose_lhl_residues=data.design_lhl_residues,
                ref_helix_residues=data.ss_helix_residues,
                pose_helix_residues=data.design_helix_residues,
                align_residues=align_residues)

        metrics['helix_rmsd_fd_to_ss_' + atom_types_name] = design_ss_helix_rmsd

        if data.args.verbose:
            print(f'Helix RMSDs: {atom_types_name}')
            print(design_ss_helix_rmsd)

        # Calculate Helix RMSD for each individual reshaped helix
        for helix_idx in data.helix_idxs:
            design_ss_single_helix_rmsd, trimmed_ss_single_helix_residues, trimmed_design_single_helix_residues = utils.calculate_bb_remodeled_region_rmsd(data.args,
                    data.args.starting_structure_path,
                    data.design_path,
                    helix_idxs=[helix_idx],
                    is_af_design=False,
                    atom_types=helix_rmsd_atom_types,
                    ref_lhl_residues=data.ss_lhl_residues,
                    pose_lhl_residues=data.design_lhl_residues,
                    ref_helix_residues=data.ss_helix_residues,
                    pose_helix_residues=data.design_helix_residues,
                    align_residues=align_residues)

            metrics[f'helix{helix_idx}_rmsd_fd_to_ss_' + atom_types_name] = design_ss_single_helix_rmsd

            if data.args.verbose:
                print(f'Helix {helix_idx} RMSD: ' + atom_types_name)
                print(design_ss_single_helix_rmsd)
                print(f"\nTrimmed SS and FD Helix Residues:")
                print(' '.join([str(r) for r in trimmed_ss_helix_residues]))
                print(' '.join([str(r) for r in trimmed_design_helix_residues]))

            # Save lists of trimmed helix residues
            metrics[f'trimmed_ss_helix{helix_idx}_residues'] = ' '.join([str(r) for r in trimmed_ss_helix_residues])
            metrics[f'trimmed_design_helix{helix_idx}_residues'] = ' '.join([str(r) for r in trimmed_design_helix_residues])

        # Save lists of Design Model helix residues
        metrics['design_helix_residues'] = ' '.join([str(r) for res_list in data.design_helix_residues for r in res_list])
        for helix_idx in data.helix_idxs:
            metrics[f'fd_helix{helix_idx}_residues'] = ' '.join([str(r) for r in data.design_helix_residues[helix_idx]])

        # Save lists of Starting Structure helix residues
        metrics['ss_helix_residues'] = ' '.join([str(r) for res_list in data.ss_helix_residues for r in res_list])
        for helix_idx in data.helix_idxs:
            metrics[f'ss_helix{helix_idx}_residues'] = ' '.join([str(r) for r in data.ss_helix_residues[helix_idx]])

        # Calculate Steph's Helix RMSD functions
        # If align_residues is not defined, align structures on their non-remodeled residues.
        if align_residues == None:
            design_non_bb_rem_align_residues = utils.split_contiguous_residues(design_non_bb_remodeled_residues)
            ss_non_bb_rem_align_residues = utils.split_contiguous_residues(ss_non_bb_remodeled_residues)
        else:
            design_non_bb_rem_align_residues = align_residues[1]
            ss_non_bb_rem_align_residues = align_residues[0]

        design_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in design_non_bb_rem_align_residues]
        ss_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in ss_non_bb_rem_align_residues]

        if data.args.verbose:
            print('design_align_res_str_list:')
            print(design_align_res_str_list)
            print('ss_align_res_str_list:')
            print(ss_align_res_str_list)

        # Align the design and starting structures on their non-remodeled residues
        design_struct, start_struct = utils.steph_align_by_residues(
            design_struct, start_struct,
            design_align_res_str_list, ss_align_res_str_list)

        # Define lists of strings of the helix residues for the design and starting structures
        # Ensure that each list of helix residues is of the same length, for utils.steph_calculate_RMSDs_on_beta_only_helix()
        design_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_design_helix_residues]
        ss_helix_residues_str_list = [f'{res[0]}-{res[-1]}' for res in trimmed_ss_helix_residues]

        # Calculate Helix RMSDs between Design Model and Starting Struct
        # This function uses rmsd_protein_bb_heavy_including_O
        # This function also requires that the residue selectors have the same number of residues
        design_ss_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
            design_struct, start_struct,
            design_helix_residues_str_list, ss_helix_residues_str_list)

        # Save the Helix RMSDs to the sub_dict
        for i, rmsd in enumerate(design_ss_helix_rmsds):
            metrics[f'helix{i}_rmsd_fd_to_ss_bb_heavy_including_O'] = rmsd
            if data.args.verbose:
                print(f'Helix {i} RMSD FD to SS: {rmsd}')

        # Find the longest common helix indices between the design and starting structures
        fd_shared_helix_resn, ss_shared_helix_resn = utils.steph_get_shared_helical_resn(
            design_struct, start_struct,
            data.design_lhl_residues, data.ss_lhl_residues)

        # Define the shared helix residues as lists of strings
        fd_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in fd_shared_helix_resn]
        ss_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in ss_shared_helix_resn]

        # Calculate the Helix RMSDs on the longest common helices
        design_ss_shared_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
            design_struct, start_struct,
            fd_shared_helix_resn_str_list, ss_shared_helix_resn_str_list)

        # Save the shared Helix RMSDs to the sub_dict
        for i, rmsd in enumerate(design_ss_shared_helix_rmsds):
            metrics[f'common_helix{i}_rmsd_fd_to_ss_bb_heavy_including_O'] = rmsd
            if data.args.verbose:
                print(f'Common Helix {i} RMSD FD to SS: {rmsd}')


    # Store computed values back in StructureData for use by other functions
    data.trimmed_design_helix_residues = trimmed_design_helix_residues
    data.trimmed_ss_helix_residues = trimmed_ss_helix_residues
    data.design_non_bb_rem_align_residues = design_non_bb_rem_align_residues
    data.ss_non_bb_rem_align_residues = ss_non_bb_rem_align_residues
    data.design_align_res_str_list = design_align_res_str_list
    data.ss_align_res_str_list = ss_align_res_str_list
    data.design_helix_residues_str_list = design_helix_residues_str_list
    data.ss_helix_residues_str_list = ss_helix_residues_str_list

    return metrics


def process_alphafold_predictions(data: StructureData, af_structs: List[Any], 
                                  af_paths: List[str], af_model_names: List[str]) -> Dict[str, Any]:
    """
    Process AlphaFold predictions and calculate metrics.

    This function processes all AF models and calculates:
    - TM-scores
    - Helix displacements and angles
    - RMSDs to Design Model and Starting Structure

    Args:
        data: StructureData object containing all structure and residue information.
        af_structs: List of AlphaFold prediction PyRosetta poses.
        af_paths: List of paths to AlphaFold prediction pdb files.
        af_model_names: List of AlphaFold model names.

    Returns:
        Dictionary containing calculated AF metrics.
    """

    # Get local references from StructureData for readability
    args = data.args
    design_struct = data.design_struct
    start_struct = data.start_struct
    design_path = data.design_path
    design_helix_residues = data.design_helix_residues
    ss_helix_residues = data.ss_helix_residues
    pdb_design_helix_residues = data.pdb_design_helix_residues
    pdb_ss_helix_residues = data.pdb_ss_helix_residues
    design_lhl_residues = data.design_lhl_residues
    ss_lhl_residues = data.ss_lhl_residues
    pdb_design_sheet_residues = data.pdb_design_sheet_residues
    design_non_bb_remodeled_residues = data.design_non_bb_remodeled_residues
    ss_non_bb_remodeled_residues = data.ss_non_bb_remodeled_residues
    design_bb_remodeled_residues_string_list = data.design_bb_remodeled_residues_string_list
    align_residues = data.align_residues
    helix_idxs = data.helix_idxs
    num_lhls = data.num_lhls
    chain_id = data.chain_id
    trimmed_design_helix_residues = data.trimmed_design_helix_residues
    trimmed_ss_helix_residues = data.trimmed_ss_helix_residues
    design_non_bb_rem_align_residues = data.design_non_bb_rem_align_residues
    ss_non_bb_rem_align_residues = data.ss_non_bb_rem_align_residues
    design_align_res_str_list = data.design_align_res_str_list
    ss_align_res_str_list = data.ss_align_res_str_list
    design_helix_residues_str_list = data.design_helix_residues_str_list
    ss_helix_residues_str_list = data.ss_helix_residues_str_list

    metrics = {}

    all_rmsd_types = [
        pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_heavy_including_O,
        pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_ca]
    all_atom_types_names = ['all_heavy_bb', 'CA']
    helix_rmsd_alignment_atom_types = [['N', 'CA', 'C'], ['CA']]

    for af_idx in range(len(af_structs)):
        af_struct = af_structs[af_idx]
        af_model_name = af_model_names[af_idx]

        if args.verbose:
            print("\nCurrent AF structure:")
            print(af_model_name)

        af_path = af_paths[af_idx]

        # Calculate TM score between the current AF structure and the Starting Structure
        af_ss_tm_score = utils.calc_TM_score(af_path, args.starting_structure_path)
        metrics['TM_score_ss_to_' + af_model_name] = af_ss_tm_score

        # Calculate TM score between the current AF structure and the Design Model Structure
        af_fd_tm_score = utils.calc_TM_score(af_path, design_path)
        metrics['TM_score_fd_to_' + af_model_name] = af_fd_tm_score

        # Align the AF and starting structures on their align_residues, if defined
        if align_residues != None:
            af_struct, start_struct = utils.superimpose_poses_by_residues(
                af_struct,
                [i for lst in align_residues[1] for i in lst],
                start_struct,
                [i for lst in align_residues[0] for i in lst],
                atom_types=['N', 'CA', 'C'])
        else:
            af_struct, start_struct = utils.superimpose_poses_by_residues(
                af_struct,
                design_non_bb_remodeled_residues,
                start_struct,
                design_non_bb_remodeled_residues,
                atom_types=['N', 'CA', 'C'])

        # Calculate the Helix Displacements and Angles for AF predictions
        af_helix_displacements = utils.get_helix_displacements(
                af_struct,
                af_path,
                design_helix_residues,
                chain_id=chain_id,
                struct_res_start_idx=1,
                verbose=args.verbose,
                sheet_residues=pdb_design_sheet_residues)

        if args.verbose:
            print(f"\n\n\n{af_model_name} Helix Displacements: {af_helix_displacements}\n\n\n")

        # Calculate the helix displacements to Starting Structure
        af_helix_displacements_to_ss = utils.get_helix_displacements_to_ss(
                args.starting_structure_path,
                af_path,
                pdb_ss_helix_residues,
                pdb_design_helix_residues,
                ss_chain_id='A',
                design_chain_id='A',
                ss_res_start_idx=args.start_struct_res_start_idx,
                design_res_start_idx=1,
                verbose=args.verbose)

        if args.verbose:
            print(f"\n\n\n{af_model_name} Helix Displacements to Starting Structure: {af_helix_displacements_to_ss}\n\n\n")

        # Calculate sheet direction vector
        sheet_dir_vect = utils.get_strand_dir_vec(af_struct,
                pdb_design_sheet_residues,
                chain_id='A')

        af_helix_angles = utils.get_helix_angles(
                af_struct,
                design_helix_residues,
                chain_id=chain_id,
                struct_res_start_idx=1,
                verbose=args.verbose,
                sheet_residues=pdb_design_sheet_residues,
                sheet_dir_vect=sheet_dir_vect)

        if args.verbose:
            print(f"\n\n\n{af_model_name} Helix Angles: {af_helix_angles}\n\n\n")

        # Calculate Helix Angles between AF predictions and the starting structure
        af_helix_angles_to_ss = utils.get_helix_angles_to_ss(
                start_struct,
                af_struct,
                pdb_ss_helix_residues,
                pdb_design_helix_residues,
                ss_chain_id='A',
                design_chain_id='A',
                ss_struct_res_start_idx=args.start_struct_res_start_idx,
                design_struct_res_start_idx=1,
                verbose=args.verbose)

        if args.verbose:
            print(f"\n\n\n{af_model_name} Model Helix Angles to Starting Structure: {af_helix_angles_to_ss}\n\n\n")

        for lhl_idx in range(num_lhls):
            metrics[f'helix{lhl_idx}_displacement_{af_model_name}'] = af_helix_displacements[lhl_idx]
            metrics[f'helix{lhl_idx}_angle_{af_model_name}'] = af_helix_angles[lhl_idx]
            metrics[f'helix{lhl_idx}_displacement_{af_model_name}_to_ss'] = af_helix_displacements_to_ss[lhl_idx]
            metrics[f'helix{lhl_idx}_angle_{af_model_name}_to_ss'] = af_helix_angles_to_ss[lhl_idx]

        # Process each RMSD type
        for rmsd_type, atom_types_name, helix_rmsd_atom_types in zip(all_rmsd_types, all_atom_types_names, helix_rmsd_alignment_atom_types):

            # Align on non-backbone remodeled residues (FD-AF).
            design_struct, af_struct = utils.align_by_residues(args, design_struct, af_struct,
                design_bb_remodeled_residues_string_list, design_bb_remodeled_residues_string_list)

            # Calculate RMSDs over specific residue sets (FD-AF).
            af_fd_reshaped_rmsd = utils.calculate_RMSD(design_struct, af_struct, design_bb_remodeled_residues_string_list,
                rmsd_type=rmsd_type)

            # Save the all-reshaped-residue RMSD between AF and Design Model
            metrics[f'rmsd_reshaped_{atom_types_name}_fd_to_{af_model_name}'] = af_fd_reshaped_rmsd

            for helix_idx in helix_idxs:
                af_fd_lhl_rmsd = utils.calculate_RMSD(design_struct, af_struct,
                    [f"{design_lhl_residues[helix_idx][0]}-{design_lhl_residues[helix_idx][-1]}"],
                    rmsd_type=rmsd_type)

                metrics[f'lhl{helix_idx}_rmsd_{atom_types_name}_fd_to_{af_model_name}'] = af_fd_lhl_rmsd

            # Align on all residues (FD-AF).
            design_struct, af_struct = utils.align_by_residues(args, design_struct, af_struct,
                [f"1-{len(design_struct.sequence())}"], [f"1-{len(af_struct.sequence())}"])

            # Calculate RMSD over all residues (FD-AF).
            all_res_rmsd = utils.calculate_RMSD(design_struct, af_struct, [f"1-{len(design_struct.sequence())}"],
                rmsd_type=rmsd_type)

            metrics[f'rmsd_all_res_{atom_types_name}_fd_to_{af_model_name}'] = all_res_rmsd

            # Calculate Helix RMSDs between the current AF prediction and Design Model Struct
            af_fd_helix_rmsd, trimmed_design_helix_residues, trimmed_af_helix_residues = utils.calculate_bb_remodeled_region_rmsd(data.args,
                    design_path,
                    af_path,
                    helix_idxs=helix_idxs,
                    is_af_design=True,
                    atom_types=helix_rmsd_atom_types,
                    ref_lhl_residues=design_lhl_residues,
                    pose_lhl_residues=design_lhl_residues,
                    ref_helix_residues=design_helix_residues,
                    pose_helix_residues=design_helix_residues)

            metrics[f'helix_rmsd_{af_model_name}_to_fd_{atom_types_name}'] = af_fd_helix_rmsd

            if args.verbose:
                print(f"{af_model_name} path: {af_path}")
                print(f'Helix RMSD AF to FD {atom_types_name}: {af_fd_helix_rmsd}')

            # Calculate Helix RMSD between AF and FD for each individual reshaped helix
            for helix_idx in helix_idxs:
                af_fd_single_helix_rmsd, trimmed_design_single_helix_residues, trimmed_af_single_helix_residues = utils.calculate_bb_remodeled_region_rmsd(data.args,
                    design_path,
                    af_path,
                    helix_idxs=[helix_idx],
                    is_af_design=True,
                    atom_types=helix_rmsd_atom_types,
                    ref_lhl_residues=design_lhl_residues,
                    pose_lhl_residues=design_lhl_residues,
                    ref_helix_residues=design_helix_residues,
                    pose_helix_residues=design_helix_residues)

                metrics[f'helix{helix_idx}_rmsd_{af_model_name}_to_fd_{atom_types_name}'] = af_fd_single_helix_rmsd

                if args.verbose:
                    print(f'Helix {helix_idx} RMSD {af_model_name} to FD {atom_types_name}: {af_fd_single_helix_rmsd}')

            # Calculate Helix RMSDs between the current AF prediction and Starting Struct
            af_ss_helix_rmsd, trimmed_ss_helix_residues, trimmed_af_helix_residues = utils.calculate_bb_remodeled_region_rmsd(data.args,
                    args.starting_structure_path,
                    af_path,
                    helix_idxs=helix_idxs,
                    is_af_design=True,
                    atom_types=helix_rmsd_atom_types,
                    ref_lhl_residues=ss_lhl_residues,
                    pose_lhl_residues=design_lhl_residues,
                    ref_helix_residues=ss_helix_residues,
                    pose_helix_residues=design_helix_residues)

            metrics[f'helix_rmsd_{af_model_name}_to_ss_{atom_types_name}'] = af_ss_helix_rmsd

            if args.verbose:
                print(f'Helix RMSD {af_model_name} to SS {atom_types_name}: {af_ss_helix_rmsd}')

            # Calculate Helix RMSD between AF and SS for each individual reshaped helix
            for helix_idx in helix_idxs:
                af_ss_single_helix_rmsd, trimmed_ss_single_helix_residues, trimmed_af_single_helix_residues = utils.calculate_bb_remodeled_region_rmsd(data.args,
                    args.starting_structure_path,
                    af_path,
                    helix_idxs=[helix_idx],
                    is_af_design=True,
                    atom_types=helix_rmsd_atom_types,
                    ref_lhl_residues=ss_lhl_residues,
                    pose_lhl_residues=design_lhl_residues,
                    ref_helix_residues=ss_helix_residues,
                    pose_helix_residues=design_helix_residues)

                metrics[f'helix{helix_idx}_rmsd_{af_model_name}_to_ss_{atom_types_name}'] = af_ss_single_helix_rmsd

                if args.verbose:
                    print(f'Helix {helix_idx} RMSD {af_model_name} to SS {atom_types_name}: {af_ss_single_helix_rmsd}')

        # Use Steph's Alignment and RMSD functions

        # If the lengths of the AF prediction and Design Model structures are different, then trim
        # the align residues to the length of the shorter structure.
        af_non_bb_rem_align_residues = design_non_bb_rem_align_residues.copy()
        if len(design_struct.sequence()) != len(af_struct.sequence()):
            min_len = min(len(design_struct.sequence()), len(af_struct.sequence()))
            af_non_bb_rem_align_residues[-1] = [res for res in af_non_bb_rem_align_residues[-1] if res <= min_len]

        af_align_res_str_list = [f'{res[0]}-{res[-1]}' for res in af_non_bb_rem_align_residues]

        # Align the design and starting structures on their non-remodeled residues
        af_struct, start_struct = utils.steph_align_by_residues(
            af_struct, start_struct,
            af_align_res_str_list, ss_align_res_str_list)

        # Calculate Helix RMSDs between AF and Starting Struct
        af_ss_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
            af_struct, start_struct,
            design_helix_residues_str_list, ss_helix_residues_str_list)

        # Save the Helix RMSDs to the sub_dict
        for i, rmsd in enumerate(af_ss_helix_rmsds):
            metrics[f'helix{i}_rmsd2_{af_model_name}_to_ss_bb_heavy_including_O'] = rmsd
            if args.verbose:
                print(f'Helix {i} RMSD2 {af_model_name} to SS: {rmsd}')

        # Find the longest common helix indices between the design and starting structures
        af_shared_helix_resn, ss_shared_helix_resn = utils.steph_get_shared_helical_resn(
            af_struct, start_struct,
            design_lhl_residues, data.ss_lhl_residues)

        # Define the shared helix residues as lists of strings
        af_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in af_shared_helix_resn]
        ss_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in ss_shared_helix_resn]

        # Calculate the Helix RMSDs on the longest common helices
        af_ss_shared_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
            af_struct, start_struct,
            af_shared_helix_resn_str_list, ss_shared_helix_resn_str_list)

        # Save the shared Helix RMSDs to the sub_dict
        for i, rmsd in enumerate(af_ss_shared_helix_rmsds):
            metrics[f'common_helix{i}_rmsd_{af_model_name}_to_ss_bb_heavy_including_O'] = rmsd
            if args.verbose:
                print(f'Common Helix {i} RMSD {af_model_name} to SS: {rmsd}')

        # Align AF and FD structures on their non-remodeled residues
        af_struct, design_struct = utils.steph_align_by_residues(
            af_struct, design_struct,
            af_align_res_str_list, design_align_res_str_list)

        # Calculate Helix RMSDs between AF and Design Model Struct
        af_fd_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
            af_struct, design_struct,
            design_helix_residues_str_list, design_helix_residues_str_list)

        # Save the Helix RMSDs to the sub_dict
        for i, rmsd in enumerate(af_fd_helix_rmsds):
            metrics[f'helix{i}_rmsd2_{af_model_name}_to_fd_bb_heavy_including_O'] = rmsd
            if args.verbose:
                print(f'Helix {i} RMSD2 {af_model_name} to FD: {rmsd}')

        # Find the longest common helix indices between AF and FD
        af_shared_helix_resn, fd_shared_helix_resn = utils.steph_get_shared_helical_resn(
            af_struct, design_struct,
            design_lhl_residues, design_lhl_residues)

        # Define the shared helix residues as lists of strings
        af_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in af_shared_helix_resn]
        fd_shared_helix_resn_str_list = [f'{res[0]}-{res[-1]}' if len(res) > 1 else None for res in fd_shared_helix_resn]

        # Calculate the Helix RMSDs on the longest common helices
        af_fd_shared_helix_rmsds = utils.steph_calculate_RMSDs_on_beta_only_helix(
            af_struct, design_struct,
            af_shared_helix_resn_str_list, fd_shared_helix_resn_str_list)

        # Save the shared Helix RMSDs to the sub_dict
        for i, rmsd in enumerate(af_fd_shared_helix_rmsds):
            metrics[f'common_helix{i}_rmsd2_{af_model_name}_to_fd_bb_heavy_including_O'] = rmsd
            if args.verbose:
                print(f'Common Helix {i} RMSD2 {af_model_name} to FD: {rmsd}')

    return metrics


def update_dataframe(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Best-AF-model columns and Rosetta metrics to the DataFrame and save.

    Args:
        args: Parsed command line arguments.
        sub_dicts: List of dictionaries containing metrics for each design.

    Returns:
        Final DataFrame with all metrics.
    """

    if args.af_data_dirs != None or args.af_data_parents != None:
        # Add columns to the orderable_df denoting the best AF model w.r.t. the
        # specified AF metric.
        df_col_prefix_values = 'best_af_value_'
        df_col_prefix_models = 'best_af_model_'

        # Prefix to ave_plddt_af_model1
        af_metric = 'ave_plddt_af_'
        af_metric_low_better = False
        df = utils.update_df_with_best_af_model(df, df_col_prefix_values,
            df_col_prefix_models, af_metric, af_metric_low_better, add_best_af_metrics=True)

        # Prefix to rmsd_reshaped_CA_fd_to_af_model1
        af_metric = 'rmsd_reshaped_CA_fd_to_af_'
        af_metric_low_better = True
        df = utils.update_df_with_best_af_model(df, df_col_prefix_values,
            df_col_prefix_models, af_metric, af_metric_low_better, add_best_af_metrics=True)

        # Prefix to rmsd_reshaped_CA_fd_to_af_model1
        af_metric = 'rmsd_reshaped_all_heavy_bb_fd_to_af_'
        af_metric_low_better = True
        df = utils.update_df_with_best_af_model(df, df_col_prefix_values,
            df_col_prefix_models, af_metric, af_metric_low_better, add_best_af_metrics=True)

    # Update df with LUCS metrics
    df = utils.update_df_with_LUCS_metrics(df)

    # Edit the outfile to include the args.task_id, to save args.num_tasks
    # different dfs that can be combined later.
    outfile = args.outfile.replace('.csv', f'_{args.task_id}.csv')

    # Save the dataframe in a csv file
    df.to_csv(os.path.join(args.outfolder, outfile))

    print('Saved Dataframe: %s' % os.path.join(args.outfolder, outfile))

    return df


def main():
    """
    Main function to analyze LUCS designs and AlphaFold predictions.

    Orchestrates the entire analysis workflow:
    1. Parse arguments and validate
    2. Load starting structure and metadata
    3. Process each design:
        - Load LUCS structure
        - Load AlphaFold predictions
        - Calculate metrics (RMSD, TM-score, confidence)
    4. Aggregate results and save to CSV
    """
    # Parse command line arguments
    args = parse_arguments()

    # Make output folder
    os.makedirs(args.outfolder, exist_ok=True)

    # Load starting structure and all related residue data
    (start_struct, ss_bb_remodeled_residues, ss_non_bb_remodeled_residues,
     ss_lhl_residues, ss_helix_residues, pdb_ss_helix_residues, pdb_ss_lhl_residues,
     ss_non_bb_remodeled_residues_string_list, ss_bb_remodeled_residues_string_list) = \
        load_starting_structure_data(args)

    # Initialize list to store metrics for all designs
    sub_dicts: List[Dict[str, Any]] = []

    # Gather list of design IDs and their AF folders from the AF output
    design_ids_af_dirs = utils.get_design_ids_af_dirs(args)

    if args.verbose:
        print('Number of design IDs and their AF dirs:')
        print(len(design_ids_af_dirs))
        print()

    ## The Main Loop: Iterate through designs and calculate metrics
    task_idx = 0

    # Iterate over the design IDs and their AF outputs dirs
    for lucs_design_id, af_design_id, af_dir in design_ids_af_dirs:

        task_idx += 1
        # Parallelization: Skip designs not assigned to this task
        if task_idx % args.num_tasks != (args.task_id - 1):
            continue

        if args.verbose:
            print(f'\n\n\nLUCS design ID: {lucs_design_id}')
            print(f'AF design ID: {af_design_id}')
            print(f'AF dir: {af_dir}')

        # Find the LUCS dir with the current design
        # If lucs_design_ids are prefixes to full design_id (e.g., 100065_mpnnseq1),
        # find the lucs_dir with just the lucs_design_id
        # AF3 design IDs contain the appropriate AF3 prefix (--args.af_prefix 'fold_')
        if args.lucs_af_design_ids_match:
            lucs_dir = utils.find_lucs_dir(args, af_design_id)
        else:
            lucs_dir = utils.find_lucs_dir(args, lucs_design_id)

        # Initialize sub_dict to store metrics for this design (will become a row in output df)
        sub_dict: Dict[str, Any] = {}
        sub_dict['design_id'] = af_design_id
        sub_dict['af_design_id'] = af_design_id
        sub_dict['lucs_design_id'] = lucs_design_id
        sub_dict['af_location'] = af_dir
        sub_dict['lucs_location'] = lucs_dir

        # Find the original LUCS backbone's location
        if args.original_lucs_data != '':
            original_lucs_dir = utils.find_original_lucs_dir(args, lucs_design_id)
            sub_dict['original_lucs_location'] = original_lucs_dir
        else:
            original_lucs_dir = ''
            sub_dict['original_lucs_location'] = 'Not Provided'

        # Load the LUCS design structure
        design_struct, lucs_struct_path, design_path = load_design_structure(
            args, lucs_design_id, af_design_id, lucs_dir, original_lucs_dir)

        # Skip if structure failed to load
        if design_struct is None:
            continue

        # Load AlphaFold predictions if available
        af_structs, af_paths, results_plddts, af_model_names = load_alphafold_structures(
            args, af_dir, af_design_id)

        # Get the design sequence
        sequence, chain_id = get_design_sequence(args, design_path, af_dir, af_design_id, af_paths)
        sub_dict['sequence'] = sequence

        # Filter by subsequence if specified
        if args.filter_by_subsequence != '':
            if not sequence.endswith(args.filter_by_subsequence):
                print(f'Skipping design {lucs_design_id} because it does not end with the subsequence {args.filter_by_subsequence}.')
                continue

        # Get design residue information
        fd_reslist = utils.get_residue_list_from_pdb(design_path)
        fd_res_start_idx = fd_reslist[0]
        if args.verbose:
            print(f"Design model residue start index: {fd_res_start_idx}")

        # Find design_info.json file
        if os.path.exists(os.path.join(lucs_dir, "design_info.json")):
            design_info_path = os.path.join(lucs_dir, "design_info.json")
        elif os.path.exists(os.path.join(lucs_dir, f"{af_design_id}_design_info.json")):
            design_info_path = os.path.join(lucs_dir, f"{af_design_id}_design_info.json")
        else:
            print(f"Design info file not found in {lucs_dir}. Skipping.")
            continue

        # Get remodeled residues (PDB-numbered)
        pdb_design_bb_remodeled_residues, pdb_design_non_bb_remodeled_residues = \
            utils.get_remodeled_residues_from_design_info(
                pdb_file=lucs_struct_path,
                design_info_file=design_info_path)

        sub_dict['bb_remodeled_residues'] = ' '.join([str(i) for i in pdb_design_bb_remodeled_residues])

        # Get LHL and helix residues (PDB-numbered)
        pdb_design_lhl_residues, pdb_design_helix_residues = utils.get_lhl_residues(
            args,
            design_info_file=design_info_path,
            seq_len=len(design_struct.sequence()),
            pose=design_struct,
            use_dssp=args.use_dssp,
            assume_helix=args.assume_helix,
            bb_rem_all_helical=args.bb_rem_all_helical)

        num_lhls = len(pdb_design_lhl_residues)

        if args.verbose:
            print(f'Number of LHLs: {num_lhls}')
            print('Design Model LHL residues:')
            for res_list in pdb_design_lhl_residues:
                print(res_list)
            print('Design Model Helix residues:')
            for res_list in pdb_design_helix_residues:
                print(res_list)

        # Skip if no helical residues found
        if len(pdb_design_helix_residues) == 0:
            print(f'No Helical Residues Found for {lucs_design_id}. Skipping RMSD calculations.')
            continue

        # Load pLDDT values if AF data is available
        if af_structs is not None:
            all_lhl_plddts, all_non_bb_rem_plddts = \
                utils.get_lhl_plddt_values(results_plddts, pdb_design_lhl_residues)

            if args.verbose:
                print('\n\nLoading pLDDTs into sub_dict')

            sub_dict = utils.load_plddts_into_subdict(sub_dict, results_plddts,
                                                       all_lhl_plddts, all_non_bb_rem_plddts)

        # Convert PDB to PyRosetta pose numbering
        design_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(
            design_struct, pdb_design_bb_remodeled_residues, verbose=args.verbose)
        design_non_bb_remodeled_residues = utils.convert_pdb_to_pose_resnums(
            design_struct, pdb_design_non_bb_remodeled_residues, verbose=args.verbose)

        design_lhl_residues = []
        for i in range(len(pdb_design_lhl_residues)):
            design_lhl_residues.append([])
            design_lhl_residues[i] = utils.convert_pdb_to_pose_resnums(
                design_struct, pdb_design_lhl_residues[i], verbose=args.verbose)

        design_helix_residues = []
        for i in range(len(pdb_design_helix_residues)):
            design_helix_residues.append([])
            design_helix_residues[i] = utils.convert_pdb_to_pose_resnums(
                design_struct, pdb_design_helix_residues[i], verbose=args.verbose)

        # Load PyRosetta string representations of residue ranges
        design_non_bb_remodeled_residues_string_list, design_bb_remodeled_residues_string_list = \
            utils.load_pyrosetta_strings(lucs_struct_path, design_lhl_residues)

        if args.verbose:
            print('Design Model bb_rem and non-bb_rem residues:')
            print(design_bb_remodeled_residues_string_list)
            print(design_non_bb_remodeled_residues_string_list)

        # Determine helix indices to analyze
        helix_idxs = [i for i in range(len(design_helix_residues))]

        if args.verbose:
            print(f'Helix Indices: {helix_idxs}')

        # Determine alignment residues
        if args.use_align_residues_file:
            # Get specified alignment residues
            pdb_align_residues = utils.get_design_to_ss_aligned_residues(args, design_path)

            if args.verbose:
                print('Residues on which to align:')
                print(f'Starting Structure: {pdb_align_residues[0]}')
                print(f'Design Structure: {pdb_align_residues[1]}\n')

            # Convert to PyRosetta pose numbering
            align_residues = [[], []]
            for i in range(len(pdb_align_residues[0])):
                align_residues[0].append([])
                align_residues[0][i] = utils.convert_pdb_to_pose_resnums(
                    start_struct, pdb_align_residues[0][i], verbose=args.verbose)
            for i in range(len(pdb_align_residues[1])):
                align_residues[1].append([])
                align_residues[1][i] = utils.convert_pdb_to_pose_resnums(
                    design_struct, pdb_align_residues[1][i], verbose=args.verbose)

            pdb_ss_all_sheet_residues = pdb_align_residues[0]
            pdb_design_all_sheet_residues = pdb_align_residues[1]
        else:
            # Align on non-reshaped residues by default
            align_residues = None
            pdb_ss_all_sheet_residues = None
            pdb_design_all_sheet_residues = None

        # Get sheet residues (PDB-numbered for Biopython calculations)
        pdb_ss_sheet_residues = utils.get_sheet_residues(
            start_struct,
            pdb_start_idx=args.start_struct_res_start_idx,
            strand_nums=None,
            all_sheet_residues=pdb_ss_all_sheet_residues)

        pdb_design_sheet_residues = utils.get_sheet_residues(
            design_struct,
            pdb_start_idx=fd_res_start_idx,
            strand_nums=None,
            all_sheet_residues=pdb_design_all_sheet_residues)

        # Create StructureData object to pass to calculation functions
        data = StructureData(
            args=args,
            design_struct=design_struct,
            start_struct=start_struct,
            design_path=design_path,
            design_helix_residues=design_helix_residues,
            design_lhl_residues=design_lhl_residues,
            design_non_bb_remodeled_residues=design_non_bb_remodeled_residues,
            design_bb_remodeled_residues_string_list=design_bb_remodeled_residues_string_list,
            ss_helix_residues=ss_helix_residues,
            ss_lhl_residues=ss_lhl_residues,
            ss_non_bb_remodeled_residues=ss_non_bb_remodeled_residues,
            pdb_design_helix_residues=pdb_design_helix_residues,
            pdb_ss_helix_residues=pdb_ss_helix_residues,
            pdb_design_sheet_residues=pdb_design_sheet_residues,
            align_residues=align_residues,
            trimmed_design_helix_residues=[],  # Will be populated by calc_design_to_ss_metrics
            trimmed_ss_helix_residues=[],      # Will be populated by calc_design_to_ss_metrics
            design_non_bb_rem_align_residues=[], # Will be populated by calc_design_to_ss_metrics
            ss_non_bb_rem_align_residues=[],     # Will be populated by calc_design_to_ss_metrics
            design_align_res_str_list=[],      # Will be populated by calc_design_to_ss_metrics
            ss_align_res_str_list=[],          # Will be populated by calc_design_to_ss_metrics
            design_helix_residues_str_list=[], # Will be populated by calc_design_to_ss_metrics
            ss_helix_residues_str_list=[],     # Will be populated by calc_design_to_ss_metrics
            helix_idxs=helix_idxs,
            num_lhls=num_lhls,
            chain_id=chain_id
        )

        # Calculate LUCS Design Model to Starting Structure metrics
        lucs_metrics = calc_design_to_ss_metrics(data)

        # Add LUCS metrics to sub_dict
        sub_dict.update(lucs_metrics)

        # Process AlphaFold predictions if available
        if af_structs is not None:
            # All needed data is now in the StructureData object, populated by calc_design_to_ss_metrics()
            # Process all AF models using the StructureData object
            af_metrics = process_alphafold_predictions(data, af_structs, af_paths, af_model_names)

            # Add AF metrics to sub_dict
            sub_dict.update(af_metrics)

        if args.verbose:
            print(sub_dict)

        # Add completed design metrics to list
        sub_dicts.append(sub_dict)

        # Stop early if test_stop is set
        if args.test_stop is not None:
            if args.test_stop == task_idx:
                break

    # Load the sub_dicts into a pd DataFrame
    df = pd.DataFrame.from_dict(sub_dicts)
    print(df)

    # Update the df with Best-AF2-model metrics and Rosetta metrics then save results
    df = update_dataframe(args, df)

    return df


if __name__ == "__main__":
    main()
