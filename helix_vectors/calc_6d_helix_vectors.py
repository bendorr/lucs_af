#!/usr/bin/env python3
"""
Computes 6D helix direction vectors for protein structures.

Author: Ben Orr
Date: 9.9.22
Adapted from: XingJie Pan's get_6d_helix_vectors.py

This module computes 6-dimensional helix vectors for protein structures, where each
vector consists of 3D spatial coordinates (x, y, z) representing the helix centroid
position and 3D direction components (v_x, v_y, v_z) representing the helix direction.

The 6D representation combines:
    - 3D position: Average position of helix residue CA atoms (helix centroid)
    - 3D direction: Sum and normalization of backbone C->O (carbonyl) vectors

All coordinates are calculated with respect to a reference frame defined by the
structure's beta sheet residues or user-specified projection residues.

Usage
-----
Command line execution for batch processing LUCS designs:

    python calc_6d_helix_vectors.py \
        --ref_pdb reference.pdb \
        --ref_insertion_points insertion_points.json \
        --designs_df_path designs.csv \
        --output_dir ./output \
        --output_df_name results.csv \
        --num_lhls 2

Module import for programmatic use:

    from calc_6d_helix_vectors import project_a_helix_to_frame

    helix_6d = project_a_helix_to_frame(
        pose, (start_res, end_res),
        project_point, project_frame
    )

Parameters
----------
ref_pdb : str
    Path to reference structure (PDB file) for alignment
ref_insertion_points : str
    Path to insertion_points.json file for reference structure
designs_df_path : str
    Path to DataFrame (.csv) with design_id and lucs_location columns
output_dir : str
    Directory for output files (dataframes, helix_coords, sheet_coords)
output_df_name : str
    Name for output DataFrame with helix_coords columns

Returns
-------
The script generates several outputs:
    - CSV dataframes with helix coordinate columns
    - JSON files with helix coordinate dictionaries
    - JSON files with reference structure sheet coordinates
    - Optional: Aligned PDB files

Notes
-----
This script is designed for processing LUCS (Loop-Helix-Loop Unit Combinatorial
Sampling) protein designs and calculating their helix vectors relative
to a reference frame.
"""

import os
import json
import glob
import argparse
from typing import List, Tuple, Dict, Optional, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pyrosetta
from pyrosetta import rosetta
pyrosetta.init()

def xyz_to_np_array(xyz: Any) -> np.ndarray:
    """
    Convert a PyRosetta xyz vector to a numpy array.

    Parameters
    ----------
    xyz : rosetta.numeric.xyzVector
        PyRosetta xyz vector object with x, y, z attributes

    Returns
    -------
    np.ndarray
        1D numpy array of shape (3,) containing [x, y, z] coordinates
    """
    return np.array([xyz.x, xyz.y, xyz.z])

def project_point_to_frame(
    point: np.ndarray,
    vector: np.ndarray,
    project_point: Any,
    project_frame: List[np.ndarray]
) -> np.ndarray:
    """
    Project a point and its attached vector onto a reference frame.

    This function transforms both a spatial point and a direction vector into
    coordinates relative to a specified reference frame, creating a 6D representation.

    Parameters
    ----------
    point : np.ndarray
        3D coordinates of the point to project, shape (3,)
    vector : np.ndarray
        3D direction vector attached to the point, shape (3,)
    project_point : rosetta.numeric.xyzVector or np.ndarray
        Origin point of the reference frame
    project_frame : List[np.ndarray]
        List of three orthonormal basis vectors [x_axis, y_axis, z_axis],
        each with shape (3,)

    Returns
    -------
    np.ndarray
        6D array [x, y, z, v_x, v_y, v_z] where:
        - x, y, z: coordinates of the point in the reference frame
        - v_x, v_y, v_z: components of the vector in the reference frame

    Notes
    -----
    The projection transforms coordinates from the global frame to the local
    reference frame defined by project_point and project_frame.
    """
    # Project the point
    diff = point - project_point

    x = np.dot(diff, project_frame[0])
    y = np.dot(diff, project_frame[1])
    z = np.dot(diff, project_frame[2])

    # Project the attached vector
    v_x = np.dot(vector, project_frame[0])
    v_y = np.dot(vector, project_frame[1])
    v_z = np.dot(vector, project_frame[2])

    return np.array([x, y, z, v_x, v_y, v_z])

def get_project_point_and_frame(
    args: argparse.Namespace,
    pose: Any
) -> Tuple[Any, List[np.ndarray]]:
    """
    Get the projection point and reference frame for a given pose.

    Constructs an orthonormal reference frame using two projection residues.
    The frame is defined by:
    - Origin: CA atom of the first projection residue
    - Z-axis: CA->CB direction of the first projection residue
    - Y-axis: Direction from first to second projection residue CA
    - X-axis: Orthogonal to Y and Z (via cross product)

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing projection_residues string
    pose : rosetta.core.pose.Pose
        PyRosetta pose object of the protein structure

    Returns
    -------
    project_point : rosetta.numeric.xyzVector
        Origin point of the reference frame (CA of first projection residue)
    project_frame : List[np.ndarray]
        List of three orthonormal basis vectors [x, y, z], each shape (3,)

    Notes
    -----
    If the first projection residue is GLY (lacking CB), it is temporarily
    mutated to ALA to enable CB-based frame construction.
    """
    projection_residues = [int(n) for n in args.projection_residues.split(',')]

    project_point = pose.residue(projection_residues[0]).xyz('CA')

    try:
        z = xyz_to_np_array(pose.residue(projection_residues[0]).xyz('CA') - pose.residue(projection_residues[0]).xyz('CB'))
    except:
        # If residue 6 is GLY, then mutate it to Ala and use Ala's Res 6 CB atom.
        pyrosetta.toolbox.mutants.mutate_residue(pose, projection_residues[0], 'A', pack_radius=0.0, pack_scorefxn=None)
        z = xyz_to_np_array(pose.residue(projection_residues[0]).xyz('CA') - pose.residue(projection_residues[0]).xyz('CB'))

    y = xyz_to_np_array(pose.residue(projection_residues[1]).xyz('CA') - pose.residue(projection_residues[0]).xyz('CA'))
    x = np.cross(y, z)
    y = np.cross(z, x)

    project_frame = [x / np.linalg.norm(x), y / np.linalg.norm(y), z / np.linalg.norm(z)]

    return project_point, project_frame


def project_a_helix_to_frame(
    pose: Any,
    helix: Tuple[int, int],
    project_point: Any,
    project_frame: List[np.ndarray]
) -> np.ndarray:
    """
    Project a helix to a reference frame and return 6D helix coordinates.

    The helix is represented by its centroid position and direction vector.
    Centroid is calculated as the average CA atom position, and direction
    is the normalized sum of backbone C->O (carbonyl) vectors.

    Parameters
    ----------
    pose : rosetta.core.pose.Pose
        PyRosetta pose object of the protein structure
    helix : Tuple[int, int]
        Helix definition as (start_residue, stop_residue) inclusive
    project_point : rosetta.numeric.xyzVector
        Origin point of the reference frame
    project_frame : List[np.ndarray]
        List of three orthonormal basis vectors [x, y, z], each shape (3,)

    Returns
    -------
    np.ndarray
        6D helix coordinates [x, y, z, v_x, v_y, v_z] where:
        - x, y, z: helix centroid position in reference frame
        - v_x, v_y, v_z: helix direction in reference frame

    Notes
    -----
    Take the average position of helix residue CA atoms to
    calculate the helix centroids, and sum and normalize the
    C->O (backbone carbonyl) vectors to get the helix directions.
    """
    helix_cas = [xyz_to_np_array(pose.residue(i).xyz('CA')) for i in range(helix[0], helix[1] + 1)]
    helix_dirs = [xyz_to_np_array(pose.residue(i).xyz('O') - pose.residue(i).xyz('C'))
            for i in range(helix[0], min(helix[1] + 1, pose.size()))]

    center = sum(helix_cas) / len(helix_cas)
    center_dir = sum(helix_dirs)
    center_dir = center_dir / np.linalg.norm(center_dir)

    return project_point_to_frame(center, center_dir, project_point, project_frame)

def convert_pdb_to_pose_resnums(pose: Any, pdb_res_list: List[int]) -> List[int]:
    """
    Convert a list of PDB residue numbers to PyRosetta pose numbering.

    Parameters
    ----------
    pose : rosetta.core.pose.Pose
        PyRosetta pose object with PDB information
    pdb_res_list : List[int]
        List of residue numbers in PDB numbering

    Returns
    -------
    List[int]
        List of residue numbers in PyRosetta pose numbering

    Notes
    -----
    Ex. pose.pdb_info().pdb2pose('A', 601) # -> int
    To convert from pose to PDB: pose.pdb_info().pose2pdb(24) # -> '47 A ' Res num, chain ID
    Residues that convert to 0 (not found) are excluded from the output list.
    """
    pose_res_list = []
    for pdb_res_num in pdb_res_list:

        chain = pose.pdb_info().pose2pdb(1).split()[1].strip()

        # Pose to PDB
        pose_res_num = int(pose.pdb_info().pdb2pose(chain, pdb_res_num))

        if pose_res_num == 0:
            print(f'PDB res {pdb_res_num} converted to {pose_res_num}. Excluding this residue.')
        else:
            pose_res_list.append(pose_res_num)

    return pose_res_list

def get_all_helix_coords_for_one_design_by_ins_pts(
    args: argparse.Namespace,
    pose: Any,
    pdb_file: str,
    insertion_points: Tuple[int, int],
    project_point: Any,
    project_frame: List[np.ndarray],
    lhl_id: Optional[int] = None,
    is_ref_struct: bool = False
) -> List[np.ndarray]:
    """
    Get all coordinates for remodeled helices for one design.

    Uses DSSP to identify helical residues within the insertion points region,
    or assumes all insertion point residues are helical based on arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    pose : rosetta.core.pose.Pose
        PyRosetta pose object of the protein structure
    pdb_file : str
        Path to the PDB file (currently unused but kept for compatibility)
    insertion_points : Tuple[int, int]
        Start and stop residues of the helix (inclusive)
    project_point : rosetta.numeric.xyzVector
        Origin point of the reference frame
    project_frame : List[np.ndarray]
        List of three orthonormal basis vectors
    lhl_id : Optional[int], default=None
        Loop-Helix-Loop identifier
    is_ref_struct : bool, default=False
        Whether this is a reference structure

    Returns
    -------
    List[np.ndarray]
        List containing one 6D helix coordinate array

    Notes
    -----
    The function uses DSSP to identify helical regions unless ins_pts_all_helical
    is set. If fewer than 4 helical residues are found, assumes the helix is in
    the central region (5 residues from each end of insertion points).
    """
    dssp_str = rosetta.core.scoring.dssp.Dssp(pose).get_dssp_secstruct()

    if args.verbose:
        print('DSSP String:')
        print(dssp_str)

    helix_coords = []

    # At start of loop, set start to stop_idx and stop to start_idx, then
    # update these values over loop iterations.
    h_start = insertion_points[1]
    h_stop = insertion_points[0]

    print(f'Insertion Points: {insertion_points}')

    if args.ins_pts_all_helical and not is_ref_struct:
        h_start = insertion_points[0]
        h_stop = insertion_points[1]
    else:
        for i in range(insertion_points[0], insertion_points[1] + 1):
            if dssp_str[i - 1] == 'H':
                if i < h_start:
                    h_start = i
                if i > h_stop:
                    h_stop = i

            # if h_start > h_stop: continue

        # If fewer than 4 helical residues were found, set the central residues (-5 from each end of the LHL) to be the helix,
        # as no LUCS loops are longer than 5 residues.
        if h_stop - h_start < 4:
            h_start = insertion_points[0] + 5
            h_stop = insertion_points[1] - 5
            if args.verbose:
                print('Assuming helix is ins_pts[5:-5]:')
                print(h_start)
                print(h_stop)

    if args.verbose:
        print('Helix Start and Stop:')
        print(h_start)
        print(h_stop)

    if args.convert_to_pyrosetta_nums:
        pose_reslist = convert_pdb_to_pose_resnums(pose, [h_start, h_stop])

        h_start = pose_reslist[0]
        h_stop = pose_reslist[1]

    helix_coords.append(project_a_helix_to_frame(pose, (h_start, h_stop), project_point, project_frame))

    return helix_coords

def get_all_helix_coords_for_data_set(
    data_path: str,
    project_point: Any,
    project_frame: List[np.ndarray],
    lhl_id: Optional[int] = None
) -> List[np.ndarray]:
    """
    Get all coordinates for remodeled helices in a data set.

    Parameters
    ----------
    data_path : str
        Path to directory containing PDB files and insertion point JSON files
    project_point : rosetta.numeric.xyzVector
        Origin point of the reference frame
    project_frame : List[np.ndarray]
        List of three orthonormal basis vectors
    lhl_id : Optional[int], default=None
        Loop-Helix-Loop identifier

    Returns
    -------
    List[np.ndarray]
        List of 6D helix coordinate arrays for all designs in the dataset

    Notes
    -----
    Expects files named 'model_###.pdb' and 'insertion_points_###.json'.
    """
    # Get all pdb files and insertion files
    pdb_files = []
    insertion_files = []
    pdb_ids = []

    for f in sorted(os.listdir(data_path)):
        if f.endswith('.pdb.gz') or f.endswith('.pdb'):
            pdb_id = f.split('.')[0][6:]  # gets rid of 'model_' in 'model_###.pdb'

            insertion_file = os.path.join(data_path, 'insertion_points_{0}.json'.format(pdb_id))

            if os.path.exists(insertion_file):
                insertion_files.append(insertion_file)
                pdb_files.append(os.path.join(data_path, f))
                pdb_ids.append(pdb_id)

    # Get the helix coords
    helix_coords = []

    for i in range(len(pdb_files)):
        helix_coords += get_all_helix_coords_for_one_design(pdb_files[i], insertion_files[i], project_point, project_frame, lhl_id=lhl_id)

    return helix_coords

def get_all_helix_dicts_for_data_set(
    data_path: str,
    project_point: Any,
    project_frame: List[np.ndarray],
    lhl_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all coordinates for remodeled helices in a data set as dictionaries.

    Parameters
    ----------
    data_path : str
        Path to directory containing PDB files and insertion point JSON files
    project_point : rosetta.numeric.xyzVector
        Origin point of the reference frame
    project_frame : List[np.ndarray]
        List of three orthonormal basis vectors
    lhl_id : Optional[int], default=None
        Loop-Helix-Loop identifier

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries, each with keys 'design_id' and 'helix{lhl_id}_coords'

    Notes
    -----
    helix_dicts is a list of dictionaries, each with keys=design_id and helix_coords
    """
    # Get all pdb files and insertion files
    pdb_files = []
    insertion_files = []
    pdb_ids = []

    for f in sorted(os.listdir(data_path)):
        if f.endswith('.pdb.gz') or f.endswith('.pdb'):
            pdb_id = f.split('.')[0][6:]  # gets rid of 'model_' in 'model_###.pdb'

            insertion_file = os.path.join(data_path, 'insertion_points_{0}.json'.format(pdb_id))

            if os.path.exists(insertion_file):
                insertion_files.append(insertion_file)
                pdb_files.append(os.path.join(data_path, f))
                pdb_ids.append(pdb_id)

    # Get the helix coords
    # helix_dicts is a list of dictionaries, each with keys=design_id and helix_coords
    helix_dicts = []

    for i in range(len(pdb_files)):
        design_dict = {}
        design_dict['design_id'] = pdb_ids[i]
        design_dict[f'helix{lhl_id}_coords'] = \
            get_all_helix_coords_for_one_design(pdb_files[i], \
                insertion_files[i], project_point, project_frame, lhl_id=lhl_id)
        helix_dicts.append(design_dict)

    return helix_dicts

def get_ins_pts_from_design_info(
    design_info_path: str,
    lhl_id: Optional[int] = None
) -> Union[List[Tuple[int, int]], Tuple[int, int]]:
    """
    Infer the insertion points for a LUCS design from its backbone remodeled residues.

    Parameters
    ----------
    design_info_path : str
        Path to design_info.json file containing bb_remodeled_residues
    lhl_id : Optional[int], default=None
        Loop-Helix-Loop identifier. If None, returns all insertion points.

    Returns
    -------
    Union[List[Tuple[int, int]], Tuple[int, int]]
        If lhl_id is None: List of (start, stop) tuples for all contiguous regions
        If lhl_id is specified: Single (start, stop) tuple for that specific LHL

    Notes
    -----
    Identifies contiguous stretches of backbone remodeled residues and returns
    their start and stop positions.
    """
    with open(design_info_path, 'r') as f:
        design_info = json.load(f)

    bb_rem_res = design_info['bb_remodeled_residues']

    sublists = []
    contig = [bb_rem_res[0]]

    for i in range(1, len(bb_rem_res)):
        if bb_rem_res[i] == bb_rem_res[i - 1] + 1:
            contig.append(bb_rem_res[i])
        else:
            sublists.append(contig)
            contig = [bb_rem_res[i]]

    sublists.append(contig)

    if lhl_id is None:
        return [(contig[0], contig[-1]) for contig in sublists]
    else:
        return (sublists[lhl_id][0], sublists[lhl_id][-1])

def get_ins_pts(ins_pts_path: str) -> List[Tuple[int, int]]:
    """
    Get insertion points from a JSON file.

    Parameters
    ----------
    ins_pts_path : str
        Path to JSON file containing insertion points with 'start' and 'stop' keys

    Returns
    -------
    List[Tuple[int, int]]
        List of (start, stop) tuples for each insertion point
    """
    with open(ins_pts_path, 'r') as f:
        ins_pts = json.load(f)

    ret = []
    for ip in ins_pts:
        ret.append((ip['start'], ip['stop']))
    return ret

def parse_termanal_scores(score_file_path: str) -> List[Tuple[str, float]]:
    """
    Parse TERMANAL score file and extract residue-score pairs.

    Parameters
    ----------
    score_file_path : str
        Path to TERMANAL score file (.dsc50, .abd50, or .ssc50)

    Returns
    -------
    List[Tuple[str, float]]
        List of (residue_identifier, score) tuples
    """
    list_of_tuples = []
    with open(score_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            new_tuple = (line.split()[0], float(line.split()[1]))
            list_of_tuples.append(new_tuple)
    return list_of_tuples


def get_termanal_scores_for_res(
    termanal_scores: List[Tuple[str, float]],
    res_list: List[int]
) -> List[float]:
    """
    Extract TERMANAL scores for specific residues.

    Parameters
    ----------
    termanal_scores : List[Tuple[str, float]]
        List of (residue_identifier, score) tuples from TERMANAL output
    res_list : List[int]
        List of residue indices to extract scores for

    Returns
    -------
    List[float]
        List of scores for the specified residues

    Notes
    -----
    Get TERMANAL scores for res_idxs in res_list
    """
    res_scores = []
    for res_idx in res_list:
        for res_chn_idx, score in termanal_scores:
            if res_chn_idx.split(',')[1].rstrip("'") == str(res_idx):
                res_scores.append(score)
    return res_scores

def get_termanal_scores(
    termanal_folder: str,
    design_id: Union[str, int],
    ins_pts: Tuple[int, int]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return the average TERMANAL scores for the residues within the ins_pts range.

    Parameters
    ----------
    termanal_folder : str
        Path to folder containing TERMANAL output files
    design_id : Union[str, int]
        Design identifier
    ins_pts : Tuple[int, int]
        Insertion points (start, stop) defining the residue range

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[float]]
        Average design score, abundance score, and structure score.
        Returns (None, None, None) if scores not found.

    Notes
    -----
    Some termanal outputs are written in separate subdirs, named by
    design_id, in which case the termanal_output_path is termanal_folder+subdir
    """
    subdir = str(design_id)

    # Some termanal outputs are written in separate subdirs, named by
    # design_id, in which case the termanal_output_path is termanal_folder+subdir
    if os.path.exists(os.path.join(termanal_folder, subdir + '.dsc50')):
        termanal_output_path = termanal_folder
    elif os.path.exists(os.path.join(termanal_folder, subdir, subdir + '.dsc50')):
        termanal_output_path = os.path.join(termanal_folder, subdir)
    else:
        print('TERMANAL Scores not found for design %d. Skipping.' % design_id)
        return None, None, None

    design_scores = parse_termanal_scores(os.path.join(termanal_output_path, subdir + '.dsc50'))
    abundance_scores = parse_termanal_scores(os.path.join(termanal_output_path, subdir + '.abd50'))
    structure_scores = parse_termanal_scores(os.path.join(termanal_output_path, subdir + '.ssc50'))

    design_scores = get_termanal_scores_for_res(design_scores, list(range(ins_pts[0], ins_pts[1])))
    abundance_scores = get_termanal_scores_for_res(abundance_scores, list(range(ins_pts[0], ins_pts[1])))
    structure_scores = get_termanal_scores_for_res(structure_scores, list(range(ins_pts[0], ins_pts[1])))

    return np.mean(design_scores), np.mean(abundance_scores), np.mean(structure_scores)

def get_design_to_ref_aligned_residues(
    args: argparse.Namespace,
    design_path: str
) -> Optional[List[List[List[int]]]]:
    """
    Get alignment residue lists for design and reference structures.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing ref_align_residues_file
    design_path : str
        Path to design PDB file

    Returns
    -------
    Optional[List[List[List[int]]]]
        List of two lists: [reference_residues, design_residues], where each
        contains lists of residue indices for alignment. Returns None if files not found.

    Notes
    -----
    Look in the design's folder for a {design_id}_align_residues.json file
    Return a list of two lists, containing 1) the starting structure's and 2) the design's residues on which to align
    """
    # Look in the design's folder for a {design_id}_align_residues.json file
    design_align_res_file = design_path.replace('.pdb', '_align_residues.json')

    if not os.path.exists(design_align_res_file):
        print(f'{design_align_res_file} not found. Not using align_residues.json file for structure alignment.')
        return None

    if not os.path.exists(args.ref_align_residues_file):
        print(f'{args.ref_align_residues_file} not found. Not using align_residues.json file for structure alignment.')
        return None

    with open(design_align_res_file, 'r') as f:
        design_align_res = json.load(f)

    with open(args.ref_align_residues_file, 'r') as f:
        ref_align_res = json.load(f)

    # Return a list of two lists, containing 1) the starting structure's and 2) the design's residues on which to align
    align_residues = [[], []]
    for ref_res, des_res in zip(ref_align_res, design_align_res):
        if len(ref_res) > len(des_res):
            start_idx = (len(ref_res) - len(des_res)) // 2
            ref_res = ref_res[start_idx:start_idx + len(des_res)]
        elif len(des_res) > len(ref_res):
            start_idx = (len(des_res) - len(ref_res)) // 2
            des_res = des_res[start_idx:start_idx + len(ref_res)]

        align_residues[0].append(ref_res)
        align_residues[1].append(des_res)

    return align_residues

def xyzV_to_np_array(xyz: Any) -> np.ndarray:
    """
    Convert a PyRosetta xyzVector to a numpy array.

    Parameters
    ----------
    xyz : rosetta.numeric.xyzVector
        PyRosetta xyz vector object with x, y, z attributes

    Returns
    -------
    np.ndarray
        1D numpy array of shape (3,) containing [x, y, z] coordinates
    """
    return np.array([xyz.x, xyz.y, xyz.z])

def get_backbone_points(
    pose: Any,
    residues: List[int],
    atom_types: List[str] = ['N', 'CA', 'C']
) -> List[np.ndarray]:
    """
    Get backbone points for residues in a pose.

    Parameters
    ----------
    pose : rosetta.core.pose.Pose
        PyRosetta pose object
    residues : List[int]
        List of residue indices
    atom_types : List[str], default=['N', 'CA', 'C']
        Backbone atom types to extract

    Returns
    -------
    List[np.ndarray]
        List of 3D coordinates for requested backbone atoms.
        Missing atoms are represented as (-9999, -9999, -9999).
    """
    points = []

    for res in residues:
        for atom in atom_types:
            try:
                points.append(xyzV_to_np_array(pose.residue(res).xyz(atom)))
            except:
                print(f'Residue {res} not found in pose. Skipping alignment on this residue')
                points.append((-9999, -9999, -9999))

    return points

def get_superimpose_transformation(
    P1: List[np.ndarray],
    P2: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the superimpose transformation that transforms a list of points P1 to another list of points P2.

    Uses Kabsch algorithm for optimal rotation matrix calculation.

    Parameters
    ----------
    P1 : List[np.ndarray]
        Source point set
    P2 : List[np.ndarray]
        Target point set

    Returns
    -------
    M : np.ndarray
        Rotation matrix (3x3)
    t : np.ndarray
        Translation vector (3,)

    Raises
    ------
    Exception
        If P1 and P2 have different number of points
    """
    if len(P1) != len(P2):
        raise Exception("Sets to be superimposed must have same number of points.")

    com1 = np.mean(P1, axis=0)
    com2 = np.mean(P2, axis=0)

    R = np.dot(np.transpose(np.array(P1) - com1), np.array(P2) - com2)
    V, S, W = np.linalg.svd(R)

    if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
        V[:, -1] = -V[:, -1]

    M = np.transpose(np.array(np.dot(V, W)))

    return M, com2 - np.dot(M, com1)


def np_array_to_xyzV(a: np.ndarray) -> Any:
    """
    Convert numpy array to PyRosetta xyzVector.

    Parameters
    ----------
    a : np.ndarray
        1D array of shape (3,) with [x, y, z] coordinates

    Returns
    -------
    rosetta.numeric.xyzVector_double_t
        PyRosetta xyz vector
    """
    return rosetta.numeric.xyzVector_double_t(a[0], a[1], a[2])


def np_array_to_xyzM(a: np.ndarray) -> Any:
    """
    Convert numpy array to PyRosetta xyzMatrix.

    Parameters
    ----------
    a : np.ndarray
        2D array of shape (3, 3) representing a rotation matrix

    Returns
    -------
    rosetta.numeric.xyzMatrix_double_t
        PyRosetta xyz matrix
    """
    return rosetta.numeric.xyzMatrix_double_t.rows(
        a[0][0], a[0][1], a[0][2],
        a[1][0], a[1][1], a[1][2],
        a[2][0], a[2][1], a[2][2])

def superimpose_poses_by_residues(
    pose_source: Any,
    residues_source: List[int],
    pose_target: Any,
    residues_target: List[int],
    atom_types: List[str] = ['N', 'CA', 'C']
) -> Tuple[Any, Any]:
    """
    Superimpose residues in a source pose onto residues in a target pose.

    Only backbone atoms are used for the superimposition.

    Parameters
    ----------
    pose_source : rosetta.core.pose.Pose
        Source pose to be transformed
    residues_source : List[int]
        List of residue indices in source pose for alignment
    pose_target : rosetta.core.pose.Pose
        Target pose (remains unchanged)
    residues_target : List[int]
        List of residue indices in target pose for alignment
    atom_types : List[str], default=['N', 'CA', 'C']
        Backbone atom types to use for alignment

    Returns
    -------
    Tuple[rosetta.core.pose.Pose, rosetta.core.pose.Pose]
        Transformed source pose and unchanged target pose

    Notes
    -----
    If these two residue lists are of different
    lengths, trim the N-terminal residues of the longer sequence.
    """
    # If these two residue lists are of different
    # lengths, trim the N-terminal residues of the longer sequence.
    if len(residues_source) != len(residues_target):
        min_len = min(len(residues_source), len(residues_target))
        residues_source = residues_source[-1 * min_len:]
        residues_target = residues_target[-1 * min_len:]

    # Get the points to be superimposed
    points_source_unfiltered = get_backbone_points(pose_source, residues_source, atom_types=atom_types)
    points_target_unfiltered = get_backbone_points(pose_target, residues_target, atom_types=atom_types)

    points_source, points_target = [], []
    for ps, pt in zip(points_source_unfiltered, points_target_unfiltered):
        if ps[0] == -9999 or pt[0] == -9999:
            continue
        else:
            points_source.append(ps)
            points_target.append(pt)

    points_source = np.array(points_source)
    points_target = np.array(points_target)

    # Get the rigid body transformation
    M, t = get_superimpose_transformation(points_source, points_target)

    # Transform the source pose
    pose_source.apply_transform_Rx_plus_v(np_array_to_xyzM(M),
            np_array_to_xyzV(t))

    return pose_source, pose_target

def get_all_helix_dicts_for_dataframe(
    args: argparse.Namespace,
    df: pd.DataFrame,
    ref_pdb: str,
    termanal_folder: Optional[str] = None,
    lhl_id: Optional[int] = None,
    end_of_df_col: str = ''
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Any, Any]:
    """
    For each design (row) in the dataframe, calculate a new reference frame
    and helix coordinates for the design, then return a list of dictionaries
    with these helix coordinates, design IDs, and other LUCS-AF metrics.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    df : pd.DataFrame
        DataFrame with design information (design_id, lucs_location, etc.)
    ref_pdb : str
        Path to reference PDB file
    termanal_folder : Optional[str], default=None
        Path to TERMANAL scores folder (currently unused)
    lhl_id : Optional[int], default=None
        Loop-Helix-Loop identifier
    end_of_df_col : str, default=''
        Unused parameter kept for compatibility

    Returns
    -------
    Tuple[List[Dict[str, Any]], pd.DataFrame, Any, Any]
        helix_dicts: List of dictionaries with helix coordinates and design IDs
        df: Updated dataframe with helix_coords columns
        pose: Last processed design pose
        ref_pose: Reference structure pose
    """
    helix_dicts = []

    ref_pose = rosetta.core.import_pose.pose_from_file(ref_pdb)

    project_point, project_frame = get_project_point_and_frame(args, ref_pose)

    ref_ins_pts = get_ins_pts(args.ref_insertion_points)
    ref_bb_rem_res = [int(i) for ins_pts in ref_ins_pts for i in range(ins_pts[0], ins_pts[1] + 1)]
    ref_non_bb_rem_res = [int(i) for i in range(1, ref_pose.size() + 1) if i not in ref_bb_rem_res]

    if args.verbose:
        print(f"Reference insertion points: {ref_ins_pts}")
        print(f"Reference non-backbone remodeled residues: {ref_non_bb_rem_res}")

    task_idx = 0
    for idx, row in df.iterrows():

        task_idx += 1
        if task_idx % args.num_tasks != (args.task_id-1):
            continue

        print(f'Processing Design {row["design_id"]}')
        design_dict = {}
        design_dict['design_id'] = row['design_id']

        if args.use_rank1_af:
            design_folder = row['af_location']
            design_info_folder = row['lucs_location']
            design_pdbs = glob.glob(os.path.join(design_folder, f"*{row['af_design_id']}{args.pdb_suffix}*rank_1*.pdb"))
            design_path = design_pdbs[0] if design_pdbs else None
        elif args.af_dir != '':
            design_folder = args.af_dir
            design_info_folder = row['lucs_location']
            design_path = glob.glob(os.path.join(design_folder, f"*{row['af_design_id']}{args.pdb_suffix}*rank_1*.pdb"))[0]
        else:
            design_folder = row['lucs_location']
            design_info_folder = row['lucs_location']

            if args.use_af_designid:
                design_path = os.path.join(design_folder, f"{row['af_design_id']}{args.pdb_suffix}.pdb")
            else:
                design_path = os.path.join(design_folder, f"{row['design_id']}{args.pdb_suffix}.pdb")

        # Some LUCS dirs use the following format:
        # design_path = os.path.join(design_folder, 'unzipped_design.pdb')
        design_info_path = None
        for design_info_name in ['design_info.json', f"{row['design_id']}_design_info.json"]:
            if os.path.exists(os.path.join(design_info_folder, design_info_name)):
                design_info_path = os.path.join(design_info_folder, design_info_name)
                break 
        if design_info_path is None:
            print(f"Design info file not found for design {row['design_id']}. Skipping this design.")
            continue

        # Recalculate the reference frame for the current design.
        pose = rosetta.core.import_pose.pose_from_file(design_path)

        # Use align_residues_files to align designs to the reference structure
        if args.use_align_residues_file:
            pdb_ref_align_res, pdb_design_align_res = get_design_to_ref_aligned_residues(args, design_path)
            ref_align_res = []
            for res_list in pdb_ref_align_res:
                ref_align_res.append(convert_pdb_to_pose_resnums(ref_pose, res_list))
            design_align_res = []
            for res_list in pdb_design_align_res:
                design_align_res.append(convert_pdb_to_pose_resnums(pose, res_list))
            pose, ref_pose = superimpose_poses_by_residues(
				pose, 
				[i for lst in design_align_res for i in lst], 
				ref_pose, 
				[i for lst in ref_align_res for i in lst], 
				atom_types=['N', 'CA', 'C'])
            
        # Use non_bb_rem residues to align designs to the reference structure
        if args.align_by_non_bb_rem:
            bb_rem_res = [int(i) for i in row['bb_remodeled_residues'].split()]
            non_bb_rem_res = [i for i in range(1, pose.size() + 1) if i not in bb_rem_res]
            pose, ref_pose = superimpose_poses_by_residues(
				pose, 
				non_bb_rem_res, 
				ref_pose, 
				ref_non_bb_rem_res, 
				atom_types=['N', 'CA', 'C'])
             
            
        if args.aligned_pdb_outdir != None:
            os.makedirs(args.aligned_pdb_outdir, exist_ok=True)
            aligned_pdb_outpath = os.path.join(args.aligned_pdb_outdir, design_path.split('/')[-1].replace('.pdb', '_aligned_to_ref.pdb'))
            pose.dump_pdb(aligned_pdb_outpath)
            if args.verbose:
                print(f"Saved aligned PDB to Reference Structure: {aligned_pdb_outpath}")

        ins_pts = get_ins_pts_from_design_info(design_info_path, lhl_id)

        design_dict[f'helix{lhl_id}_coords'] = \
            get_all_helix_coords_for_one_design_by_ins_pts(args, pose, design_path, \
                ins_pts, project_point, project_frame, lhl_id=lhl_id, is_ref_struct=False)

        if args.verbose:
            print(f'Helix {lhl_id} Coords:')
            print(design_dict[f'helix{lhl_id}_coords'])
            print(f"design_dict[helix{lhl_id}_coords][0]")
            print(design_dict[f'helix{lhl_id}_coords'][0])
            print('String list:')
            print([str(val) for val in design_dict[f'helix{lhl_id}_coords'][0]])
            print('Joined list:')
            print(' '.join([str(val) for val in design_dict[f'helix{lhl_id}_coords'][0]]))

        df.loc[idx, f'helix{lhl_id}_coords'] = ' '.join([str(val) for val in design_dict[f'helix{lhl_id}_coords'][0]])

        helix_dicts.append(design_dict)

    return helix_dicts, df, pose, ref_pose

def get_helix_coords_from_helix_dicts(
    helix_dicts: List[Dict[str, Any]],
    lhl_id: int
) -> List[np.ndarray]:
    """
    Return a list of helix coordinate arrays extracted from helix dictionaries.

    Parameters
    ----------
    helix_dicts : List[Dict[str, Any]]
        List of dictionaries containing helix coordinates
    lhl_id : int
        Loop-Helix-Loop identifier

    Returns
    -------
    List[np.ndarray]
        Flattened list of helix coordinate arrays
    """
    helix_coords = []
    for design_dict in helix_dicts:
        helix_coords += design_dict[f'helix{lhl_id}_coords']
    return helix_coords


def dump_helix_coords(helix_coords: List[np.ndarray], file_name: str) -> None:
    """
    Dump helix coordinates to a JSON file.

    Parameters
    ----------
    helix_coords : List[np.ndarray]
        List of 6D helix coordinate arrays
    file_name : str
        Output JSON file path
    """
    h_coords_serial = [list(c) for c in helix_coords]

    with open(file_name, 'w') as f:
        json.dump(h_coords_serial, f)


def dump_helix_dicts(
    helix_dicts: List[Dict[str, Any]],
    file_name: str,
    lhl_id: int
) -> None:
    """
    Dump a list of helix dictionaries to a JSON file.

    Parameters
    ----------
    helix_dicts : List[Dict[str, Any]]
        List of dictionaries containing design_id and helix coordinates
    file_name : str
        Output JSON file path
    lhl_id : int
        Loop-Helix-Loop identifier
    """
    for design_dict in helix_dicts:
        design_dict[f'helix{lhl_id}_coords'] = [list(c) for c in design_dict[f'helix{lhl_id}_coords']]
    with open(file_name, 'w') as f:
        json.dump(helix_dicts, f)


def load_helix_coords(file_name: str) -> List[np.ndarray]:
    """
    Load helix coordinates from a JSON file.

    Parameters
    ----------
    file_name : str
        Input JSON file path

    Returns
    -------
    List[np.ndarray]
        List of 6D helix coordinate arrays
    """
    with open(file_name, 'r') as f:
        h_coords_serial = json.load(f)

    return [np.array(c) for c in h_coords_serial]


def plot_helices(helix_coords: List[np.ndarray], axis3d: Any) -> None:
    """
    Plot the helix coordinates in 3D as vectors.

    Parameters
    ----------
    helix_coords : List[np.ndarray]
        List of 6D helix coordinates [x, y, z, v_x, v_y, v_z]
    axis3d : mpl_toolkits.mplot3d.Axes3D
        Matplotlib 3D axis object for plotting
    """
    X = [h[0] for h in helix_coords]
    Y = [h[1] for h in helix_coords]
    Z = [h[2] for h in helix_coords]
    U = [h[3] for h in helix_coords]
    V = [h[4] for h in helix_coords]
    W = [h[5] for h in helix_coords]

    axis3d.quiver(X, Y, Z, U, V, W, length=1)

def get_sheet_coords(
    pose: Any,
    sheet_residues: List[List[int]],
    project_point: Any,
    project_frame: List[np.ndarray]
) -> List[List[np.ndarray]]:
    """
    Get all coordinates for beta sheet residues.

    Parameters
    ----------
    pose : rosetta.core.pose.Pose
        PyRosetta pose object
    sheet_residues : List[List[int]]
        List of lists, each sublist contains PDB-numbered residues for a single beta strand
    project_point : rosetta.numeric.xyzVector
        Origin point of the reference frame
    project_frame : List[np.ndarray]
        List of three orthonormal basis vectors

    Returns
    -------
    List[List[np.ndarray]]
        Projected coordinates for each strand, each strand is a list of 3D points
    """
    sheet_coords = []
    for strand in sheet_residues:
        strand_coords = []
        for pdb_resnum in strand:
            pyr_resnum = pose.pdb_info().pdb2pose('A', pdb_resnum)
            strand_coords.append(pose.residue(pyr_resnum).xyz('N'))
            strand_coords.append(pose.residue(pyr_resnum).xyz('CA'))
            strand_coords.append(pose.residue(pyr_resnum).xyz('C'))
        sheet_coords.append(strand_coords)

    return project_strand_points_to_frame(sheet_coords, project_point, project_frame)


def project_strand_points_to_frame(
    sheet_coords: List[List[Any]],
    project_point: Any,
    project_frame: List[np.ndarray]
) -> List[List[np.ndarray]]:
    """
    Project each coordinate in sheet_coords to the reference point and frame.

    Parameters
    ----------
    sheet_coords : List[List[Any]]
        List of lists, each sublist contains coordinates (xyzVectors) of a single beta strand
    project_point : rosetta.numeric.xyzVector
        Origin point of the reference frame
    project_frame : List[np.ndarray]
        List of three orthonormal basis vectors

    Returns
    -------
    List[List[np.ndarray]]
        Projected coordinates for each strand in the reference frame

    Notes
    -----
    sheet_coords is a list of lists, each sublist contains coordinates (lists
    of length 3) of a single beta strand.
    """
    projected_sheet_coords = []
    for strand in sheet_coords:
        projected_strand_coords = []
        for point in strand:
            diff = point - project_point
            x = np.dot(diff, project_frame[0])
            y = np.dot(diff, project_frame[1])
            z = np.dot(diff, project_frame[2])
            projected_strand_coords.append(np.array([x, y, z]))
        projected_sheet_coords.append(projected_strand_coords)

    return projected_sheet_coords


def dump_sheet_coords(sheet_coords: List[List[np.ndarray]], file_name: str) -> None:
    """
    Dump sheet coordinates to JSON files. Separate JSON file for each strand.

    Parameters
    ----------
    sheet_coords : List[List[np.ndarray]]
        List of strands, each strand is a list of 3D coordinate arrays
    file_name : str
        Base file name, will be modified to include strand index
    """
    for i in range(len(sheet_coords)):
        strand = sheet_coords[i]
        s_coords_serial = [list(c) for c in strand]

        fname_final = file_name.replace('.json', f'_strand{i}.json')

        with open(fname_final, 'w') as f:
            json.dump(s_coords_serial, f)


def get_sheet_residues(ref_pdb: str) -> List[List[int]]:
    """
    Use PyRosetta DSSP to find continuous beta strands in the reference structure.

    Parameters
    ----------
    ref_pdb : str
        Path to reference PDB file

    Returns
    -------
    List[List[int]]
        List of beta strand residue ranges, each strand is a list of residue indices
    """
    # Use pyrosetta DSSP to find continuous beta strands in the reference structure
    sheet_residues = []
    ref_pose = rosetta.core.import_pose.pose_from_file(ref_pdb)
    dssp = rosetta.core.scoring.dssp.Dssp(ref_pose)
    dssp_str = dssp.get_dssp_secstruct()
    sheet_start = None
    for i in range(1, len(dssp_str)):
        if dssp_str[i] == 'E':
            if sheet_start is None:
                sheet_start = i
        else:
            if sheet_start is not None:
                sheet_residues.append(list(range(sheet_start, i)))
                sheet_start = None
    return sheet_residues 


if __name__ == '__main__':

    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Calculate the 6D helix vectors for LUCS designs' reshaped helices.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Inputs
    parser.add_argument('--ref_pdb', type=str, required=True,
            help='Path to a Reference Structure (PDB file) onto which all designs are aligned.')
    parser.add_argument('--ref_insertion_points', type=str, required=True,
            help='Path to an insertion_points.json file for the reference structure.')
    parser.add_argument('--designs_df_path', type=str, required=True,
            help='Path to a DataFrame (.csv) file containing a design_id column and a \
            lucs_location column.')
    parser.add_argument('--af_dir', type=str, required=False, default='',
            help='Path to the directory containing AF2/3 predictions (PDB files) for design_ids in designs_df.\
                Rank 1 predictions are used by default. If this is not provided, then the script will use design models \
                under lucs_location in the designs_df. Expected in colabfold format, with all predictions in a single directory.')
    parser.add_argument('--use_rank1_af', action='store_true',
            help='Use the rank 1 AF2/3 predictions under af_location in the designs_df for the design IDs in design_df.')
    parser.add_argument('--use_af_designid', action='store_true',
            help='Use AF2/3 design IDs in design_df.')
    parser.add_argument('--num_lhls', type=int, required=False, default=2,
            help='Number of Reshaped LHLs from which to find 6D helix vectors.')
    parser.add_argument('--pdb_suffix', type=str, required=False, default='',
            help='Suffix for DESIGNID_SUFFIX.pdb files.')
    parser.add_argument('--ins_pts_all_helical', action='store_true',
            help='Assume that all residues specified by insertion points are helical.')
    parser.add_argument('--convert_to_pyrosetta_nums', action='store_true',
            help='Convert insertion point residue indices to PyRosetta numbering (necessary when PDB files have chain breaks).')
    
    # Alignment
    parser.add_argument('--projection_residues', type=str, required=False, default='6,8',
			help='Residues used in calculating the projection point and frame. The CA of the first residue will be used \
                as the projection point, and frame between the two residues CAs (and residue 1s CB) will be the \
                projection frame. Format: <res1>,<res2>.')
    parser.add_argument('--align_by_non_bb_rem', action='store_true',
			help='Align the design structures to the reference structure by their non-backbone remodeled residues.')
    parser.add_argument('--use_align_residues_file', action='store_true',
			help='Align the design structures on the residues contained in their <lucs_dir>/<design_id>_align_residues.json \
			files, which contain lists of lists of residues.')
    parser.add_argument('--ref_align_residues_file', type=str, required=False, default=None,
			help='If using --use_align_residues_file to select residues on which to align for each design, then \
			provide an align_residues.json file (list of lists of residue indices) for the starting structure.')
    parser.add_argument('--aligned_pdb_outdir', type=str, required=False, default=None,
			help='If provided, then save PDB files of the aligned design models to the ref structure.')

    # Parallelization arguments
    parser.add_argument('--num_tasks', type=int, required=False, default=1,
            help='Number of tasks by which to divide the calculations.')
    parser.add_argument('--task_id', type=int, required=False, default=1,
            help='ID of the current task (1-indexed, e.g. $SGE_TASK_ID for SGE job distributor).')

    # Debugging arguments
    parser.add_argument('--verbose', action='store_true',
            help='Print extra print statements.')

    # Outputs
    parser.add_argument('--output_dir', type=str, required=True,
            help='Path to an output directory, containing dataframes, helix_coords, and sheet_coords\
            subdirectories.')
    parser.add_argument('--output_df_name', type=str, required=True,
            help='Names of output DataFrame (.csv) files, which will contain the original \
            dataframe plus a helix_coords column (written out to args.output_dir/dataframes).')

    args = parser.parse_args()

    # Load the dataframe containing LUCS designs.
    designs_df = pd.read_csv(args.designs_df_path)

    # Return a list of lists of sheet residues
    sheet_residues = get_sheet_residues(args.ref_pdb)

    helix_coords_out_dir = os.path.join(args.output_dir, 'helix_coords')
    os.makedirs(helix_coords_out_dir, exist_ok=True)

    # Save a list of lists of sheet backbone atom coords.
    sheet_coords_out_dir = os.path.join(args.output_dir, 'sheet_coords')
    os.makedirs(sheet_coords_out_dir, exist_ok=True)

    helix_dicts_list = []
    for lhl_id in range(args.num_lhls):
        helix_dicts, designs_df, pose, ref_pose = get_all_helix_dicts_for_dataframe(args, designs_df, args.ref_pdb, lhl_id=lhl_id)
        helix_coords = get_helix_coords_from_helix_dicts(helix_dicts, lhl_id=lhl_id)
        dump_helix_dicts(helix_dicts, os.path.join(helix_coords_out_dir, f'helix_dicts_lhl{lhl_id}_{args.task_id}.json'), lhl_id=lhl_id)
        dump_helix_coords(helix_coords, os.path.join(helix_coords_out_dir, f'helix_vectors_lhl{lhl_id}_{args.task_id}.json'))
        helix_dicts_list.append(helix_dicts)

    if args.verbose:
        print(f'Length of designs_df before dropping NA values: {len(designs_df)}')

    for lhl_id in range(args.num_lhls):
        # Drop the designs_df rows which were not processed in the current task
        designs_df = designs_df.dropna(subset=[f'helix{lhl_id}_coords'])

    if args.verbose:
        print(f'Length of designs_df after dropping NA values: {len(designs_df)}')

    # Write out the updated designs_df (with helix coords columns).
    dataframe_outdir = os.path.join(args.output_dir, 'dataframes')
    os.makedirs(dataframe_outdir, exist_ok=True)
    output_df_path = os.path.join(dataframe_outdir, args.output_df_name.split('/')[-1].replace('.csv', f'_{args.task_id}.csv'))
    designs_df.to_csv(output_df_path)

    if args.verbose:
        print(f'Saved Dataframe: {output_df_path}')

    # Use the residues specified in args.projection_residues to calculate the projection point and frame
    projection_residues = [int(r) for r in args.projection_residues.split(',')]
    project_point = ref_pose.residue(projection_residues[0]).xyz('CA')
    z = xyz_to_np_array(ref_pose.residue(projection_residues[0]).xyz('CA') - ref_pose.residue(projection_residues[0]).xyz('CB'))
    y = xyz_to_np_array(ref_pose.residue(projection_residues[1]).xyz('CA') - ref_pose.residue(projection_residues[0]).xyz('CA'))
    
    x = np.cross(y, z)
    y = np.cross(z, x)
    project_frame = [x / np.linalg.norm(x), y / np.linalg.norm(y), z / np.linalg.norm(z)]

    # Get ins_pts for the starting structure
    all_ins_pts = get_ins_pts(args.ref_insertion_points)
    for i, ins_pts in enumerate(all_ins_pts):
        helix_coords_ref = get_all_helix_coords_for_one_design_by_ins_pts(args, ref_pose, args.ref_pdb,
            ins_pts, project_point, project_frame, lhl_id=i, is_ref_struct=True)
        dump_helix_coords(helix_coords_ref,
            os.path.join(helix_coords_out_dir, f'ref_helix_coords_lhl{i}.json'))

    # Calculate the sheet coordinates in the new reference frame. Save them
    # in a list of lists in a json file.
    sheet_coords = get_sheet_coords(ref_pose, sheet_residues, project_point, project_frame)
    dump_sheet_coords(sheet_coords, os.path.join(sheet_coords_out_dir, 'sheet_coords.json'))

