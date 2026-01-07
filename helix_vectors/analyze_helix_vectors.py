#!/usr/bin/env python3
"""
Analyzes and visualizes 6D helix vectors from protein structures.

Author: Ben Orr
Date: 9.9.22
Adapted from: XingJie Pan's measure_structure_space.py

This script analyzes and visualizes 6D helix vectors (3D position + 3D direction)
calculated by calc_6d_helix_vectors.py. It provides tools for spatial binning,
filtering, coloring, and 3D visualization of helix vectors from input protein
structures.

The script supports:
- Loading and binning helix vectors in 6D space
- Multiple coloring schemes (by design ID, experimental validation, computational metrics)
- Non-redundant helix visualization (one per spatial bin)
- 3D plotting
- Movie frame generation for rotating visualizations
- Loading beta sheet coordinates from a reference structure

Helix Vector Format
-------------------
Each helix is represented as a 6-element vector:
    [x, y, z, vx, vy, vz]
where (x,y,z) is the helix centroid position and (vx,vy,vz) is the direction vector.

Spatial Binning
---------------
Helix vectors are binned into voxels based on:
- Position: 2 Angstrom bins for (x,y,z)
- Direction: Binary bins (positive/negative) for (vx,vy,vz)

This creates discrete spatial regions for analyzing helix distribution and redundancy.

Usage
-----
Basic usage:
    python analyze_helix_vectors.py --helix_coords_dir <dir> --sheet_coords_dir <dir> \\
        --starting_structure_name <name> --designs_df_path <df.csv> \\
        --color_by <metric> --output_dir <output>

Color by design ID:
    python analyze_helix_vectors.py ... --color_by design_id

Color by continuous metric:
    python analyze_helix_vectors.py ... --color_by helix_rmsd --colorbar_title "Helix RMSD"

Plot one helix per bin:
    python analyze_helix_vectors.py ... --plot_one_helix_per_bin

Generate movie frames:
    python analyze_helix_vectors.py ... --save_movie_frames

Parameters
----------
helix_coords_dir : str
    Directory containing helix_dicts_lhl{i}.json files with 6D helix vectors.

sheet_coords_dir : str
    Directory containing sheet_coords files for beta strand visualization.

starting_structure_name : str
    Name of starting structure (scaffold) used in file naming.

designs_df_path : str
    Path to CSV file with design_id column and metrics for coloring.

color_by : str
    Coloring scheme. Options:
    - 'design_id': Color by unique design
    - 'color_group': Color by binary group (0/1) in dataframe
    - 'folded_group': Color by experimental validation (folded/not_folded/low_expression)
    - 'train_test': Color by train/test split based on metric threshold
    - 'indiv_<colors>': Individual colors per LHL (e.g., 'indiv_blue,red')
    - 'color_all_<color>': Single color for all helices
    - '<metric_name>': Color by continuous metric from dataframe

num_lhls : int, optional
    Number of Loop-Helix-Loop units to analyze. Default is 2.

num_strands : int, optional
    Number of beta strands to plot. Default is 4.

plot_one_helix_per_bin : bool, optional
    If set, plot only one helix per spatial bin (non-redundant visualization).

plot_only_color_group : int, optional
    Plot only designs from specified color group (0 or 1). Default -1 (all).

hide_sheet : bool, optional
    If set, do not plot the beta sheet scaffold.

save_movie_frames : bool, optional
    If set, save 360 frames for movie generation.

elevation : int, optional
    Camera elevation angle for saved images. Default -1 (use preset orientations).

azimuth : int, optional
    Camera azimuth angle for saved images. Default -1 (use preset orientations).

output_dir : str
    Directory for output files including movie frames and hashed helix data.

Output Files
------------
The following files are saved to output_dir:
    - hashed_helix_dicts_list.json: Spatial bins with helix vectors
    - movies/<metric>/orient*.png: Oriented view images
    - movies/<metric>/movie*.png: Movie frames (if --save_movie_frames)
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Any, Optional, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import pandas as pd
import math


def load_helix_coords(file_name: str) -> List[npt.NDArray[np.float64]]:
    """
    Load helix coordinates from a JSON file.

    Parameters
    ----------
    file_name : str
        Path to JSON file containing serialized helix coordinates.

    Returns
    -------
    List[npt.NDArray[np.float64]]
        List of helix coordinate arrays, each with shape (6,) representing
        [x, y, z, vx, vy, vz] for centroid position and direction vector.
    """
    with open(file_name, 'r') as f:
        h_coords_serial = json.load(f)

    return [np.array(c) for c in h_coords_serial]


def load_helix_dicts(
    folder: str,
    file_header: str,
    lhl_id: int = 0
) -> List[Dict[str, Any]]:
    """
    Load list of helix dictionaries from JSON files.

    This function loads all JSON files matching the file_header pattern
    and aggregates them into a single list of design dictionaries.

    Parameters
    ----------
    folder : str
        Directory containing helix dictionary JSON files.
    file_header : str
        File name prefix to match (e.g., 'helix_dicts_lhl0_').
    lhl_id : int, optional
        Loop-Helix-Loop ID number. Default is 0.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries, each containing:
        - 'design_id': Design identifier
        - f'helix{lhl_id}_coords': NumPy array of helix coordinates

    Notes
    -----
    Helix coordinates are stored as lists in JSON and converted back to
    NumPy arrays for computational efficiency.
    """
    all_helix_dicts = []
    file_list = [f for f in os.listdir(folder) if file_header in f]
    for file_name in file_list:
        with open(os.path.join(folder, file_name), 'r') as f:
            helix_dicts = json.load(f)
        for design_dict in helix_dicts:
            all_helix_dicts.append(design_dict)

    # Convert helix coords back to numpy arrays.
    for design_dict in all_helix_dicts:
        design_dict[f'helix{lhl_id}_coords'] = np.array(design_dict[f'helix{lhl_id}_coords'])

    return all_helix_dicts


def load_sheet_coords(
    sheet_coords_out_dir: str,
    sheet_coords_fnames: List[str]
) -> List[List[npt.NDArray[np.float64]]]:
    """
    Load beta sheet coordinates from JSON files.

    Each JSON file contains coordinates for one beta strand. This function
    aggregates multiple strand coordinates for complete sheet visualization.

    Parameters
    ----------
    sheet_coords_out_dir : str
        Directory containing sheet coordinate JSON files.
    sheet_coords_fnames : List[str]
        List of JSON filenames to load (one per strand).

    Returns
    -------
    List[List[npt.NDArray[np.float64]]]
        List of lists, where each sublist contains coordinate arrays for
        one beta strand. Each coordinate array has shape (3,) for [x, y, z].

    Notes
    -----
    Files that cannot be loaded are skipped with a warning message.
    """
    sheet_coords = []
    for fname in sheet_coords_fnames:
        try:
            with open(os.path.join(sheet_coords_out_dir, fname), 'r') as f:
                s_coords_serial = json.load(f)
            print(f'Loaded {fname} in load_sheet_coords.')
        except:
            print(f'Could not load {fname} in load_sheet_coords.')
            continue

        sheet_coords.append([np.array(c) for c in s_coords_serial])

    return sheet_coords


def plot_helices(
    helix_coords: List[npt.NDArray[np.float64]],
    axis3d: Axes3D,
    length: float = 1,
    color: Union[str, List[str], Any] = 'blue',
    alpha: float = 1
) -> None:
    """
    Plot helix vectors as 3D arrows (quivers).

    Parameters
    ----------
    helix_coords : List[npt.NDArray[np.float64]]
        List of helix coordinate arrays, each with shape (6,) for
        [x, y, z, vx, vy, vz].
    axis3d : Axes3D
        Matplotlib 3D axis object for plotting.
    length : float, optional
        Length scaling factor for arrow vectors. Default is 1.
    color : Union[str, List[str], Any], optional
        Color specification:
        - str: Single color name for all helices
        - List[str]: Individual color for each helix
        - Other: Numeric color values (not recommended, see note)
        Default is 'blue'.
    alpha : float, optional
        Transparency level (0=transparent, 1=opaque). Default is 1.

    Notes
    -----
    When using a list of colors, matplotlib's quiver requires individual
    plotting per helix. Numeric color arrays don't work properly with quiver.
    """
    if type(color) == str:
        for h in helix_coords:
            axis3d.quiver(h[0], h[1], h[2], h[3], h[4], h[5], color=color, length=length, alpha=alpha)
    elif type(color) == list:
        for i in range(len(helix_coords)):
            h = helix_coords[i]
            c = color[i]
            axis3d.quiver(h[0], h[1], h[2], h[3], h[4], h[5], color=c, length=length, alpha=alpha)
    else:
        X = [h[0] for h in helix_coords]
        Y = [h[1] for h in helix_coords]
        Z = [h[2] for h in helix_coords]
        U = [h[3] for h in helix_coords]
        V = [h[4] for h in helix_coords]
        W = [h[5] for h in helix_coords]

        # for some reason this doesn't work with a list of colors
        axis3d.quiver(X, Y, Z, U, V, W, color=color, length=length, alpha=alpha)


def plot_sheet(
    sheet_coords: List[List[npt.NDArray[np.float64]]],
    axis3d: Axes3D,
    color: str = 'green',
    cmap: Optional[str] = None
) -> None:
    """
    Plot beta sheet as connected 3D lines.

    Each beta strand is plotted as a separate line. Optionally applies
    a colormap gradient along the strand from N-terminus to C-terminus.

    Parameters
    ----------
    sheet_coords : List[List[npt.NDArray[np.float64]]]
        List of beta strands, each containing coordinate arrays.
    axis3d : Axes3D
        Matplotlib 3D axis object for plotting.
    color : str, optional
        Default color for sheet strands. Default is 'green'.
    cmap : Optional[str], optional
        Colormap name for gradient coloring along strands. If None, uses
        solid color. Default is None.

    Notes
    -----
    When using a colormap, scatter points are added with gradient colors
    while lines are drawn with low transparency for structural context.
    """
    if cmap:
        cmap = plt.cm.get_cmap(cmap)
    for strand in sheet_coords:
        X = [c[0] for c in strand]
        Y = [c[1] for c in strand]
        Z = [c[2] for c in strand]
        axis3d.plot(X, Y, Z, color=color, alpha=0.3)
        if cmap:
            c = [cmap(i/len(strand)) for i in range(len(strand))]
            axis3d.scatter(X, Y, Z, color=c)


def bin_helix_dicts_by_hash(
    helix_dicts: List[Dict[str, Any]],
    lhl_id: int = 0
) -> Dict[Tuple[int, int, int, int, int, int], List[Dict[str, Any]]]:
    """
    Bin helix vectors into spatial voxels using hash-based binning.

    This function creates a spatial hash map where helices with similar
    positions and directions are grouped together. This enables:
    - Identification of redundant or similar helices
    - Non-redundant visualization
    - Analysis of structural diversity

    Parameters
    ----------
    helix_dicts : List[Dict[str, Any]]
        List of design dictionaries containing helix coordinates.
    lhl_id : int, optional
        Loop-Helix-Loop ID number. Default is 0.

    Returns
    -------
    Dict[Tuple[int, int, int, int, int, int], List[Dict[str, Any]]]
        Dictionary mapping spatial bins to design dictionaries.
        Keys are tuples (x, y, z, vx, vy, vz) representing:
        - (x, y, z): Position bin indices (2 Angstrom bins)
        - (vx, vy, vz): Direction signs (-1 or +1 for each component)
        Values are lists of design dictionaries in that bin.

    Notes
    -----
    Binning strategy:
    - Position: Discretized into 2 Angstrom voxels
    - Direction: Binary classification (positive/negative) per component

    This binary direction binning groups helices pointing in generally
    the same octant of 3D space, regardless of exact angle.
    """
    position_bin_size = 2

    hashed_helices = {}

    for design_dict in helix_dicts:
        for h in design_dict[f'helix{lhl_id}_coords']:
            x = int(h[0] / position_bin_size)
            y = int(h[1] / position_bin_size)
            z = int(h[2] / position_bin_size)

            # v_x = int(h[3] / direction_bin_size)
            # v_y = int(h[4] / direction_bin_size)
            # v_z = int(h[5] / direction_bin_size)

            ### Instead, bin by positive and negative directions for each directional component.
            v_x = 1 if h[3] >= 0 else -1
            v_y = 1 if h[4] >= 0 else -1
            v_z = 1 if h[5] >= 0 else -1

            key = (x, y, z, v_x, v_y, v_z)

            if key in hashed_helices.keys():
                hashed_helices[key].append(design_dict)
            else:
                hashed_helices[key] = [design_dict]

    return hashed_helices


def get_all_helices(
    hashed_helix_dicts: Dict[Tuple[int, int, int, int, int, int], List[Dict[str, Any]]],
    lhl_id: int = 0
) -> List[npt.NDArray[np.float64]]:
    """
    Extract all helix vectors from hashed dictionary.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Tuple, List[Dict]]
        Dictionary mapping spatial bins to design dictionaries.
    lhl_id : int, optional
        Loop-Helix-Loop ID number. Default is 0.

    Returns
    -------
    List[npt.NDArray[np.float64]]
        List of all helix coordinate arrays, each with shape (6,).

    Notes
    -----
    Each design dictionary contains a list of helix coordinates.
    This function extracts the first (and typically only) helix from each design.
    """
    helices = []
    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            # index 0 because value is a list of length 1, list[0] is list of coords
            helices.append(design_dict[f'helix{lhl_id}_coords'][0])

    # helices has shape (1,num_designs,6), so need index [0] to get (num_designs,6)
    return helices


def get_nonredundant_helices(
    hashed_helix_dicts: Dict[Tuple[int, int, int, int, int, int], List[Dict[str, Any]]],
    lhl_id: int = 0
) -> List[npt.NDArray[np.float64]]:
    """
    Extract one representative helix per spatial bin.

    This function returns a non-redundant set of helices by selecting
    the first helix from each spatial bin. Useful for visualization
    when full structural diversity is more important than showing all designs.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Tuple, List[Dict]]
        Dictionary mapping spatial bins to design dictionaries.
    lhl_id : int, optional
        Loop-Helix-Loop ID number. Default is 0.

    Returns
    -------
    List[npt.NDArray[np.float64]]
        List of non-redundant helix coordinate arrays, one per bin.

    Notes
    -----
    The number of returned helices equals the number of occupied spatial bins,
    which is typically much smaller than the total number of designs.
    """
    helices = []
    for k in hashed_helix_dicts.keys():
        helices.append(hashed_helix_dicts[k][0][f'helix{lhl_id}_coords'][0])

    return helices


def get_nonredundant_color_group_helices(
    hashed_helix_dicts: Dict[Tuple[int, int, int, int, int, int], List[Dict[str, Any]]],
    lhl_id: int = 0,
    color_group_col: str = 'id_color'
) -> List[npt.NDArray[np.float64]]:
    """
    Extract one helix per spatial bin per unique color group.

    This provides a semi-redundant representation where different experimental
    groups or design families are all represented in each spatial bin.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Tuple, List[Dict]]
        Dictionary mapping spatial bins to design dictionaries.
    lhl_id : int, optional
        Loop-Helix-Loop ID number. Default is 0.
    color_group_col : str, optional
        Dictionary key for color group labels. Default is 'id_color'.

    Returns
    -------
    List[npt.NDArray[np.float64]]
        List of helix coordinate arrays, one per bin per unique color group.

    Notes
    -----
    If a bin contains designs from multiple color groups, one representative
    from each group is included in the output.
    """
    helices = []
    for k in hashed_helix_dicts.keys():
        id_colors = []
        for design_dict in hashed_helix_dicts[k]:
            if design_dict[color_group_col] not in id_colors:
                id_colors.append(design_dict[color_group_col])
                helices.append(design_dict[f'helix{lhl_id}_coords'][0])

    return helices


def get_nonredundant_color_group_colors(
    hashed_helix_dicts: Dict[Tuple[int, int, int, int, int, int], List[Dict[str, Any]]],
    color_group_col: str = 'id_color'
) -> List[Any]:
    """
    Get colors corresponding to helices from get_nonredundant_color_group_helices.

    This function must be called with the same hashed_helix_dicts and
    color_group_col as get_nonredundant_color_group_helices to ensure
    color-to-helix correspondence.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Tuple, List[Dict]]
        Dictionary mapping spatial bins to design dictionaries.
    color_group_col : str, optional
        Dictionary key for color group labels. Default is 'id_color'.

    Returns
    -------
    List[Any]
        List of colors in the same order as helices from
        get_nonredundant_color_group_helices.

    Notes
    -----
    The ordering is guaranteed to match because Python 3.7+ dicts maintain
    insertion order.
    """
    colors = []
    for k in hashed_helix_dicts.keys():
        id_colors = []
        for design_dict in hashed_helix_dicts[k]:
            if design_dict[color_group_col] not in id_colors:
                id_colors.append(design_dict[color_group_col])
                colors.append(design_dict[color_group_col])

    return colors


def get_nonredundant_helices_from_hashed_coords(
    hashed_helices: Dict[Any, List[npt.NDArray[np.float64]]]
) -> List[npt.NDArray[np.float64]]:
    """
    Get one helix coordinate per hash bin.

    Parameters
    ----------
    hashed_helices : Dict[Any, List[npt.NDArray]]
        Dictionary mapping bins to lists of helix coordinate arrays.

    Returns
    -------
    List[npt.NDArray[np.float64]]
        List of helix coordinate arrays, one per bin.
    """
    helices = []

    for k in hashed_helices.keys():
        helices.append(hashed_helices[k][0])

    return helices


def get_all_helices_from_hashed_coords(
    hashed_helices: Dict[Any, List[npt.NDArray[np.float64]]]
) -> List[npt.NDArray[np.float64]]:
    """
    Extract all helix coordinates from hashed dictionary.

    Parameters
    ----------
    hashed_helices : Dict[Any, List[npt.NDArray]]
        Dictionary mapping bins to lists of helix coordinate arrays.

    Returns
    -------
    List[npt.NDArray[np.float64]]
        Flattened list of all helix coordinate arrays.
    """
    helices = []

    for k in hashed_helices.keys():
        for helix in hashed_helices[k]:
            helices.append(helix)

    return helices


def get_common_bins(
    hashed_helices1: Dict[Any, Any],
    hashed_helices2: Dict[Any, Any]
) -> List[Any]:
    """
    Find spatial bins occupied by both helix sets.

    This is useful for comparing structural diversity between two LHL units
    or design sets. Common bins indicate spatial regions where both sets
    have helix placement.

    Parameters
    ----------
    hashed_helices1 : Dict[Any, Any]
        First hashed helix dictionary.
    hashed_helices2 : Dict[Any, Any]
        Second hashed helix dictionary.

    Returns
    -------
    List[Any]
        List of bin keys present in both dictionaries.
    """
    keys1 = hashed_helices1.keys()
    keys2 = hashed_helices2.keys()

    common_bins = []

    for k in keys1:
        if k in keys2:
            common_bins.append(k)

    return common_bins


def make_movie_frames(
    ax: Axes3D,
    movie_folder: str,
    num_frames: int = 360,
    elev: int = 10
) -> None:
    """
    Generate rotating movie frames for 3D visualization.

    Saves a sequence of images with incremental azimuthal rotation.
    These frames can be compiled into a movie using ffmpeg or similar tools.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axis with plotted data.
    movie_folder : str
        Directory to save movie frame images.
    num_frames : int, optional
        Number of frames to generate (azimuth angles). Default is 360.
    elev : int, optional
        Camera elevation angle in degrees. Default is 10.

    Notes
    -----
    Each frame is saved as movie{i}.png where i is the azimuth angle.
    Full 360-degree rotation creates smooth looping animations.

    Examples
    --------
    To compile frames into a movie with ffmpeg:
    ```bash
    ffmpeg -framerate 30 -i movie%d.png -c:v libx264 output.mp4
    ```
    """
    # Save 360 images of the 3d plot, each at a different angle
    print('Generating movie frames...')
    # for ii in range(0,360,1):
    for ii in range(0, num_frames, 1):
        print('Saving movie number %d' % ii)
        ax.view_init(elev=elev, azim=ii)
        plt.savefig(os.path.join(movie_folder, "movie%d.png" % ii), dpi=300)
    print('Saved movie frames to %s' % movie_folder)


def save_oriented_frames(
    ax: Axes3D,
    movie_folder: str,
    elev: int = -1,
    azim: int = -1
) -> None:
    """
    Save images from multiple preset camera orientations.

    Generates standard views (top, sides, etc.) or a single custom view
    for publication-quality figures or multi-angle analysis.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axis with plotted data.
    movie_folder : str
        Directory to save oriented frame images.
    elev : int, optional
        Camera elevation angle. If -1, uses preset orientations. Default is -1.
    azim : int, optional
        Camera azimuth angle. If -1, uses preset orientations. Default is -1.

    Notes
    -----
    Preset orientations include 8 standard views:
    - 4 side views at 90-degree intervals (elev=90)
    - 4 angled views at 90-degree intervals (elev=0)

    If both elev and azim are specified (not -1), saves a single view.

    Output files are named: orient{i}_elev{elev}_azim{azim}_roll{roll}.png
    """
    # Save 360 images of the 3d plot, each at a different angle
    print('Generating oriented frames...')
    if args.elevation == -1 and args.azimuth == -1:
        orientations = [(90, 0),
                        (90, 90),
                        (90, 180),
                        (90, 270),
                        (0, 0),
                        (0, 90),
                        (0, 180),
                        (0, 270)]
    else:
        orientations = [(args.elevation, args.azimuth)]
    for i, (elev, azim) in enumerate(orientations):
        roll = 0
        print(f'Saving oriented frame number {i}, azim={azim}, roll={roll}')
        ax.view_init(elev=elev, azim=azim, roll=roll)
        outfile = os.path.join(movie_folder, f"orient{i}_elev{elev}_azim{azim}_roll{roll}.png")
        plt.savefig(outfile, dpi=300)
    print(f'Saved oriented frames to {outfile}')


def add_color_to_hashed_helices_dicts_by_bin(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]]
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Add 'bin_color' key to design dictionaries based on spatial bin.

    Assigns a unique color to all designs within each spatial bin.
    Useful for visualizing spatial clustering of designs.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.

    Returns
    -------
    Dict[Any, List[Dict[str, Any]]]
        Updated dictionary with 'bin_color' added to each design dictionary.

    Notes
    -----
    Limited to 9 distinct colors. Bins beyond the 9th will cycle colors.
    """
    colors = ['blue', 'red', 'orange', 'purple', 'gold', 'magenta', 'cyan',
              'grey', 'black']
    i = 0
    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            design_dict['bin_color'] = colors[i]
        i += 1
    return hashed_helix_dicts


def add_color_to_hashed_helices_dicts_by_design_id(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]]
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Add 'id_color' key using unique color per design ID.

    Assigns a continuous colormap color to each unique design ID.
    Useful for tracking individual designs across spatial bins.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.

    Returns
    -------
    Dict[Any, List[Dict[str, Any]]]
        Updated dictionary with 'id_color' added to each design dictionary.

    Notes
    -----
    Uses the 'Spectral' colormap for maximum visual distinction between designs.
    Colors are assigned based on design index order to ensure consistency.
    """
    colors = ['blue', 'red', 'orange', 'purple', 'gold', 'magenta', 'cyan',
              'grey', 'black']
    cmap = plt.cm.get_cmap('Spectral')

    # Count the number of unique design IDs
    num_design_ids = 0
    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            num_design_ids += 1

    prev_design_ids = {}
    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            if design_dict['design_id'] in prev_design_ids:
                design_dict['id_color'] = prev_design_ids[design_dict['design_id']]
            else:
                design_dict['id_color'] = cmap(len(prev_design_ids)/num_design_ids)
                prev_design_ids[design_dict['design_id']] = cmap(len(prev_design_ids)/num_design_ids)
    return hashed_helix_dicts


def add_color_to_hashed_helices_dicts_by_folded_group(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]]
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Add 'folded_group' key based on experimental validation results.

    This function is specific to the RO2 designs from Xingjie Pan et al.
    Science (2020). It classifies designs based on experimental outcomes:
    folded, not_folded, or low_expression.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.

    Returns
    -------
    Dict[Any, List[Dict[str, Any]]]
        Updated dictionary with 'folded_group' added to each design dictionary.

    Notes
    -----
    Design classification based on experimental validation:
    - folded: Designs confirmed to fold correctly
    - not_folded: Designs that failed to fold
    - low_expression: Designs with insufficient expression for testing

    Only applicable to the 25 experimentally validated RO2 designs.
    """
    name_map = {1: 835, 2: 2061, 3: 2970, 4: 8464, 5: 8651, 6: 8761, 7: 8893, 8: 10585, 9: 28492, 10: 28983,
                11: 29125, 12: 29635, 13: 29835, 14: 31765, 15: 33368, 16: 33559, 17: 35006, 18: 36272,
                19: 36455, 20: 36830, 21: 37025, 22: 37749, 23: 38420, 24: 41516, 25: 49223}
    # ro2_<ID> (paper IDs) of folded, not_folded, and low_expression groups of
    # experimentally validated RO2 designs.
    folded = [1, 5, 6, 9, 10, 15, 20, 25]
    not_folded = [3, 4, 7, 8, 11, 16, 17, 18, 19, 21, 22]
    low_expression = [2, 12, 13, 14, 23, 24]

    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            for paper_id in name_map.keys():
                if name_map[paper_id] == design_dict['design_id']:
                    if paper_id in folded:
                        design_dict['folded_group'] = 'folded'
                    elif paper_id in not_folded:
                        design_dict['folded_group'] = 'not_folded'
                    else:
                        design_dict['folded_group'] = 'low_expression'
    return hashed_helix_dicts


def add_color_to_hashed_helices_dicts_by_train_test(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]],
    df: pd.DataFrame,
    metric: str = 'helix_rmsd',
    threshold: float = 5,
    new_col: str = 'train_test'
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Add binary train/test classification based on metric threshold.

    Classifies designs as 'train' or 'test' based on whether a continuous
    metric exceeds a threshold. Useful for visualizing data splits in
    machine learning contexts.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.
    df : pd.DataFrame
        DataFrame containing design metrics.
    metric : str, optional
        Column name in df for the metric to threshold. Default is 'helix_rmsd'.
    threshold : float, optional
        Threshold value for classification. Default is 5.
    new_col : str, optional
        Name for the new classification column. Default is 'train_test'.

    Returns
    -------
    Dict[Any, List[Dict[str, Any]]]
        Updated dictionary with new_col added to each design dictionary.

    Notes
    -----
    Classification logic:
    - metric > threshold → 'test'
    - metric ≤ threshold → 'train'
    """

    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            df_row = df.loc[df['design_id'] == design_dict['design_id']].iloc[0]
            print('\n\ndf row is:')
            print(df_row)
            if df_row['helix_rmsd'] > threshold:
                design_dict[new_col] = 'test'
            else:
                design_dict[new_col] = 'train'

    return hashed_helix_dicts


def get_color_by_dict_key(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]],
    key: str
) -> List[Any]:
    """
    Extract color values for all helices by dictionary key.

    Returns colors in the same order as get_all_helices(), enabling
    direct use for matplotlib plotting.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.
    key : str
        Dictionary key containing color values.

    Returns
    -------
    List[Any]
        List of color values in order matching get_all_helices().

    Notes
    -----
    Order is guaranteed consistent with get_all_helices() because
    Python 3.7+ dictionaries maintain insertion order.

    Raises
    ------
    SystemExit
        If the specified key is not found in design dictionaries.
    """
    colors = []

    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            try:
                colors.append(design_dict[key])
            except:
                print(f"Key {key} not found in design_dict.")
                print(f"Design dict is: {design_dict}")
                exit(1)

    return colors


def get_nonredundant_color_by_dict_key(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]],
    key: str
) -> List[Any]:
    """
    Extract color values for non-redundant helices by dictionary key.

    Returns colors in the same order as get_nonredundant_helices(),
    with one color per spatial bin.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.
    key : str
        Dictionary key containing color values.

    Returns
    -------
    List[Any]
        List of color values, one per bin, matching get_nonredundant_helices().

    Notes
    -----
    Only the first design in each bin is used for color extraction.
    """
    colors = []

    for k in hashed_helix_dicts.keys():
        colors.append(hashed_helix_dicts[k][0][key])

    return colors


def get_continuous_colors_by_dict_key(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]],
    key: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    cmap: str = 'viridis'
) -> Tuple[List[Any], float, float]:
    """
    Map continuous metric values to colormap colors.

    Converts continuous numerical values (e.g., RMSD, pLDDT) to colors
    using a matplotlib colormap. Useful for visualizing metric distributions.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.
    key : str
        Dictionary key for the continuous metric.
    min_val : Optional[float], optional
        Minimum value for colormap normalization. If None, uses data minimum.
    max_val : Optional[float], optional
        Maximum value for colormap normalization. If None, uses data maximum.
    cmap : str, optional
        Matplotlib colormap name. Default is 'viridis'.

    Returns
    -------
    colors : List[Any]
        List of matplotlib colors in order matching get_all_helices().
    min_val : float
        Actual minimum value used for normalization.
    max_val : float
        Actual maximum value used for normalization.

    Notes
    -----
    Designs with None or NaN metric values are colored as 'slategrey'.
    Color mapping is linear between min_val and max_val.
    """
    cmap = plt.cm.get_cmap(cmap)
    values, colors = [], []
    NoneColor = 'slategrey'

    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            if design_dict[key] == None or pd.isna(design_dict[key]):
                continue
            else:
                values.append(design_dict[key])

    if min_val == None and max_val == None:
        max_val = max(values)
        min_val = min(values)

    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            if design_dict[key] == None or pd.isna(design_dict[key]):
                colors.append(NoneColor)
            else:
                if (design_dict[key]-min_val) / (max_val-min_val) > max_val and (max_val-min_val) > 1 and min_val > 0:
                    colors.append(cmap(1))
                else:
                    if min_val < 0:
                        colors.append(cmap((design_dict[key]-min_val) / (max_val-min_val)))
                    else:
                        colors.append(cmap((design_dict[key]-min_val) / (max_val-min_val)))

    return colors, min_val, max_val


def get_nonredundant_continuous_colors_by_dict_key(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]],
    key: str,
    cmap: str = 'viridis'
) -> List[Any]:
    """
    Map continuous metric values to colors for non-redundant helices.

    Similar to get_continuous_colors_by_dict_key() but returns one color
    per spatial bin instead of one per design.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.
    key : str
        Dictionary key for the continuous metric.
    cmap : str, optional
        Matplotlib colormap name. Default is 'viridis'.

    Returns
    -------
    List[Any]
        List of colors, one per bin, matching get_nonredundant_helices().

    Notes
    -----
    Uses the first design in each bin for color extraction.
    Automatically determines min/max from all first-in-bin values.
    """
    cmap = plt.cm.get_cmap(cmap)
    values, colors = [], []

    for k in hashed_helix_dicts.keys():
        values.append(hashed_helix_dicts[k][0][key])

    for k in hashed_helix_dicts.keys():
        colors.append(cmap((hashed_helix_dicts[k][0][key]-min(values)) / (max(values)-min(values))))

    return colors


def get_color_by_folded_group(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]]
) -> List[str]:
    """
    Get colors based on experimental folding validation groups.

    Maps experimental outcome classifications to standard colors:
    - folded → blue
    - not_folded → red
    - low_expression → mediumorchid

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary with 'folded_group' keys in design dictionaries.

    Returns
    -------
    List[str]
        List of color names matching get_all_helices() order.

    Notes
    -----
    Requires hashed_helix_dicts to have been processed by
    add_color_to_hashed_helices_dicts_by_folded_group() first.
    """
    color_map = {'folded': 'blue', 'not_folded': 'red', 'low_expression': 'mediumorchid'}
    colors = []
    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            colors.append(color_map[design_dict['folded_group']])
    return colors


def get_colors_by_train_test(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]]
) -> List[str]:
    """
    Get colors based on train/test classification.

    Maps binary classification to standard colors:
    - train → blue
    - test → red

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary with 'train_test' keys in design dictionaries.

    Returns
    -------
    List[str]
        List of color names matching get_all_helices() order.

    Notes
    -----
    Requires hashed_helix_dicts to have been processed by
    add_color_to_hashed_helices_dicts_by_train_test() first.
    """
    color_map = {'train': 'blue', 'test': 'red'}
    colors = []
    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            colors.append(color_map[design_dict['train_test']])
    return colors


def subset_helix_dicts(
    helix_dicts: List[Dict[str, Any]],
    design_ids: List[int]
) -> List[Dict[str, Any]]:
    """
    Filter helix dictionaries to include only specified design IDs.

    Parameters
    ----------
    helix_dicts : List[Dict]
        List of design dictionaries with helix coordinates.
    design_ids : List[int]
        List of design IDs to retain.

    Returns
    -------
    List[Dict[str, Any]]
        Filtered list containing only dictionaries with design_id in design_ids.

    Notes
    -----
    Useful for analyzing specific subsets such as stable designs or
    experimentally validated designs.
    """
    subset_helix_dicts = []
    for d in helix_dicts:
        if d['design_id'] in design_ids:
            subset_helix_dicts.append(d)
    return subset_helix_dicts


def filter_helix_dicts(
    hashed_helix_dicts: Dict[Any, List[Dict[str, Any]]],
    continuous_metric: str
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Remove designs with missing values for continuous metric.

    Filters out design dictionaries where the specified metric is None,
    enabling cleaner colormap visualization.

    Parameters
    ----------
    hashed_helix_dicts : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.
    continuous_metric : str
        Dictionary key for the metric to check.

    Returns
    -------
    Dict[Any, List[Dict[str, Any]]]
        Filtered dictionary with only designs having non-None metric values.

    Notes
    -----
    Bins may become empty after filtering. Empty bins remain as empty lists
    in the returned dictionary.
    """
    filtered_hashed_helix_dicts = {}
    for k in hashed_helix_dicts.keys():
        filtered_hashed_helix_dicts[k] = []
        for design_dict in hashed_helix_dicts[k]:
            if design_dict[continuous_metric] != None:
                filtered_hashed_helix_dicts[k].append(design_dict)
    return filtered_hashed_helix_dicts


def add_color_to_helix_dict(
    hashed_helix_dict: Dict[Any, List[Dict[str, Any]]],
    df: pd.DataFrame,
    continuous_metric: str
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Add continuous metric values from dataframe to helix dictionaries.

    Looks up metric values in the dataframe by design_id and adds them
    to the corresponding helix dictionaries.

    Parameters
    ----------
    hashed_helix_dict : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.
    df : pd.DataFrame
        DataFrame containing design_id and continuous_metric columns.
    continuous_metric : str
        Column name in df to extract and add to dictionaries.

    Returns
    -------
    Dict[Any, List[Dict[str, Any]]]
        Updated dictionary with continuous_metric values added.

    Notes
    -----
    Each design dictionary is updated in-place with the metric value.
    """
    hashed_helix_dict_with_color = {}
    for k in hashed_helix_dict.keys():
        hashed_helix_dict_with_color[k] = []
        for design_dict in hashed_helix_dict[k]:
            # Find the df row with the corresponding design_id
            df_row = df.loc[df['design_id'] == design_dict['design_id']].iloc[0]
            design_dict[continuous_metric] = df_row[continuous_metric]
            hashed_helix_dict_with_color[k].append(design_dict)

    return hashed_helix_dict_with_color


def add_color_to_hashed_helices_dicts_by_group(
    hashed_helix_dict: Dict[Any, List[Dict[str, Any]]],
    df: pd.DataFrame,
    new_col: str = 'color_group',
    color_dict: Dict[int, str] = {0: 'gray', 1: 'blue'},
    color_group_color: str = 'blue'
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Add color labels based on binary group classification in dataframe.

    Maps group labels (0 or 1) from dataframe to color names for visualization.

    Parameters
    ----------
    hashed_helix_dict : Dict[Any, List[Dict]]
        Dictionary mapping bins to design dictionaries.
    df : pd.DataFrame
        DataFrame with 'color_group' or 'Color_Group' column.
    new_col : str, optional
        Name for the new color column. Default is 'color_group'.
    color_dict : Dict[int, str], optional
        Mapping from group labels to colors. Default {0:'gray', 1:'blue'}.
    color_group_color : str, optional
        Color for group 1 (overrides color_dict[1]). Default is 'blue'.

    Returns
    -------
    Dict[Any, List[Dict[str, Any]]]
        Updated dictionary with color labels added.

    Notes
    -----
    Handles column name variations ('color_group' or 'Color_Group').
    The color_group_color parameter allows runtime color customization.
    """
    if 'color_group' in df.columns:
        color_group_col = 'color_group'
    else:
        color_group_col = 'Color_Group'

    color_dict[1] = color_group_color
    hashed_helix_dict_with_color = {}
    for k in hashed_helix_dict.keys():
        hashed_helix_dict_with_color[k] = []
        for design_dict in hashed_helix_dict[k]:
            # Find the df row with the corresponding design_id
            df_row = df.loc[df['design_id'] == design_dict['design_id']].iloc[0]
            design_dict[new_col] = color_dict[df_row[color_group_col]]
            hashed_helix_dict_with_color[k].append(design_dict)

    return hashed_helix_dict_with_color


if __name__ == '__main__':

    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description='Sample LUCS backbones and copy them to a new folder, split into subdirectories in the output folder.\
        Also copy their insertion_points.json or design_info.json files to a specified output folder.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Inputs
    parser.add_argument('--helix_coords_dir', type=str, required=True,
                        help='Path to a directory containing helix_coords{i}.json files, including \
            files for the starting structures helices.')
    parser.add_argument('--sheet_coords_dir', type=str, required=True,
                        help='Path to a directory containing sheet_coords{i}.json files.')
    parser.add_argument('--starting_structure_name', type=str, required=True,
                        help='Name of the starting structure which precedes _helix_coords_lhl{i}.json \
            in its helix_coords, and is in sheet_coords_{args.starting_structure_name}_strand1.json \
            in its sheet_coords files.')
    parser.add_argument('--designs_df_path', type=str, required=True,
                        help='Path to a DataFrame (.csv) file containing a design_id column and a \
            lucs_location column.')
    parser.add_argument('--num_lhls', type=int, required=False, default=2,
                        help='Number of Reshaped LHLs from which to find 6D helix vectors.')
    parser.add_argument('--num_strands', type=int, required=False, default=4,
                        help='Number of beta strands onto which to plot 6D helix vectors.')
    parser.add_argument('--stable_only', action='store_true',
                        help='Only plot stable designs helices.')
    parser.add_argument('--color_by', type=str, required=True,
                        help='Dataframe column by which to color helix vectors. Can have any of the \
            following values: [indiv_<comma_separated_color_names>, color_group, design_id, \
            folded_group, train_test, color_all_<color_name>]')
    parser.add_argument('--color_group_color', type=str, required=False, default='blue',
                        help='Color by which to color the color_group 1 designs.')
    parser.add_argument('--colorbar_title', type=str, required=False, default='',
                        help='Title for the 3D plot color bar. Underscores will be replaced by spaces.')
    parser.add_argument('--no_colorbar', action='store_true',
                        help='Remove the colorbar from the plot.')
    parser.add_argument('--plot_one_helix_per_bin', action='store_true',
                        help='Only plot one helix per helix coordinate bin.')
    parser.add_argument('--plot_only_color_group', type=int, required=False, default=-1,
                        help='Only plot one specified color group.')
    parser.add_argument('--hide_sheet', action='store_true',
                        help='Do not plot the underlying beta sheet.')
    parser.add_argument('--save_movie_frames', action='store_true',
                        help='Save 360 frames for each azimuth integer value.')
    parser.add_argument('--elevation', type=int, required=False, default=-1,
                        help='Set the elevation for the saved image.')
    parser.add_argument('--azimuth', type=int, required=False, default=-1,
                        help='Set the azimuth for the saved image.')

    # Parallelization arguments
    parser.add_argument('--num_tasks', type=int, required=False, default=1,
                        help='Number of tasks by which to divide the calculations.')
    parser.add_argument('--task_id', type=int, required=False, default=1,
                        help='ID of the current task (1-indexed, e.g. $SGE_TASK_ID for SGE job distributor).')

    # Debugging arguments
    parser.add_argument('--verbose', action='store_true',
                        help='Print extra print statements.')
    parser.add_argument('--trim_df', type=int, required=False, default=0,
                        help='Trim the number of designs plotted.')

    # Outputs
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to an output directory, which will contain a movies/color_by folder containing movie frames.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    designs_df = pd.read_csv(args.designs_df_path)

    if args.trim_df > 0:
        designs_df = designs_df.head(args.trim_df)

    if args.stable_only:
        stable_df = designs_df.loc[designs_df['stable_by_ec50'] == 'stable']
        design_ids = stable_df['design_id'].tolist()
    else:
        design_ids = designs_df['design_id'].tolist()

    if args.plot_only_color_group > -1:
        if 'color_group' in designs_df.columns:
            color_group_col = 'color_group'
        else:
            color_group_col = 'Color_Group'
        designs_df = designs_df.loc[designs_df[color_group_col] == args.plot_only_color_group]
        design_ids = designs_df['design_id'].tolist()

    print(f"\nNumber of designs to plot: {len(design_ids)}")

    helix_dicts_list = []
    hashed_helix_dicts_list = []
    for i in range(args.num_lhls):
        helix_dicts = load_helix_dicts(args.helix_coords_dir,
                                       f'helix_dicts_lhl{i}_', lhl_id=i)
        helix_dicts = subset_helix_dicts(helix_dicts, design_ids)
        hashed_helix_dicts = bin_helix_dicts_by_hash(helix_dicts, lhl_id=i)  # each key in hashed_helix_dicts is a bin (voxel)
        helix_dicts_list.append(helix_dicts)
        hashed_helix_dicts_list.append(hashed_helix_dicts)

    # Print the number of bins occupied by helices in each LHL
    for i, hashed_helix_dict in enumerate(hashed_helix_dicts_list):
        print(f'Number of bins occupied by helices in LHL {i}: {len(hashed_helix_dict.keys())}')
    # Saved the hashed helix dicts to a json file.
    hashed_helix_dicts_out_fname = os.path.join(args.output_dir, f'hashed_helix_dicts_list.json')
    # Convert the keys of each hashed helix dict to a string to make it JSON serializable.
    json_hashed_helix_dicts_list = []
    for i, hashed_helix_dict in enumerate(hashed_helix_dicts_list):
        new_hashed_helix_dict = {}
        for k in hashed_helix_dict.keys():
            new_hashed_helix_dict[str(k)] = []
            # hashed_helix_dict[k] is a list of dictionsaries, each contains {'design_id': ..., 'helix{lhl_id}_coords': ...}
            for design_dict in hashed_helix_dict[k]:
                # Convert the values in each key to lists from ndarrays
                new_hashed_helix_dict[str(k)].append((design_dict['design_id'], design_dict[f'helix{i}_coords'][0].tolist()))
        json_hashed_helix_dicts_list.append(new_hashed_helix_dict)
    with open(hashed_helix_dicts_out_fname, 'w') as f:
        json.dump(json_hashed_helix_dicts_list, f, indent=4)
    print(f'Saved hashed helix dicts to {hashed_helix_dicts_out_fname}')

    # If coloring by "group" column in designs_df, count the number of
    # bins occupied by helices in each group
    if args.color_by == 'color_group':
        for i, hashed_helix_dict in enumerate(hashed_helix_dicts_list):
            color_dict = {0: 'gray', 1: 'blue'}
            hashed_helix_dicts_list[i] = add_color_to_hashed_helices_dicts_by_group(hashed_helix_dict, designs_df, new_col='id_color', color_dict=color_dict, color_group_color=args.color_group_color)
            for group in color_dict.values():
                group_count = 0
                for k in hashed_helix_dict.keys():
                    for design_dict in hashed_helix_dict[k]:
                        if design_dict['id_color'] == group:
                            group_count += 1
                            break  # go to the next key

                print(f'Number of bins occupied by helices in LHL {i} with color {group}: {group_count}')

    # Count the number of helices across the 2 LHLs that are in common bins
    if args.num_lhls == 2:
        # Find bins that are occupied by both sets of hashed helices
        # Currently, this only works with 2 LHLs
        common_bins = get_common_bins(hashed_helix_dicts_list[0], hashed_helix_dicts_list[1])
        print('Num bins 1 = ', len(hashed_helix_dicts_list[0].keys()))
        print('Num bins 2 = ', len(hashed_helix_dicts_list[1].keys()))
        print('Num common bins = ', len(common_bins))

        n_helices_in_common_bins = 0
        for k in common_bins:
            n_helices_in_common_bins += len(hashed_helix_dicts_list[1][k])

        print('Num helices in common bins = ', n_helices_in_common_bins)

        print('Num designs in hashed helix dicts')
        num1, num2 = 0, 0
        for k in hashed_helix_dicts_list[0].keys():
            for design_dict in hashed_helix_dicts_list[0][k]:
                num1 += 1
        for k in hashed_helix_dicts_list[1].keys():
            for design_dict in hashed_helix_dicts_list[1][k]:
                num2 += 1
        print(num1)
        print(num2)

    if not args.hide_sheet:
        # Load and plot sheet coordinates.
        sheet_coords_fnames = []
        for i in range(args.num_strands):
            sheet_coords_fnames.append(f'sheet_coords_{args.starting_structure_name}_strand{i}.json')
            sheet_coords_fnames.append(f'sheet_coords_strand{i}.json')

        sheet_coords = load_sheet_coords(args.sheet_coords_dir, sheet_coords_fnames)

        plot_sheet(sheet_coords, ax, color='grey', cmap='Greys')  # with cmap='Greys', N-term is white and C-term is black


    loaded_colors = False

    if 'indiv' in args.color_by:
        indiv_helix_colors = args.color_by.replace('indiv_', '').split(',')
        indiv_helix_colors = [c.strip() for c in indiv_helix_colors]
        if len(indiv_helix_colors) != args.num_lhls:
            print(f'Number of colors provided ({len(indiv_helix_colors)}) does not match number of LHLs ({args.num_lhls}).')
            exit(1)
        for i, hashed_helix_dict in enumerate(hashed_helix_dicts_list):
            for k in hashed_helix_dicts.keys():
                for design_dict in hashed_helix_dicts[k]:
                    design_dict['id_color'] = indiv_helix_colors[i]

    elif 'color_all_' in args.color_by:
        c = args.color_by.replace('color_all_', '')
        for hashed_helix_dict in hashed_helix_dicts_list:
            for k in hashed_helix_dicts.keys():
                for design_dict in hashed_helix_dicts[k]:
                    design_dict['id_color'] = c
    # Color helices by color_group
    elif args.color_by == 'color_group':
        for i, hashed_helix_dict in enumerate(hashed_helix_dicts_list):
            hashed_helix_dicts_list[i] = add_color_to_hashed_helices_dicts_by_group(hashed_helix_dict, designs_df, new_col='id_color', color_group_color=args.color_group_color)

    elif args.color_by == 'design_id':
        for i, hashed_helix_dict in enumerate(hashed_helix_dicts_list):
            hashed_helix_dicts_list[i] = add_color_to_hashed_helices_dicts_by_design_id(hashed_helix_dict)
            loaded_colors = True

    elif args.color_by == 'folded_group':
        # This function only works for Exp Val RO2 designs.
        for i, hashed_helix_dict in enumerate(hashed_helix_dicts_list):
            hashed_helix_dicts_list[i] = add_color_to_hashed_helices_dicts_by_folded_group(hashed_helix_dict)
            loaded_colors = True

    elif args.color_by == 'train_test':
        for i, hashed_helix_dict in enumerate(hashed_helix_dicts_list):
            hashed_helix_dicts_list[i] = add_color_to_hashed_helices_dicts_by_train_test(hashed_helix_dict, designs_df, metric='helix_rmsd', threshold=5, new_col='train_test')
            loaded_colors = True

    if loaded_colors:
        if args.plot_one_helix_per_bin:
            print('\nPlotting one helix per bin.\n')
            plot_helices(get_nonredundant_helices(hashed_helix_dicts_list[0]), ax, length=3,
                         color=get_nonredundant_color_by_dict_key(hashed_helix_dicts_list[0], 'id_color'))
            plot_helices(get_nonredundant_helices(hashed_helix_dicts_list[1]), ax, length=3,
                         color=get_nonredundant_color_by_dict_key(hashed_helix_dicts_list[1], 'id_color'))
        else:
            plot_helices(get_all_helices(hashed_helix_dicts_list[0]), ax, length=3,
                         color=get_color_by_dict_key(hashed_helix_dicts_list[0], 'id_color'))
            plot_helices(get_all_helices(hashed_helix_dicts_list[1]), ax, length=3,
                         color=get_color_by_dict_key(hashed_helix_dicts_list[1], 'id_color'))

    else:
        # Color helices by continuous values.
        continuous_metric = args.color_by

        continuous_metric_label = args.colorbar_title.replace('_', ' ')

        # For coloring by a continuous metric
        if 'plddt' in args.color_by and 'rmsd' not in args.color_by:
            min_val, max_val = 60, 100
            tick_width = 10
            ticks = list(range(min_val, max_val+1, tick_width))
            cmap = 'rainbow_r'
            alpha = 0.5

        elif 'rmsd' in args.color_by and 'plddt' not in args.color_by:
            min_val, max_val = 0, 5
            tick_width = 1
            ticks = list(range(min_val, max_val+1, tick_width))
            cmap = 'rainbow'
            alpha = 0.5

        if not args.no_colorbar:
            # Display a colorbar
            norm = matplotlib.colors.Normalize()
            norm.autoscale(ticks)
            cmap = plt.cm.get_cmap(cmap)
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ticks=ticks, ax=ax)
            cbar.set_label(args.colorbar_title.replace('_', ' '), size=15)
            cbar.ax.set_yticklabels(ticks)
            cbar.ax.tick_params(labelsize=15)

            if 'plddt' in args.color_by and 'rmsd' not in args.color_by:
                # Manually change last tick to have < or >
                cbar.ax.set_yticklabels(['< %s'%str(min_val)]+ticks[1:])
                # cbar.ax.set_yticklabels(ticks[:-1]+['> %s'%str(max_val)])

            elif 'rmsd' in args.color_by and 'plddt' not in args.color_by:
                # Manually change last tick to have < or >
                # cbar.ax.set_yticklabels(['< %s'%str(min_val)]+ticks[1:])
                cbar.ax.set_yticklabels(ticks[:-1]+['> %s'%str(max_val)])

            else:
                min_val, max_val = None, None
                num_ticks = 6
                alpha = 0.1

    filtered_hashed_helix_dicts_list = []
    for i, hashed_helix_dict in enumerate(hashed_helix_dicts_list):
        # filter_helix_dicts is only necessary when some designs have nan values for the continuous metric
        # filtered_hashed_helix_dicts = filter_helix_dicts(hashed_helix_dict, continuous_metric)
        if 'color_all_' in args.color_by:
            c = args.color_by.replace('color_all_', '')

            # With thousands of helix vectors, use these params:
            alpha = 0.1
            length = 3

            # For smaller sets of helix vectors, use these params:
            # alpha = 1
            # length = 5

            hashed_helix_dict_with_color = hashed_helix_dict
            if args.plot_one_helix_per_bin:
                alpha = 0.5
                length = 10
                plot_helices(get_nonredundant_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=length,
                             color=c, alpha=alpha)
            else:
                plot_helices(get_all_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=length,
                             color=c, alpha=alpha)

        # Color helices by individual provided colors
        elif 'indiv' in args.color_by:
            indiv_helix_colors = args.color_by.replace('indiv_', '').split(',')
            indiv_helix_colors = [c.strip() for c in indiv_helix_colors]
            hashed_helix_dict_with_color = hashed_helix_dict
            # With thousands of helix vectors, use these params:
            alpha = 0.1
            length = 5
            if args.plot_one_helix_per_bin:
                alpha = 0.5
                length = 10
                plot_helices(get_nonredundant_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=length,
                             color=indiv_helix_colors[i], alpha=alpha)
            else:
                plot_helices(get_all_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=length,
                             color=indiv_helix_colors[i], alpha=alpha)

        elif args.color_by == 'color_group':
            # With thousands of helix vectors, use these params:
            alpha = 0.5
            length = 5

            # For smaller sets of helix vectors, use these params:
            # alpha = 1
            # length = 5

            hashed_helix_dict_with_color = hashed_helix_dict

            if args.plot_one_helix_per_bin:
                alpha = 0.75
                length = 5
                plot_helices(get_nonredundant_color_group_helices(hashed_helix_dict_with_color, lhl_id=i, color_group_col='id_color'), ax, length=length,
                             color=get_nonredundant_color_group_colors(hashed_helix_dict_with_color, color_group_col='id_color'), alpha=alpha)
            else:
                plot_helices(get_all_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=length,
                             color=get_color_by_dict_key(hashed_helix_dict_with_color, 'id_color'), alpha=alpha)
        else:
            hashed_helix_dict_with_color = add_color_to_helix_dict(hashed_helix_dict, designs_df, continuous_metric)
            if args.plot_one_helix_per_bin:
                alpha = 0.75
                length = 5
                plot_helices(get_nonredundant_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=3,
                             color=get_nonredundant_continuous_colors_by_dict_key(hashed_helix_dict_with_color, continuous_metric)[0], alpha=alpha)
            else:
                plot_helices(get_all_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=3,
                             color=get_continuous_colors_by_dict_key(hashed_helix_dict_with_color,
                                                                      continuous_metric, min_val, max_val, cmap=cmap)[0], alpha=alpha)
        filtered_hashed_helix_dicts_list.append(hashed_helix_dict_with_color)

    # Remove the 3D axes
    ax_lims = (-15, 15)
    ax.set_xlim(ax_lims)
    ax.set_ylim(ax_lims)
    ax.set_zlim(ax_lims)
    ax.set_axis_off()

    # Make frames for a movie that can be compiled with ffmpeg
    movie_folder = os.path.join(args.output_dir, 'movies', continuous_metric)
    os.makedirs(movie_folder, exist_ok=True)
    if args.save_movie_frames:
        make_movie_frames(ax, movie_folder, num_frames=360, elev=args.elevation, azim=args.azimuth)
    save_oriented_frames(ax, movie_folder, elev=args.elevation, azim=args.azimuth)
    # plt.show()
    plt.close()
