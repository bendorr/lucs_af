#!/usr/bin/env python3
"""
analyze_helix_vectors.py

Ben Orr
9.9.22

Adapted from XingJie Pan's measure_structure_space.py

Analyze and plot 6D helix vectors calculated by calc_6d_helix_vectors.py

"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import pandas as pd
import math


def load_helix_coords(file_name):
    """
    Load helix coordinates from a json file
    """
    with open(file_name, 'r') as f:
        h_coords_serial = json.load(f)

    return [np.array(c) for c in h_coords_serial]

def load_helix_dicts(folder, file_header, lhl_id=0):
    """
    Load list of helix dictionaries from a json file.
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

def load_sheet_coords(sheet_coords_out_dir, sheet_coords_fnames):
    """
    Load sheet coordinates from json files.  Each sheet_coord_fname has
    coordinates for one beta strand.
    Return a list of lists. Each sublist contains coordinates for one strand.
    """
    sheet_coords = []
    for fname in sheet_coords_fnames:
        try:
            with open(os.path.join(sheet_coords_out_dir,fname), 'r') as f:
                s_coords_serial = json.load(f)
            print(f'Loaded {fname} in load_sheet_coords.')
        except:
            print(f'Could not load {fname} in load_sheet_coords.')
            continue

        sheet_coords.append([np.array(c) for c in s_coords_serial])

    return sheet_coords


def plot_helices(helix_coords, axis3d, length=1, color='blue', alpha=1):
    """
    Plot the helix coordinates in 3D
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

def plot_sheet(sheet_coords, axis3d, color='green', cmap=None):
    """
    Plot each sublist in sheet_coords as a separate beta strand.
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

def bin_helix_dicts_by_hash(helix_dicts, lhl_id=0):
    """
    Bin the helices in the helix_dicts by hashing.
    Return a dictionary of hashed vectors. Each dict value is a list of
    dictionaries, each subdictionary represents a single design's helix coords.

    Bin helix direction vectors by positive and negative directions 
    for each directional component.
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

def get_nonredundant_helices(hashed_helix_dicts):
    """
    Get one helix for each hash bin.
    """
    helices = []

    for k in hashed_helix_dicts.keys():
        helices.append(hashed_helix_dicts[k][0]['helix_coords'][0])

    return helices

def get_all_helices(hashed_helix_dicts, lhl_id=0):
    """
    Return a list of lists, each sublist containing the centroid coordinates
    and direction vector of a single helix.
    """
    helices = []
    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            # index 0 because value is a list of length 1, list[0] is list of coords
            helices.append(design_dict[f'helix{lhl_id}_coords'][0])

    # helices has shape (1,num_designs,6), so need index [0] to get (num_designs,6)
    return helices

def get_nonredundant_helices(hashed_helix_dicts, lhl_id=0):
    """
    Return a list of lists, each sublist containing the centroid coordinates
    and direction vector of a single helix.
    """
    helices = []
    for k in hashed_helix_dicts.keys():
        helices.append(hashed_helix_dicts[k][0][f'helix{lhl_id}_coords'][0])

    return helices

def get_nonredundant_color_group_helices(hashed_helix_dicts, lhl_id=0, color_group_col='id_color'):
    """
    Return one helix per 3D bin per unique color_group
    """
    helices = []
    for k in hashed_helix_dicts.keys():
        id_colors = []
        for design_dict in hashed_helix_dicts[k]:
            if design_dict[color_group_col] not in id_colors:
                id_colors.append(design_dict[color_group_col])
                helices.append(design_dict[f'helix{lhl_id}_coords'][0])

    return helices

def get_nonredundant_color_group_colors(hashed_helix_dicts, color_group_col='id_color'):
    """
    Return the colors corresponding to the helices returned by get_nonredundant_color_group_helices()
    """
    colors = []
    for k in hashed_helix_dicts.keys():
        id_colors = []
        for design_dict in hashed_helix_dicts[k]:
            if design_dict[color_group_col] not in id_colors:
                id_colors.append(design_dict[color_group_col])
                colors.append(design_dict[color_group_col])

    return colors


def get_nonredundant_helices_from_hashed_coords(hashed_helices):
    """
    Get one helix for each hash bin.
    """
    helices = []

    for k in hashed_helices.keys():
        helices.append(hashed_helices[k][0])

    return helices

def get_all_helices_from_hashed_coords(hashed_helices):
    """
    Return a list of lists, each sublist containing the centroid coordinates
    and direction vector of a single helix.
    """
    helices = []

    for k in hashed_helices.keys():
        for helix in hashed_helices[k]:
            helices.append(helix)

    return helices

def get_common_bins(hashed_helices1, hashed_helices2):
    """
    Get the keys of common bins.
    """
    keys1 = hashed_helices1.keys()
    keys2 = hashed_helices2.keys()

    common_bins = []

    for k in keys1:
        if k in keys2:
            common_bins.append(k)

    return common_bins

def make_movie_frames(ax, movie_folder, num_frames=360, elev=10):
    # Save 360 images of the 3d plot, each at a different angle
    print('Generating movie frames...')
    # for ii in range(0,360,1):
    for ii in range(0,num_frames,1):
        print('Saving movie number %d' % ii)
        ax.view_init(elev=elev, azim = ii)
        plt.savefig(os.path.join(movie_folder,"movie%d.png" % ii), dpi=300)
    print('Saved movie frames to %s' % movie_folder)

def save_oriented_frames(ax, movie_folder, elev=-1, azim=-1):
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

def add_color_to_hashed_helices_dicts_by_bin(hashed_helix_dicts):
    """
    Add keys for color to each subdict in the provided list of dictionaries.
    """
    colors = ['blue', 'red', 'orange', 'purple', 'gold', 'magenta', 'cyan', 
        'grey', 'black']
    i = 0
    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            design_dict['bin_color'] = colors[i]
        i+=1
    return hashed_helix_dicts

def add_color_to_hashed_helices_dicts_by_design_id(hashed_helix_dicts):
    """
    Add keys for color to each subdict in the provided list of dictionaries.
    Color determined by design ID
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

def add_color_to_hashed_helices_dicts_by_folded_group(hashed_helix_dicts):
    """
    Add keys for color to each subdict in the provided list of dictionaries.
    Color determined by Experimentally Validated RO2 Folded Group from
    Xingjie Pan et al. Science (2020)
    """
    name_map = {1:835, 2:2061, 3:2970, 4:8464, 5:8651, 6:8761, 7:8893, 8:10585, 9:28492, 10:28983,
            11:29125, 12:29635, 13:29835, 14:31765, 15:33368, 16:33559, 17:35006, 18:36272,
            19:36455, 20:36830, 21:37025, 22:37749, 23:38420, 24:41516, 25:49223}
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

def add_color_to_hashed_helices_dicts_by_train_test(hashed_helix_dicts, df, metric='helix_rmsd', threshold=5, new_col='train_test'):
    """
    Add keys for color to each subdict in the provided list of dictionaries.
    Color determined by train test split (thresholded metric)
    """

    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            df_row = df.loc[df['design_id']==design_dict['design_id']].iloc[0]
            print('\n\ndf row is:')
            print(df_row)
            if df_row['helix_rmsd'] > threshold:
                design_dict[new_col] = 'test'
            else:
                design_dict[new_col] = 'train'

    return hashed_helix_dicts

def get_color_by_dict_key(hashed_helix_dicts, key):
    """
    Return a list of colors that matches the order of the list returned by
    get_all_helices(hashed_helix_dicts).  Since dicts are ordered in python3,
    these orderings should match.
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

def get_nonredundant_color_by_dict_key(hashed_helix_dicts, key):
    """
    Return a list of colors that matches the order of the list returned by
    get_nonredundant_helices(hashed_helix_dicts).  Since dicts are ordered in python3,
    these orderings should match.
    """
    colors = []

    for k in hashed_helix_dicts.keys():
        colors.append(hashed_helix_dicts[k][0][key])

    return colors

def get_continuous_colors_by_dict_key(hashed_helix_dicts, key, 
    min_val=None, max_val=None, cmap='viridis'):
    """
    Return a list of colors that matches the order of the list returned by
    get_nonredundant_helices(hashed_helix_dicts).  Since dicts are ordered in python3,
    these orderings should match.
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

    if min_val==None and max_val==None:
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

def get_nonredundant_continuous_colors_by_dict_key(hashed_helix_dicts, key, cmap='viridis'):
    """
    Return a list of colors that matches the order of the list returned by
    get_nonredundant_helices(hashed_helix_dicts).  Since dicts are ordered in python3,
    these orderings should match.
    """
    cmap = plt.cm.get_cmap(cmap)
    values, colors = [], []

    for k in hashed_helix_dicts.keys():
        values.append(hashed_helix_dicts[k][0][key])

    for k in hashed_helix_dicts.keys():
        colors.append(cmap((hashed_helix_dicts[k][0][key]-min(values)) / (max(values)-min(values))))

    return colors

def get_color_by_folded_group(hashed_helix_dicts):
    """
    Return a list of colors by the 'folded_group' values in a dictionary with
    keys=helix bin, values=lists of dictionaries (each subdict represents one
    design)
    """
    color_map = {'folded':'blue', 'not_folded':'red', 'low_expression':'mediumorchid'}
    colors = []
    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            colors.append(color_map[design_dict['folded_group']])
    return colors

def get_colors_by_train_test(hashed_helix_dicts):
    """
    Return a list of colors by the a thresholded continuous metric
    """
    color_map = {'train':'blue', 'test':'red'}
    colors = []
    for k in hashed_helix_dicts.keys():
        for design_dict in hashed_helix_dicts[k]:
            colors.append(color_map[design_dict['train_test']])
    return colors

def subset_helix_dicts(helix_dicts, design_ids):
    """
    Return a list of dictionaries (in the same format as helix_dicts),
    each subdictionary with a design_id value in the list of provided
    design IDs.
    Exclude dictionaries with design IDs not in the design_ids list.
    """
    subset_helix_dicts = []
    for d in helix_dicts:
        if d['design_id'] in design_ids:
            subset_helix_dicts.append(d)
    return subset_helix_dicts

def filter_helix_dicts(hashed_helix_dicts, continuous_metric):
    """
    Eliminate design_dicts from the hashed_helix_dicts that have None
    values for the continuous_metric
    """
    filtered_hashed_helix_dicts = {}
    for k in hashed_helix_dicts.keys():
        filtered_hashed_helix_dicts[k] = []
        for design_dict in hashed_helix_dicts[k]:
            if design_dict[continuous_metric] != None:
                filtered_hashed_helix_dicts[k].append(design_dict)
    return filtered_hashed_helix_dicts

def add_color_to_helix_dict(hashed_helix_dict, df, continuous_metric):
    """
    Add colors to the hashed_helix_dict using the continuous_metric 
    values in the provided df.
    """
    hashed_helix_dict_with_color = {}
    for k in hashed_helix_dict.keys():
        hashed_helix_dict_with_color[k] = []
        for design_dict in hashed_helix_dict[k]:
            # Find the df row with the corresponding design_id
            df_row = df.loc[df['design_id']==design_dict['design_id']].iloc[0]
            design_dict[continuous_metric] = df_row[continuous_metric]
            hashed_helix_dict_with_color[k].append(design_dict)

    return hashed_helix_dict_with_color

def add_color_to_hashed_helices_dicts_by_group(hashed_helix_dicts, df, new_col='color_group', color_dict = {0:'gray', 1:'blue'}, color_group_color='blue'):
    """
    Add colors to the hashed_helix_dict using the group label
    in the input df
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
            df_row = df.loc[df['design_id']==design_dict['design_id']].iloc[0]
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
        stable_df = designs_df.loc[designs_df['stable_by_ec50']=='stable']
        design_ids = stable_df['design_id'].tolist()
    else:
        design_ids = designs_df['design_id'].tolist()

    if args.plot_only_color_group > -1:
        if 'color_group' in designs_df.columns:
            color_group_col = 'color_group'
        else:
            color_group_col = 'Color_Group'
        designs_df = designs_df.loc[designs_df[color_group_col]==args.plot_only_color_group]
        design_ids = designs_df['design_id'].tolist()

    print(f"\nNumber of designs to plot: {len(design_ids)}")

    helix_dicts_list = []
    hashed_helix_dicts_list = []
    for i in range(args.num_lhls):
        helix_dicts = load_helix_dicts(args.helix_coords_dir, 
            f'helix_dicts_lhl{i}_', lhl_id=i)
        helix_dicts = subset_helix_dicts(helix_dicts, design_ids)
        hashed_helix_dicts = bin_helix_dicts_by_hash(helix_dicts, lhl_id=i) # each key in hashed_helix_dicts is a bin (voxel)
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
            color_dict = {0:'gray', 1:'blue'}
            hashed_helix_dicts_list[i] = add_color_to_hashed_helices_dicts_by_group(hashed_helix_dict, designs_df, new_col='id_color', color_dict=color_dict, color_group_color=args.color_group_color)
            for group in color_dict.values():
                group_count = 0
                for k in hashed_helix_dict.keys():
                    for design_dict in hashed_helix_dict[k]:
                        if design_dict['id_color'] == group:
                            group_count += 1
                            break # go to the next key

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
                num1+=1
        for k in hashed_helix_dicts_list[1].keys():
            for design_dict in hashed_helix_dicts_list[1][k]:
                num2+=1
        print(num1)
        print(num2)

    if not args.hide_sheet:
        # Load and plot sheet coordinates.
        sheet_coords_fnames = []
        for i in range(args.num_strands):
            sheet_coords_fnames.append(f'sheet_coords_{args.starting_structure_name}_strand{i}.json')
            sheet_coords_fnames.append(f'sheet_coords_strand{i}.json')
        
        sheet_coords = load_sheet_coords(args.sheet_coords_dir, sheet_coords_fnames)

        plot_sheet(sheet_coords, ax, color='grey', cmap='Greys') # with cmap='Greys', N-term is white and C-term is black


    loaded_colors = False

    if 'indiv' in args.color_by:
        indiv_helix_colors = args.color_by.replace('indiv_','').split(',')
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
            plot_helices(get_nonredundant_helices(hashed_helix_dicts_list[0]), ax, length=3, \
                color=get_nonredundant_color_by_dict_key(hashed_helix_dicts_list[0], 'id_color'))
            plot_helices(get_nonredundant_helices(hashed_helix_dicts_list[1]), ax, length=3, \
                color=get_nonredundant_color_by_dict_key(hashed_helix_dicts_list[1], 'id_color'))
        else:
            plot_helices(get_all_helices(hashed_helix_dicts_list[0]), ax, length=3, \
                color=get_color_by_dict_key(hashed_helix_dicts_list[0], 'id_color'))
            plot_helices(get_all_helices(hashed_helix_dicts_list[1]), ax, length=3, \
                color=get_color_by_dict_key(hashed_helix_dicts_list[1], 'id_color'))
    
    else:
        # Color helices by continuous values.
        continuous_metric = args.color_by

        continuous_metric_label = args.colorbar_title.replace('_',' ')

        # For coloring by a continuous metric
        if 'plddt' in args.color_by and 'rmsd' not in args.color_by:
            min_val, max_val = 60, 100
            tick_width = 10
            ticks = list(range(min_val,max_val+1,tick_width))
            cmap='rainbow_r'
            alpha=0.5

        elif 'rmsd' in args.color_by and 'plddt' not in args.color_by:
            min_val, max_val = 0, 5
            tick_width = 1
            ticks = list(range(min_val,max_val+1,tick_width))
            cmap='rainbow'
            alpha=0.5

        if not args.no_colorbar:
            # Display a colorbar
            norm = matplotlib.colors.Normalize()
            norm.autoscale(ticks)
            cmap = plt.cm.get_cmap(cmap)
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ticks=ticks, ax=ax)
            cbar.set_label(args.colorbar_title.replace('_',' '), size=15)
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
                alpha=0.1

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
                plot_helices(get_nonredundant_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=length, \
                    color=c, alpha=alpha)
            else:
                plot_helices(get_all_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=length, \
                    color=c, alpha=alpha)
        
        # Color helices by individual provided colors
        elif 'indiv' in args.color_by:
            indiv_helix_colors = args.color_by.replace('indiv_','').split(',')
            indiv_helix_colors = [c.strip() for c in indiv_helix_colors]
            hashed_helix_dict_with_color = hashed_helix_dict
            # With thousands of helix vectors, use these params:
            alpha = 0.1
            length = 5
            if args.plot_one_helix_per_bin:
                alpha = 0.5
                length = 10
                plot_helices(get_nonredundant_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=length, \
                    color=indiv_helix_colors[i], alpha=alpha)
            else:
                plot_helices(get_all_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=length, \
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
                plot_helices(get_nonredundant_color_group_helices(hashed_helix_dict_with_color, lhl_id=i, color_group_col='id_color'), ax, length=length, \
                    color=get_nonredundant_color_group_colors(hashed_helix_dict_with_color, color_group_col='id_color'), alpha=alpha)
            else:
                plot_helices(get_all_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=length, \
                    color=get_color_by_dict_key(hashed_helix_dict_with_color, 'id_color'), alpha=alpha)
        else:
            hashed_helix_dict_with_color = add_color_to_helix_dict(hashed_helix_dict, designs_df, continuous_metric)
            if args.plot_one_helix_per_bin:
                alpha = 0.75
                length = 5
                plot_helices(get_nonredundant_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=3, \
                    color=get_nonredundant_continuous_colors_by_dict_key(hashed_helix_dict_with_color, continuous_metric)[0], alpha=alpha)
            else:
                plot_helices(get_all_helices(hashed_helix_dict_with_color, lhl_id=i), ax, length=3, \
                    color=get_continuous_colors_by_dict_key(hashed_helix_dict_with_color, \
                        continuous_metric, min_val, max_val, cmap=cmap)[0], alpha=alpha)
        filtered_hashed_helix_dicts_list.append(hashed_helix_dict_with_color)

    # Remove the 3D axes
    ax_lims = (-15,15)
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
