"""
<<<<<<< HEAD
Ben Orr
4.12.23

read_loss_from_txt.py

Read and plot loss values from an alphafold_finetune txt file containing logged loss values.

Usage:
	read_loss_from_log.py --job_names <job_name1> [ <job_name2> ... ] --train_loss_files <loss_file1> [ <loss_file2> ... ]
		--valid_loss_files <loss_file1> [ <loss_file2> ... ]
		--outfolder <output_folder> [ --train_rolling_window <training_loss_rolling_window>
		--batch_size <batch_size> --combine_plots --reformat_valid ]

Inputs:

	job_names: Name(s) for the jobs which will be used in the output .png files. The order of job names must match that
		of log files.

	train_loss_files: Paths to alphafold_finetune train loss .txt files from which to read loss values. The order of log files must match that
		of job names.

	valid_loss_files: Paths to alphafold_finetune valid loss .txt files from which to read loss values. The order of log files must match that
		of job names.

	outfolder: Folder in which to output figures.

	train_rolling_window: Rolling window size over which to average training loss values. Default is 1.

	batch_size: Average training losses over batch_size examples, and plot losses over batches.

	combine_plots: If used, then all the loss curves (one from each job name, log file pair) will be plotted together
		in one figure.

	reformat_valid: Reformat the validation loss files, so that each validation example has its own line.


Outputs:

	Figures and data files written to the outfolder_parent/job_name.
=======
Reads, processes, and visualizes AlphaFold fine-tuning training loss curves.

Author: Ben Orr
Date: 4.12.23

This module provides tools for reading, processing, and visualizing training and
validation loss curves from AlphaFold fine-tuning experiments. It supports multiple
loss types including total loss, FAPE loss, stability (binder) loss, and protease
EC50 losses.

The module can generate individual loss curves for each training run or combine
multiple experiments into comparative plots. It supports rolling window averaging
for training losses and batch-based aggregation.

Usage
-----
Basic usage with single experiment::

    python read_loss_from_txt.py \\
        --job_names experiment_1 \\
        --train_loss_files /path/to/train_loss.txt \\
        --valid_loss_files /path/to/valid_loss.txt \\
        --outfolder /path/to/output

Multiple experiments with combined plots::

    python read_loss_from_txt.py \\
        --job_names exp1 exp2 exp3 \\
        --train_loss_files train1.txt train2.txt train3.txt \\
        --valid_loss_files valid1.txt valid2.txt valid3.txt \\
        --outfolder /path/to/output \\
        --combine_plots \\
        --train_rolling_window 100 \\
        --batch_size 8

Parameters
----------
job_names : list of str
    Name(s) for the jobs which will be used in output .png files. The order of
    job names must match that of loss files.
train_loss_files : list of str
    Paths to AlphaFold fine-tune train loss .txt files. The order must match
    that of job names.
valid_loss_files : list of str
    Paths to AlphaFold fine-tune validation loss .txt files. The order must
    match that of job names.
outfolder : str
    Directory in which to save output figures.
train_rolling_window : int, optional
    Rolling window size for averaging training loss values. Default is 1.
batch_size : int, optional
    Number of examples per batch for averaging training losses. Default is 1.
combine_plots : bool, optional
    If True, plot all loss curves together in combined figures.
reformat_valid : bool, optional
    If True, reformat validation loss files so each example has its own line.
debug : bool, optional
    If True, print additional debugging statistics.

Outputs
-------
The module generates PNG files containing loss curves:
    - Individual plots: {job_name}_{loss_type}.png
    - Combined plots: combined_plot_with_{first_job_name}_{loss_type}.png

Loss types include:
    - Total loss (loss)
    - FAPE loss (fape)
    - Stability/binder loss (binder)
    - Chymotrypsin EC50 loss (chymEC50)
    - Trypsin EC50 loss (trypEC50)

Examples
--------
Visualize a single training run with rolling average::

    python read_loss_from_txt.py \\
        --job_names my_experiment \\
        --train_loss_files train_losses.txt \\
        --valid_loss_files valid_losses.txt \\
        --outfolder ./results \\
        --train_rolling_window 50

Compare multiple experiments::

    python read_loss_from_txt.py \\
        --job_names finetune1 finetune2 \\
        --train_loss_files ft1_train.txt ft2_train.txt \\
        --valid_loss_files ft1_valid.txt ft2_valid.txt \\
        --outfolder ./comparison \\
        --combine_plots \\
        --batch_size 16
>>>>>>> be02a1e (lucs_af refactor and cleanup)
"""

import os
import argparse
<<<<<<< HEAD
=======
from typing import List, Dict, Tuple, Optional, Any
>>>>>>> be02a1e (lucs_af refactor and cleanup)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

<<<<<<< HEAD
def reformat_valid_loss_files(args):
	"""
	Reformat the validation loss files, so that each
	validation example has its own line.
=======
def reformat_valid_loss_files(args: argparse.Namespace) -> None:
	"""Reformat validation loss files to have one example per line.

	Some validation loss files may have multiple validation examples on the same
	line. This function reformats them so that each validation example starts on
	its own line with the 'test_Step: ' prefix.

	Parameters
	----------
	args : argparse.Namespace
		Parsed command line arguments containing:
		- valid_loss_files : list of str
			Paths to validation loss files to reformat.

	Returns
	-------
	None
		The function prints the reformatted content but does not write it by default.
		Uncomment the write section in the code to save the reformatted files.

	Notes
	-----
	The actual file writing is commented out by default to prevent accidental
	overwriting. Verify the output format is correct before uncommenting the
	write operation.

	Examples
	--------
	>>> import argparse
	>>> args = argparse.Namespace(valid_loss_files=['valid_loss.txt'])
	>>> reformat_valid_loss_files(args)
>>>>>>> be02a1e (lucs_af refactor and cleanup)
	"""
	for valid_file in args.valid_loss_files:
		new_file_string = ''
		with open(valid_file, 'r') as f:
			text = f.read()
			print(text)
			text_split = text.split('test_Step: ')
			for line in text_split:
				if line.strip():
					new_file_string = new_file_string + 'test_Step: ' + line + '\n'
			f.close()

		## Uncomment this only when you're sure that the output is in
		## the right format, or else the valid files will be overwritten.
		# with open(valid_file, 'w') as f:
		# 	f.write(new_file_string)

		with open(valid_file, 'r') as f:
			text = f.read()
			print(text)

<<<<<<< HEAD
def rolling_means(array, window_size):
	"""
	Return an array of rolling means
	"""
	i=0
	moving_aves=[]
	while i < len(array) - window_size + 1:
		window = array[i : i + window_size]
		window_ave = sum(window) / window_size
		moving_aves.append(window_ave)
		i+=1
	return moving_aves

def load_loss_values(args):
	"""
	Load loss values from files into a list of dicts.
=======
def rolling_means(array: List[float], window_size: int) -> List[float]:
	"""Calculate rolling (moving) average of an array.

	Computes the mean over a sliding window of specified size across the input
	array. This is useful for smoothing noisy loss curves to better visualize
	training trends.

	Parameters
	----------
	array : list of float
		Input array of numerical values (e.g., loss values).
	window_size : int
		Size of the sliding window for computing averages. Must be >= 1.

	Returns
	-------
	list of float
		Array of rolling mean values. Length is len(array) - window_size + 1.

	Notes
	-----
	- The output array is shorter than the input by (window_size - 1) elements.
	- For window_size=1, returns a copy of the input array.
	- Each output value is the mean of window_size consecutive input values.

	Examples
	--------
	>>> rolling_means([1, 2, 3, 4, 5], 3)
	[2.0, 3.0, 4.0]

	>>> rolling_means([10, 20, 30], 1)
	[10.0, 20.0, 30.0]
	"""
	i = 0
	moving_aves = []
	while i < len(array) - window_size + 1:
		window = array[i:i + window_size]
		window_ave = sum(window) / window_size
		moving_aves.append(window_ave)
		i += 1
	return moving_aves

def load_loss_values(args: argparse.Namespace) -> List[Dict[str, Any]]:
	"""Load and parse loss values from training and validation files.

	Reads training and validation loss files for multiple experiments and extracts
	all loss metrics including total loss, FAPE loss, stability (binder) loss, and
	optionally protease EC50 losses. Organizes the data into structured dictionaries
	for plotting.

	Parameters
	----------
	args : argparse.Namespace
		Parsed command line arguments containing:
		- job_names : list of str
			Names for each experiment/job.
		- train_loss_files : list of str
			Paths to training loss files.
		- valid_loss_files : list of str
			Paths to validation loss files.
		- batch_size : int
			Number of examples per batch for aggregation.

	Returns
	-------
	list of dict
		List of dictionaries, one per experiment, each containing:
		- job_name : str
			Name of the experiment.
		- train_loss_file : str
			Path to training loss file.
		- valid_loss_file : str
			Path to validation loss file.
		- epochs : list of int
			Epoch numbers for each training step.
		- steps : list of int
			Training step numbers.
		- losses : list of float
			Total loss values at each step.
		- fapes : list of float
			FAPE loss values at each step.
		- binder_losses : list of float
			Stability/binder loss values at each step.
		- batches : list of int
			Batch numbers.
		- batch_losses : list of float
			Average total loss per batch.
		- batch_fapes : list of float
			Average FAPE loss per batch.
		- batch_binder_losses : list of float
			Average binder loss per batch.
		- fapes_batches : list of tuple
			(fape_value, step) tuples excluding zero FAPE losses.
		- valid_dict : dict
			Nested dictionary with training steps as keys, containing validation
			loss arrays for each metric.
		- chym_ec50_losses : list of float, optional
			Chymotrypsin EC50 loss values (if present in logs).
		- tryp_ec50_losses : list of float, optional
			Trypsin EC50 loss values (if present in logs).
		- weighted_binder_losses : list of float, optional
			Weighted binder loss values (if present in logs).
		- weighted_chym_ec50_losses : list of float, optional
			Weighted chymotrypsin EC50 loss values (if present).
		- weighted_tryp_ec50_losses : list of float, optional
			Weighted trypsin EC50 loss values (if present).
		- pred_chym_ec50s : list of float, optional
			Predicted chymotrypsin EC50 values (if present).
		- pred_tryp_ec50s : list of float, optional
			Predicted trypsin EC50 values (if present).
		- true_chym_ec50_classes : list of float, optional
			True chymotrypsin EC50 classes (if present).
		- true_tryp_ec50_classes : list of float, optional
			True trypsin EC50 classes (if present).

	Notes
	-----
	- Files that do not exist are skipped with a warning message.
	- FAPE losses of 0 are excluded from fapes_batches as they indicate no
	  FAPE loss was backpropagated for that example.
	- Validation losses are organized by training step for easy comparison.
	- Protease EC50 losses are only included if present in the log files.

	Examples
	--------
	>>> import argparse
	>>> args = argparse.Namespace(
	...     job_names=['exp1'],
	...     train_loss_files=['train.txt'],
	...     valid_loss_files=['valid.txt'],
	...     batch_size=8
	... )
	>>> data_dicts = load_loss_values(args)
	>>> print(data_dicts[0]['job_name'])
	exp1
>>>>>>> be02a1e (lucs_af refactor and cleanup)
	"""
	data_dicts = []

	for job_name, train_file, valid_file in zip(args.job_names, args.train_loss_files, args.valid_loss_files):

<<<<<<< HEAD
		if os.path.exists(train_file) == False or os.path.exists(valid_file) == False:
=======
		if not os.path.exists(train_file) or not os.path.exists(valid_file):
>>>>>>> be02a1e (lucs_af refactor and cleanup)
			print(f'File does not exist: {train_file}: {os.path.exists(train_file)}, {valid_file}: {os.path.exists(valid_file)}')
			continue

		# Load a data dict for each job_name, containing data for loss curves
<<<<<<< HEAD
		data_dict = {'job_name':job_name, 'train_loss_file':train_file, 'valid_loss_file':valid_file}
=======
		data_dict = {'job_name': job_name, 'train_loss_file': train_file, 'valid_loss_file': valid_file}
>>>>>>> be02a1e (lucs_af refactor and cleanup)

		epochs = []
		steps = []
		losses = []
		fapes = []
<<<<<<< HEAD
		fapes_batches = [] # To exclude examples with FAPE loss = 0 (no FAPE loss backprop)
=======
		fapes_batches = []  # To exclude examples with FAPE loss = 0 (no FAPE loss backprop)
>>>>>>> be02a1e (lucs_af refactor and cleanup)
		binder_losses = []

		# Split loss values by their batches
		batches = []
		batch_losses = []
		batch_fapes = []
		batch_binder_losses = []

		# ProteaseEC50 Losses
		chym_ec50_losses = []
		tryp_ec50_losses = []
		weighted_binder_losses = []
		weighted_chym_ec50_losses = []
		weighted_tryp_ec50_losses = []
		true_chym_ec50_classes = []
		true_tryp_ec50_classes = []
		pred_chym_ec50s = []
		pred_tryp_ec50s = []

		temp_batch_loss_dict = {
<<<<<<< HEAD
			'batch_losses':[],
			'batch_fapes':[],
			'batch_binder_losses':[]
=======
			'batch_losses': [],
			'batch_fapes': [],
			'batch_binder_losses': []
>>>>>>> be02a1e (lucs_af refactor and cleanup)
		}

		step_idx = 0
		with open(train_file, 'r') as f:
			for line in f.readlines():
<<<<<<< HEAD
				line = line.replace(',','').replace('[','').replace(']','')
=======
				line = line.replace(',', '').replace('[', '').replace(']', '')
>>>>>>> be02a1e (lucs_af refactor and cleanup)
				line_split = line.split()
				epochs.append(int(line_split[1]))
				steps.append(int(line_split[1]))
				losses.append(float(line_split[3]))
				binder_losses.append(float(line_split[5]))
				fapes.append(float(line_split[9]))

				if fapes[-1] == 0:
					pass
				else:
<<<<<<< HEAD
					fapes_batches.append((fapes[-1], steps[-1])) # Includes only examples with FAPE loss != 0 (for which FAPE loss is backpropagated)
=======
					fapes_batches.append((fapes[-1], steps[-1]))  # Includes only examples with FAPE loss != 0 (for which FAPE loss is backpropagated)
>>>>>>> be02a1e (lucs_af refactor and cleanup)
					temp_batch_loss_dict['batch_fapes'].append(fapes[-1])

				# Load loss values for the current batch
				temp_batch_loss_dict['batch_losses'].append(losses[-1])
				temp_batch_loss_dict['batch_binder_losses'].append(binder_losses[-1])

				step_idx += 1
				if step_idx % args.batch_size == 0 and step_idx != 0:
					# Split loss values by their batches
<<<<<<< HEAD
					batches.append(int(step_idx/args.batch_size))
=======
					batches.append(int(step_idx / args.batch_size))
>>>>>>> be02a1e (lucs_af refactor and cleanup)
					batch_losses.append(np.mean(temp_batch_loss_dict['batch_losses']))
					batch_fapes.append(np.mean(temp_batch_loss_dict['batch_fapes']))
					batch_binder_losses.append(np.mean(temp_batch_loss_dict['batch_binder_losses']))

					# Reset the loss dict for the current batch
					temp_batch_loss_dict = {
<<<<<<< HEAD
						'batch_losses':[],
						'batch_fapes':[],
						'batch_binder_losses':[]
=======
						'batch_losses': [],
						'batch_fapes': [],
						'batch_binder_losses': []
>>>>>>> be02a1e (lucs_af refactor and cleanup)
					}

				# Protease EC50 losses
				if 'chym_ec50_loss' in line:
					chym_ec50_losses.append(float(line.split(' chym_ec50_loss:')[-1].split()[0]))
					tryp_ec50_losses.append(float(line.split(' tryp_ec50_loss:')[-1].split()[0]))
					weighted_binder_losses.append(float(line.split(' weighted_binder_loss:')[-1].split()[0]))
					weighted_chym_ec50_losses.append(float(line.split(' weighted_chym_ec50_loss:')[-1].split()[0]))
					weighted_tryp_ec50_losses.append(float(line.split(' weighted_tryp_ec50_loss:')[-1].split()[0]))
				if 'true_chym_ec50_class' in line:
					pred_chym_ec50s.append(float(line.split(' chym_ec50_pred:')[-1].split()[0].strip(',')))
					pred_tryp_ec50s.append(float(line.split(' tryp_ec50_pred:')[-1].split()[0].strip(',')))
					true_chym_ec50_classes.append(float(line.split(' true_chym_ec50_class:')[-1].split()[0].strip(',')))
					true_tryp_ec50_classes.append(float(line.split(' true_tryp_ec50_class:')[-1].split()[0].strip(',')))

		# The valid_dict will be added to the data dict
		# Keys are training steps, values are arrays of losses across
		# validation examples with the model at that training step.
		valid_dict = {}

		with open(valid_file, 'r') as f:
			for text in f:
				# This is because some of my valid loss files do not have line breaks
<<<<<<< HEAD
				lines = text.replace(',','').split('test_Step: ')
				for line in lines:
					# 4.16.24: a bug with my resume-fine-tuning code caused the valid file to contain the full training loss file at the start
					if line.startswith('Step:'): continue
					if line.strip(): # only process lines that are non-whitespace
						line = line.replace('[','').replace(']','') # remove brackets from string
=======
				lines = text.replace(',', '').split('test_Step: ')
				for line in lines:
					# 4.16.24: a bug with my resume-fine-tuning code caused the valid file to contain the full training loss file at the start
					if line.startswith('Step:'):
						continue
					if line.strip():  # only process lines that are non-whitespace
						line = line.replace('[', '').replace(']', '')  # remove brackets from string
>>>>>>> be02a1e (lucs_af refactor and cleanup)
						line_split = line.split()
						if int(line_split[0]) in valid_dict.keys():
							valid_dict[int(line_split[0])]['losses'].append(float(line_split[3].strip(',')))
							valid_dict[int(line_split[0])]['binder_losses'].append(float(line_split[5].strip(',')))
							valid_dict[int(line_split[0])]['fapes'].append(float(line_split[9].strip(',')))

							# Exclude validation FAPE losses = 0 (no FAPE loss backprop). In valid_dics, fapes_batches only
							# contains fape values (as valid loss is plotted as average over valid epoch)
							if valid_dict[int(line_split[0])]['fapes'][-1] != 0:
								valid_dict[int(line_split[0])]['fapes_batches'].append(float(line_split[9].strip(',')))

							# Protease EC50 losses
							if 'chym_ec50_loss' in line:
								valid_dict[int(line_split[0])]['chym_ec50_losses'].append(float(line.split(' chym_ec50_loss:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['tryp_ec50_losses'].append(float(line.split(' tryp_ec50_loss:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['weighted_binder_losses'].append(float(line.split(' weighted_binder_loss:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['weighted_chym_ec50_losses'].append(float(line.split(' weighted_chym_ec50_loss:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['weighted_tryp_ec50_losses'].append(float(line.split(' weighted_tryp_ec50_loss:')[-1].split()[0].strip(',')))
							if 'true_chym_ec50_class' in line:
								valid_dict[int(line_split[0])]['pred_chym_ec50s'].append(float(line.split(' chym_ec50_pred:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['pred_tryp_ec50s'].append(float(line.split(' tryp_ec50_pred:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['true_chym_ec50_classes'].append(float(line.split(' true_chym_ec50_class:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['true_tryp_ec50_classes'].append(float(line.split(' true_chym_ec50_class:')[-1].split()[0].strip(',')))

						else:
<<<<<<< HEAD
							new_dict = {'losses':[], 'fapes':[], 'binder_losses':[], 'chym_ec50_losses':[], 'tryp_ec50_losses':[],
										'weighted_binder_losses':[], 'weighted_chym_ec50_losses':[], 'weighted_tryp_ec50_losses':[],
										'fapes_batches':[], 'pred_chym_ec50s':[], 'pred_tryp_ec50s':[], 'true_chym_ec50_classes':[],
										'true_tryp_ec50_classes':[]}
=======
							new_dict = {
								'losses': [], 'fapes': [], 'binder_losses': [], 'chym_ec50_losses': [], 'tryp_ec50_losses': [],
								'weighted_binder_losses': [], 'weighted_chym_ec50_losses': [], 'weighted_tryp_ec50_losses': [],
								'fapes_batches': [], 'pred_chym_ec50s': [], 'pred_tryp_ec50s': [], 'true_chym_ec50_classes': [],
								'true_tryp_ec50_classes': []
							}
>>>>>>> be02a1e (lucs_af refactor and cleanup)
							valid_dict[int(line_split[0])] = new_dict
							valid_dict[int(line_split[0])]['losses'].append(float(line_split[3].strip(',')))
							valid_dict[int(line_split[0])]['fapes'].append(float(line_split[9].strip(',')))
							valid_dict[int(line_split[0])]['binder_losses'].append(float(line_split[5].strip(',')))

							# Exclude validation FAPE losses = 0 (no FAPE loss backprop). In valid_dics, fapes_batches only
							# contains fape values (as valid loss is plotted as average over valid epoch)
							if valid_dict[int(line_split[0])]['fapes'][-1] != 0:
								valid_dict[int(line_split[0])]['fapes_batches'].append(float(line_split[9].strip(',')))

							# Protease EC50 losses
							if 'chym_ec50_loss' in line:
								valid_dict[int(line_split[0])]['chym_ec50_losses'].append(float(line.split(' chym_ec50_loss:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['tryp_ec50_losses'].append(float(line.split(' tryp_ec50_loss:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['weighted_binder_losses'].append(float(line.split(' weighted_binder_loss:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['weighted_chym_ec50_losses'].append(float(line.split(' weighted_chym_ec50_loss:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['weighted_tryp_ec50_losses'].append(float(line.split(' weighted_tryp_ec50_loss:')[-1].split()[0].strip(',')))
							if 'true_chym_ec50_class' in line:
								valid_dict[int(line_split[0])]['pred_chym_ec50s'].append(float(line.split(' chym_ec50_pred:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['pred_tryp_ec50s'].append(float(line.split(' tryp_ec50_pred:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['true_chym_ec50_classes'].append(float(line.split(' true_chym_ec50_class:')[-1].split()[0].strip(',')))
								valid_dict[int(line_split[0])]['true_tryp_ec50_classes'].append(float(line.split(' true_chym_ec50_class:')[-1].split()[0].strip(',')))


		data_dict['epochs'] = epochs
		data_dict['steps'] = steps
		data_dict['losses'] = losses
		data_dict['fapes'] = fapes
		data_dict['binder_losses'] = binder_losses
		data_dict['valid_dict'] = valid_dict
		data_dict['batches'] = batches
		data_dict['batch_losses'] = batch_losses
		data_dict['batch_fapes'] = batch_fapes
		data_dict['batch_binder_losses'] = batch_binder_losses

		# Excludes FAPE = 0 samples.
		data_dict['fapes_batches'] = fapes_batches

		if 'chym_ec50_loss' in line:
			# Protease EC50 losses
			data_dict['chym_ec50_losses'] = chym_ec50_losses
			data_dict['tryp_ec50_losses'] = tryp_ec50_losses
			data_dict['weighted_binder_losses'] = weighted_binder_losses
			data_dict['weighted_chym_ec50_losses'] = weighted_chym_ec50_losses
			data_dict['weighted_tryp_ec50_losses'] = weighted_tryp_ec50_losses
		if 'true_chym_ec50_class' in line:
			data_dict['pred_chym_ec50s'] = pred_chym_ec50s
			data_dict['pred_tryp_ec50s'] = pred_tryp_ec50s
			data_dict['true_chym_ec50_classes'] = true_chym_ec50_classes
			data_dict['true_tryp_ec50_classes'] = true_tryp_ec50_classes

		data_dicts.append(data_dict)

	return data_dicts

<<<<<<< HEAD
def print_min_loss(name, vals):
=======
def print_min_loss(name: str, vals: List[float]) -> None:
	"""Print minimum and maximum values with their indices for debugging.

	Helper function to display summary statistics of loss arrays during debugging.

	Parameters
	----------
	name : str
		Name of the loss metric being analyzed.
	vals : list of float
		Array of loss values.

	Returns
	-------
	None
		Prints statistics to stdout.

	Examples
	--------
	>>> print_min_loss('FAPE', [1.5, 0.8, 1.2, 0.5, 1.0])
	Min FAPE: 0.5
	Min Index FAPE: 3
	Max FAPE: 1.5
	Max Index FAPE: 0
	"""
>>>>>>> be02a1e (lucs_af refactor and cleanup)
	print(f'Min {name}: {min(vals)}')
	print(f'Min Index {name}: {vals.index(min(vals))}')
	print(f'Max {name}: {max(vals)}')
	print(f'Max Index {name}: {vals.index(max(vals))}')

<<<<<<< HEAD
def plot_loss_curve(args, data_dict,
		x='steps', y='losses', xlabel='Batch', ylabel='Loss',
		title='Total loss curve',
		outfile='',
		alpha=1):

	fig = plt.figure(figsize=(8,5))
=======
def plot_loss_curve(
		args: argparse.Namespace,
		data_dict: Dict[str, Any],
		x: str = 'steps',
		y: str = 'losses',
		xlabel: str = 'Batch',
		ylabel: str = 'Loss',
		title: str = 'Total loss curve',
		outfile: str = '',
		alpha: float = 1) -> None:
	"""Plot training and validation loss curves for a single experiment.

	Generates a line plot showing both training and validation loss over time.
	Training loss is displayed as a solid line (optionally smoothed with rolling
	average) and validation loss as a dashed line.

	Parameters
	----------
	args : argparse.Namespace
		Parsed command line arguments containing:
		- train_rolling_window : int
			Window size for smoothing training loss.
		- batch_size : int
			Batch size for normalizing x-axis when plotting by batches.
		- debug : bool
			If True, print debugging statistics.
	data_dict : dict
		Dictionary containing loss data for one experiment, as returned by
		load_loss_values(). Must contain keys matching the x and y parameters.
	x : str, optional
		Key for x-axis data. Options: 'steps', 'batches', or 'fapes_batches'.
		Default is 'steps'.
	y : str, optional
		Key for y-axis data. Options: 'losses', 'fapes', 'binder_losses',
		'batch_losses', 'batch_fapes', 'batch_binder_losses', 'fapes_batches',
		'chym_ec50_losses', 'tryp_ec50_losses'. Default is 'losses'.
	xlabel : str, optional
		Label for x-axis. Default is 'Batch'.
	ylabel : str, optional
		Label for y-axis. Default is 'Loss'.
	title : str, optional
		Plot title. Default is 'Total loss curve'.
	outfile : str, optional
		Path where the figure will be saved. Default is ''.
	alpha : float, optional
		Transparency of plot lines (0-1). Default is 1.

	Returns
	-------
	None
		Saves the figure to the specified outfile path.

	Notes
	-----
	- Training loss is smoothed using a rolling window if train_rolling_window > 1.
	- Validation loss is plotted as the mean across validation examples at each step.
	- For 'fapes_batches', only non-zero FAPE values are plotted.
	- When x='batches', validation steps are divided by batch_size for alignment.

	Examples
	--------
	>>> plot_loss_curve(
	...     args, data_dict,
	...     x='steps', y='losses',
	...     xlabel='Training Step', ylabel='Total Loss',
	...     title='Experiment Loss Curve',
	...     outfile='./output/loss.png'
	... )
	Saved figure: ./output/loss.png
	"""
	fig = plt.figure(figsize=(8, 5))
>>>>>>> be02a1e (lucs_af refactor and cleanup)
	ax = plt.subplot(111)

	if args.debug:
		print_min_loss(y, data_dict[y])

		print(f'Length of {x} for {data_dict["job_name"]}:')
		print(len(data_dict[x]))
		print(f'Length of {x} for {data_dict["job_name"]} with rolling window {args.train_rolling_window}:')
		print(len(rolling_means(data_dict[x], args.train_rolling_window)))

	if y == 'fapes_batches':
<<<<<<< HEAD
		train_x = rolling_means(np.array(data_dict[y])[:,1], args.train_rolling_window)
		train_y = rolling_means(np.array(data_dict[y])[:,0], args.train_rolling_window)
=======
		train_x = rolling_means(np.array(data_dict[y])[:, 1], args.train_rolling_window)
		train_y = rolling_means(np.array(data_dict[y])[:, 0], args.train_rolling_window)
>>>>>>> be02a1e (lucs_af refactor and cleanup)
		valid_x = data_dict['valid_dict'].keys()
		valid_y = [np.mean(data_dict['valid_dict'][k][y]) for k in data_dict['valid_dict'].keys()]
	elif x == 'batches':
		train_x = rolling_means(data_dict[x], args.train_rolling_window)
		train_y = rolling_means(data_dict[y], args.train_rolling_window)
		# Divide the valid dict keys (training step) by the batch size
<<<<<<< HEAD
		valid_x = [val/args.batch_size for val in data_dict['valid_dict'].keys()]
		valid_y = [np.mean(data_dict['valid_dict'][k][y.replace('batch_','')]) for k in data_dict['valid_dict'].keys()]
=======
		valid_x = [val / args.batch_size for val in data_dict['valid_dict'].keys()]
		valid_y = [np.mean(data_dict['valid_dict'][k][y.replace('batch_', '')]) for k in data_dict['valid_dict'].keys()]
>>>>>>> be02a1e (lucs_af refactor and cleanup)
	else:
		train_x = rolling_means(data_dict[x], args.train_rolling_window)
		train_y = rolling_means(data_dict[y], args.train_rolling_window)
		valid_x = data_dict['valid_dict'].keys()
		valid_y = [np.mean(data_dict['valid_dict'][k][y]) for k in data_dict['valid_dict'].keys()]

	## Plot a rolling mean of the training loss
	ax.plot(train_x, train_y, label='train')

	## Plot the validation loss
	ax.plot(valid_x, valid_y, label='validation', linestyle='dashed')

<<<<<<< HEAD
	
=======
>>>>>>> be02a1e (lucs_af refactor and cleanup)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend()
	plt.savefig(outfile)
	print(f'Saved figure: {outfile}')
	plt.close()

<<<<<<< HEAD
def plot_combined_loss_curve(args, data_dicts,
		x='steps', y='losses', xlabel='Batch', ylabel='Loss', 
		title='Total loss curve',
		outfile='', 
		alpha=0.1,
		cmap='Paired'):

	# cmap = matplotlib.cm.get_cmap(cmap)
	cmap = matplotlib.colormaps[cmap]

	fig = plt.figure(figsize=(8,5))
=======
def plot_combined_loss_curve(
		args: argparse.Namespace,
		data_dicts: List[Dict[str, Any]],
		x: str = 'steps',
		y: str = 'losses',
		xlabel: str = 'Batch',
		ylabel: str = 'Loss',
		title: str = 'Total loss curve',
		outfile: str = '',
		alpha: float = 0.1,
		cmap: str = 'Paired') -> None:
	"""Plot combined training and validation loss curves from multiple experiments.

	Generates a single plot overlaying loss curves from multiple experiments for
	direct comparison. Each experiment gets a unique color from the colormap, with
	training losses as solid lines and validation losses as dashed lines.

	Parameters
	----------
	args : argparse.Namespace
		Parsed command line arguments containing:
		- train_rolling_window : int
			Window size for smoothing training loss.
		- batch_size : int
			Batch size for normalizing x-axis when plotting by batches.
	data_dicts : list of dict
		List of dictionaries containing loss data for each experiment, as
		returned by load_loss_values().
	x : str, optional
		Key for x-axis data. Options: 'steps', 'batches', or 'fapes_batches'.
		Default is 'steps'.
	y : str, optional
		Key for y-axis data. Options: 'losses', 'fapes', 'binder_losses',
		'batch_losses', 'batch_fapes', 'batch_binder_losses', 'fapes_batches',
		'chym_ec50_losses', 'tryp_ec50_losses'. Default is 'losses'.
	xlabel : str, optional
		Label for x-axis. Default is 'Batch'.
	ylabel : str, optional
		Label for y-axis. Default is 'Loss'.
	title : str, optional
		Plot title. Default is 'Total loss curve'.
	outfile : str, optional
		Path where the figure will be saved. Default is ''.
	alpha : float, optional
		Transparency of plot lines (0-1). Lower values help distinguish
		overlapping curves. Default is 0.1.
	cmap : str, optional
		Matplotlib colormap name for coloring different experiments.
		Default is 'Paired'.

	Returns
	-------
	None
		Saves the figure to the specified outfile path at 300 DPI.

	Notes
	-----
	- Colors are assigned sequentially from the colormap: experiment i gets
	  colors cmap(2*i) for training and cmap(2*i+1) for validation.
	- Experiments missing the requested loss type are skipped automatically.
	- Legend font size is reduced to accommodate multiple entries.
	- Figure uses tight layout to prevent label cutoff.

	Examples
	--------
	>>> plot_combined_loss_curve(
	...     args, data_dicts,
	...     x='steps', y='losses',
	...     xlabel='Training Step', ylabel='Total Loss',
	...     title='Multi-Experiment Comparison',
	...     outfile='./output/combined_loss.png',
	...     alpha=0.3
	... )
	Saved figure: ./output/combined_loss.png
	"""
	# cmap = matplotlib.cm.get_cmap(cmap)
	cmap = matplotlib.colormaps[cmap]

	fig = plt.figure(figsize=(8, 5))
>>>>>>> be02a1e (lucs_af refactor and cleanup)
	ax = plt.subplot(111)
	for i in range(len(data_dicts)):

		data_dict = data_dicts[i]
		## Plot the training loss
		# ax.plot(data_dict['batches'], data_dict['losses'], label=data_dict['job_name']+'_train')

		if y not in data_dict.keys():
			continue
<<<<<<< HEAD
		
		if y == 'fapes_batches':
			train_x = rolling_means(np.array(data_dict[y])[:,1], args.train_rolling_window)
			train_y = rolling_means(np.array(data_dict[y])[:,0], args.train_rolling_window)
=======

		if y == 'fapes_batches':
			train_x = rolling_means(np.array(data_dict[y])[:, 1], args.train_rolling_window)
			train_y = rolling_means(np.array(data_dict[y])[:, 0], args.train_rolling_window)
>>>>>>> be02a1e (lucs_af refactor and cleanup)
			valid_x = data_dict['valid_dict'].keys()
			valid_y = [np.mean(data_dict['valid_dict'][k][y]) for k in data_dict['valid_dict'].keys()]
		elif x == 'batches':
			train_x = rolling_means(data_dict[x], args.train_rolling_window)
			train_y = rolling_means(data_dict[y], args.train_rolling_window)
			# Divide the valid dict keys (training step) by the batch size
<<<<<<< HEAD
			valid_x = [val/args.batch_size for val in data_dict['valid_dict'].keys()]
			valid_y = [np.mean(data_dict['valid_dict'][k][y.replace('batch_','')]) for k in data_dict['valid_dict'].keys()]
=======
			valid_x = [val / args.batch_size for val in data_dict['valid_dict'].keys()]
			valid_y = [np.mean(data_dict['valid_dict'][k][y.replace('batch_', '')]) for k in data_dict['valid_dict'].keys()]
>>>>>>> be02a1e (lucs_af refactor and cleanup)
		else:
			train_x = rolling_means(data_dict[x], args.train_rolling_window)
			train_y = rolling_means(data_dict[y], args.train_rolling_window)
			valid_x = data_dict['valid_dict'].keys()
			valid_y = [np.mean(data_dict['valid_dict'][k][y]) for k in data_dict['valid_dict'].keys()]

		## Plot a rolling mean of the training loss
<<<<<<< HEAD
		ax.plot(train_x, train_y, label=data_dict['job_name']+'_train', color=cmap(2*i), alpha=alpha)

		## Plot the validation loss
		ax.plot(valid_x, valid_y, label=data_dict['job_name']+'_valid', color=cmap(2*i+1), linestyle='dashed', alpha=alpha)
=======
		ax.plot(train_x, train_y, label=data_dict['job_name'] + '_train', color=cmap(2 * i), alpha=alpha)

		## Plot the validation loss
		ax.plot(valid_x, valid_y, label=data_dict['job_name'] + '_valid', color=cmap(2 * i + 1), linestyle='dashed', alpha=alpha)
>>>>>>> be02a1e (lucs_af refactor and cleanup)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend(fontsize="6")
	# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.tight_layout()
	plt.savefig(outfile, dpi=300)
	print(f'Saved figure: {outfile}')
	plt.close()


<<<<<<< HEAD
if __name__ == '__main__':

	# Parse command line arguments.
=======
def plot_individual_loss_curves(args: argparse.Namespace, data_dicts: List[Dict[str, Any]]) -> None:
	"""Plot individual loss curves for each experiment."""
	for data_dict in data_dicts:
		# Determine title append text based on batch size
		if args.batch_size == 1:
			title_append = f' Train rolling window = {args.train_rolling_window}'
		else:
			title_append = f' Batch Size = {args.batch_size}'

		# Plot Total Loss (by step)
		plot_loss_curve(args, data_dict, x='steps', y='losses', xlabel='Step', ylabel='Loss',
			title=f'{data_dict["job_name"]}\nTotal loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name'] + '_loss_step.png'), alpha=1)

		# Plot FAPE Loss (by step)
		plot_loss_curve(args, data_dict, x='steps', y='fapes_batches', xlabel='Step', ylabel='FAPE Loss',
			title=data_dict['job_name'] + f'\nFAPE loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name'] + '_fape_step.png'), alpha=1)

		# Plot Stability (Binder) Loss (by step)
		plot_loss_curve(args, data_dict, x='steps', y='binder_losses', xlabel='Step', ylabel='Stability Loss',
			title=data_dict['job_name'] + f'\nStability loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name'] + '_binder_step.png'), alpha=1)

		# Plot EC50 losses if available
		if 'chym_ec50_losses' in data_dict.keys():
			plot_loss_curve(args, data_dict, x='steps', y='chym_ec50_losses', xlabel='Step', ylabel='Chym EC50 Loss',
				title=data_dict['job_name'] + f'\nChym EC50 loss curve.{title_append}',
				outfile=os.path.join(args.outfolder, data_dict['job_name'] + '_chymEC50_step.png'), alpha=1)

		if 'tryp_ec50_losses' in data_dict.keys():
			plot_loss_curve(args, data_dict, x='steps', y='tryp_ec50_losses', xlabel='Step', ylabel='Tryp EC50 Loss',
				title=data_dict['job_name'] + f'\nTryp EC50 loss curve.{title_append}',
				outfile=os.path.join(args.outfolder, data_dict['job_name'] + '_trypEC50_step.png'), alpha=1)

		# Plot loss curves over batches
		plot_loss_curve(args, data_dict, x='batches', y='batch_losses', xlabel='Batch', ylabel='Loss',
			title=f'{data_dict["job_name"]}\nTotal loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name'] + '_loss.png'), alpha=1)

		plot_loss_curve(args, data_dict, x='batches', y='batch_fapes', xlabel='Batch', ylabel='FAPE Loss',
			title=data_dict['job_name'] + f'\nFAPE loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name'] + '_fape.png'), alpha=1)

		plot_loss_curve(args, data_dict, x='batches', y='batch_binder_losses', xlabel='Batch', ylabel='Stability Loss',
			title=data_dict['job_name'] + f'\nStability loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name'] + '_binder.png'), alpha=1)


def plot_combined_loss_curves(args: argparse.Namespace, data_dicts: List[Dict[str, Any]]) -> None:
	"""Plot combined loss curves from multiple experiments."""
	# Calculate alpha based on number of experiments
	alpha = 1 - 0.08 * len(data_dicts)

	# Determine title append text
	if args.batch_size == 1:
		title_append = f' Train rolling window = {args.train_rolling_window}'
	else:
		title_append = f' Batch Size = {args.batch_size}'

	# Plot combined curves (by step)
	plot_combined_loss_curve(args, data_dicts, x='steps', y='losses', xlabel='Batch', ylabel='Loss',
		title=f'Total loss curve.{title_append}',
		outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_loss.png'), alpha=alpha)

	plot_combined_loss_curve(args, data_dicts, x='steps', y='fapes_batches', xlabel='Step', ylabel='FAPE Loss',
		title=f'FAPE loss curve.{title_append}',
		outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_fape_steps.png'), alpha=alpha)

	plot_combined_loss_curve(args, data_dicts, x='steps', y='binder_losses', xlabel='Step', ylabel='Stability Loss',
		title=f'Stability loss curve.{title_append}',
		outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_binder_steps.png'), alpha=alpha)

	plot_combined_loss_curve(args, data_dicts, x='steps', y='chym_ec50_losses', xlabel='Step', ylabel='Chym EC50 Loss',
		title=f'Chym EC50 loss curve.{title_append}',
		outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_chymEC50_steps.png'), alpha=alpha)

	plot_combined_loss_curve(args, data_dicts, x='steps', y='tryp_ec50_losses', xlabel='Step', ylabel='Tryp EC50 Loss',
		title=f'Tryp EC50 loss curve.{title_append}',
		outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_trypEC50_steps.png'), alpha=alpha)

	# Plot combined curves (by batch)
	plot_combined_loss_curve(args, data_dicts, x='batches', y='batch_losses', xlabel='Batch', ylabel='Loss',
		title=f'Total loss curve.{title_append}',
		outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_loss.png'), alpha=alpha)

	plot_combined_loss_curve(args, data_dicts, x='batches', y='batch_fapes', xlabel='Batch', ylabel='FAPE Loss',
		title=f'FAPE loss curve.{title_append}',
		outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_fape.png'), alpha=alpha)

	plot_combined_loss_curve(args, data_dicts, x='batches', y='batch_binder_losses', xlabel='Batch', ylabel='Stability Loss',
		title=f'Stability loss curve.{title_append}',
		outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_binder.png'), alpha=alpha)


def main():
	"""Main entry point for loss curve visualization."""
	# Parse command line arguments
>>>>>>> be02a1e (lucs_af refactor and cleanup)
	parser = argparse.ArgumentParser(
		description='Analyze RMSDs between classic and fine-tuned AF2 predictions to their ground truth structures,\
					as well as the classic and fine-tuned AF2 confidence metrics.',
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('--job_names', nargs='*', type=str, required=True,
			help='Name(s) for the jobs which will be used in the output .png files. The order of job names must\
			match that of log files.')
	parser.add_argument('--train_loss_files', nargs='*', type=str, required=True,
			help='alphafold_finetune train loss .txt files from which to read loss values. The order of log files must match\
			that of job names.')
	parser.add_argument('--valid_loss_files', nargs='*', type=str, required=True,
			help='alphafold_finetune valid loss .txt files from which to read loss values. The order of log files must match\
			that of job names.')
	parser.add_argument('--outfolder', type=str, required=True,
			help='Folder in which to output figures.')
	parser.add_argument('--train_rolling_window', type=int, required=False, default=1,
			help='Rolling window size over which to average training loss values. Default is 1.')
	parser.add_argument('--batch_size', type=int, required=False, default=1,
			help='Average training losses over batch_size examples, and plot losses over batches.')
	parser.add_argument('--combine_plots', action='store_true',
			help='If used, then all the loss curves (one from each job name, log file pair) will be plotted\
			together in one figure.')
	parser.add_argument('--reformat_valid', action='store_true',
			help='Reformat the validation loss files, so that each validation example has its own line.')
	parser.add_argument('--debug', action='store_true',
			help='Print loss stats for debugging.')

	args = parser.parse_args()

<<<<<<< HEAD
=======
	# Validate inputs
>>>>>>> be02a1e (lucs_af refactor and cleanup)
	if len(args.job_names) != len(args.train_loss_files) or len(args.job_names) != len(args.valid_loss_files):
		print('\n\nWARNING: Number of provided job names does not match that of provided log files.\n\
			Extra job name(s) or log file(s) will be ignored.\n\n')

<<<<<<< HEAD
	os.makedirs(args.outfolder, exist_ok=True)

	if args.reformat_valid:
		reformat_valid_loss_files(args)

	data_dicts = load_loss_values(args)

	### For plotting individual loss and validation curves

	## Ensure that the lowest validation loss values correspond to the saved best_models
	# for data_dict in data_dicts:
	# 	x = list(data_dict['valid_dict'].keys())
	# 	y = [np.mean(data_dict['valid_dict'][k]['losses']) for k in data_dict['valid_dict'].keys()]

	# 	min_i = y.index(min(y))
	# 	print('Min loss value')
	# 	print(min(y))
	# 	print('Min validation batch')
	# 	print(x[min_i])
	# 	print('All batches and losses:')
	# 	print('\n'.join(['%d, %.3f'%(b, l) for b, l in zip(x, y)]))


	for data_dict in data_dicts:

		if args.batch_size == 1:
			title_append = f' Train rolling window = {args.train_rolling_window}'
		else:
			title_append = f' Batch Size = {args.batch_size}'

		# Total Loss Plot
		plot_loss_curve(args, data_dict, x='steps', y='losses', xlabel='Step', ylabel='Loss', title=f'{data_dict["job_name"]}\nTotal loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name']+'_loss_step.png'), alpha=1)

		# FAPE Loss Plot
		# plot_loss_curve(args, data_dict, x='batches', y='fapes', xlabel='Batch', ylabel='FAPE Loss', title=data_dict['job_name']+'\nFAPE loss curve',
		# 	outfile=os.path.join(args.outfolder, data_dict['job_name']+'_fape.png'), alpha=1)
		plot_loss_curve(args, data_dict, x='steps', y='fapes_batches', xlabel='Step', ylabel='FAPE Loss', title=data_dict['job_name']+f'\nFAPE loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name']+'_fape_step.png'), alpha=1)

		# Stability (Binder) Loss Plot
		plot_loss_curve(args, data_dict, x='steps', y='binder_losses', xlabel='Step', ylabel='Stability Loss', title=data_dict['job_name']+f'\nStability loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name']+'_binder_step.png'), alpha=1)

		if 'chym_ec50_losses' in data_dict.keys():
			# Chym EC50 Loss Plot
			plot_loss_curve(args, data_dict, x='steps', y='chym_ec50_losses', xlabel='Step', ylabel='Chym EC50 Loss', title=data_dict['job_name']+f'\nChym EC50 loss curve.{title_append}',
				outfile=os.path.join(args.outfolder, data_dict['job_name']+'_chymEC50_step.png'), alpha=1)
		
		if 'tryp_ec50_losses' in data_dict.keys():
			# Tryp EC50 Loss Plot
			plot_loss_curve(args, data_dict, x='steps', y='tryp_ec50_losses', xlabel='Step', ylabel='Tryp EC50 Loss', title=data_dict['job_name']+f'\nTryp EC50 loss curve.{title_append}',
				outfile=os.path.join(args.outfolder, data_dict['job_name']+'_trypEC50_step.png'), alpha=1)

		## Plot loss curves over batches instead of training steps:
		# Batches: Total Loss Plot
		plot_loss_curve(args, data_dict, x='batches', y='batch_losses', xlabel='Batch', ylabel='Loss', title=f'{data_dict["job_name"]}\nTotal loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name']+'_loss.png'), alpha=1)

		# Batches: FAPE Loss Plot
		# plot_loss_curve(args, data_dict, x='batches', y='fapes', xlabel='Batch', ylabel='FAPE Loss', title=data_dict['job_name']+'\nFAPE loss curve',
		# 	outfile=os.path.join(args.outfolder, data_dict['job_name']+'_fape.png'), alpha=1)
		plot_loss_curve(args, data_dict, x='batches', y='batch_fapes', xlabel='Batch', ylabel='FAPE Loss', title=data_dict['job_name']+f'\nFAPE loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name']+'_fape.png'), alpha=1)

		# Batches: Stability (Binder) Loss Plot
		plot_loss_curve(args, data_dict, x='batches', y='batch_binder_losses', xlabel='Batch', ylabel='Stability Loss', title=data_dict['job_name']+f'\nStability loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, data_dict['job_name']+'_binder.png'), alpha=1)


		## TO DO:
		## Plot confusion matrix for EC50 class predictions.
		## Use: 'pred_chym_ec50s':[], 'pred_tryp_ec50s':[], 'true_chym_ec50_classes':[],'true_tryp_ec50_classes':[]}
		# if 'true_chym_ec50_classes' in data_dict.keys():
		# 	plot_pred_classes(args, data_dict, x='batches', y='tryp_ec50_losses', xlabel='Batch', ylabel='Tryp EC50 Loss', title=data_dict['job_name']+'\nTryp EC50 loss curve',
		# 		outfile=os.path.join(args.outfolder, data_dict['job_name']+'_trypEC50.png'), alpha=1)

		# if 'true_tryp_ec50_classes' in data_dict.keys():
		# 	plot_pred_classes(args, data_dict, x='batches', y='tryp_ec50_losses', xlabel='Batch', ylabel='Tryp EC50 Loss', title=data_dict['job_name']+'\nTryp EC50 loss curve',
		# 		outfile=os.path.join(args.outfolder, data_dict['job_name']+'_trypEC50.png'), alpha=1)

	
	# Combine multiple loss and validation curves into one figure.
	if args.combine_plots:

		alpha = 1 - 0.08 * len(data_dicts)
		# alpha = 0.5

		if args.batch_size == 1:
			title_append = f' Train rolling window = {args.train_rolling_window}'
		else:
			title_append = f' Batch Size = {args.batch_size}'

		# Total Loss
		plot_combined_loss_curve(args, data_dicts, x='steps', y='losses', xlabel='Batch', ylabel='Loss', title=f'Total loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_loss.png'), alpha=alpha)

		# FAPE Loss
		# plot_combined_loss_curve(args, data_dicts, x='batches', y='fapes', xlabel='Batch', ylabel='FAPE Loss', title='FAPE loss curve',
		# 	outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_fape.png'), alpha=alpha)
		plot_combined_loss_curve(args, data_dicts, x='steps', y='fapes_batches', xlabel='Step', ylabel='FAPE Loss', title=f'FAPE loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_fape_steps.png'), alpha=alpha)

		# Stability (Binder) Loss
		plot_combined_loss_curve(args, data_dicts, x='steps', y='binder_losses', xlabel='Step', ylabel='Stability Loss', title=f'Stability loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_binder_steps.png'), alpha=alpha)

		# Chym EC50 Loss
		plot_combined_loss_curve(args, data_dicts, x='steps', y='chym_ec50_losses', xlabel='Step', ylabel='Chym EC50 Loss', title=f'Chym EC50 loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_chymEC50_steps.png'), alpha=alpha)

		# Tryp EC50 Loss
		plot_combined_loss_curve(args, data_dicts, x='steps', y='tryp_ec50_losses', xlabel='Step', ylabel='Tryp EC50 Loss', title=f'Tryp EC50 loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_trypEC50_steps.png'), alpha=alpha)

		## Plot loss curves over batches instead of training steps:
		# Batches: Total Loss
		plot_combined_loss_curve(args, data_dicts, x='batches', y='batch_losses', xlabel='Batch', ylabel='Loss', title=f'Total loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_loss.png'), alpha=alpha)

		# Batches: FAPE Loss
		plot_combined_loss_curve(args, data_dicts, x='batches', y='batch_fapes', xlabel='Batch', ylabel='FAPE Loss', title=f'FAPE loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_fape.png'), alpha=alpha)

		# Batches: Stability (Binder) Loss
		plot_combined_loss_curve(args, data_dicts, x='batches', y='batch_binder_losses', xlabel='Batch', ylabel='Stability Loss', title=f'Stability loss curve.{title_append}',
			outfile=os.path.join(args.outfolder, f'combined_plot_with_{args.job_names[0]}_binder.png'), alpha=alpha)
=======
	# Create output directory
	os.makedirs(args.outfolder, exist_ok=True)

	# Reformat validation files if requested
	if args.reformat_valid:
		reformat_valid_loss_files(args)

	# Load loss data from files
	data_dicts = load_loss_values(args)

	# Plot individual loss curves for each experiment
	plot_individual_loss_curves(args, data_dicts)

	# Plot combined loss curves if requested
	if args.combine_plots:
		plot_combined_loss_curves(args, data_dicts)


if __name__ == '__main__':
	main()
>>>>>>> be02a1e (lucs_af refactor and cleanup)


