"""
<<<<<<< HEAD
Ben Orr
4.4.23

analyze_checkpointed_models_metrics.py

Analyze RMSDs between classic and fine-tuned AF2 predictions to their ground truth structures, as well as the
classic and fine-tuned AF2 confidence metrics.

Usage:
	analyze_checkpointed_models_metrics.py --job_name <job_name> --checkpoint_model_data_folder <checkpoint_model_data_folder>
		--outfolder_parent <output_folder_parent> [ --rmsd_analysis --stability_data <stability_dataframe1> [ <stability_dataframe1> ... ]
		--rmsd_thresh <rmsd_threshold> --plot_rmsd_distributions --reshaped_helix_rmsd --stability_classifier 
		--plddt_thresh <pLDDT_threshold> --pae_thresh <pAE_threshold> 
		--plot_confidence_distributions --precision_recall --reshaped_residue_stability]

Inputs:

	job_name: Name for the job which will be used in the output .png files.

	checkpoint_model_data_folder: Folder containing dataframe(s) output by calc_checkpointed_models_metrics.py.

	outfolder_parent: Output files are written to outfolder_parent.

	rmsd_analysis: Analyze the RMSDs to the ground truth structures for each checkpointed model's predictions.

	stability_data: Dataframe(s) containing a 'stability_label' column with 'stable,' 'unstable,' and 'uncertain' values,
		as well as a 'sequence' column, which is used to get the sequence length.

	rmsd_thresh: Threshold for AF2 to ground truth RMSD values for plotting the fraction of designs predicted to
			sub-<rmsd_thresh> RMSD to the ground truth structures.

	plot_rmsd_distributions: Plot distributions of RMSD values for stable vs. unstable designs.

	reshaped_helix_rmsd: Plot Reshaped Helix RMSD to Rosetta values for LUCS designs.

	stability_classifier: Analyze the Precision and Recall of the AF2 confidence scores in predicting designs as stable
		and unstable.  Uses the default pLDDT and pAE thresholds or the provided thresholds.

	plddt_thresh: Threshold for AF2 pLDDT values in classifying a design as stable (if the AF2 average pLDDT for a design 
		is greater than this threshold, then the design is classified as "stable"). Default is 0.80367635 (taken from the
		alphafold_finetune github repo: https://github.com/phbradley/alphafold_finetune).

	pae_thresh: Threshold for AF2 pAE values in classifying a design as stable (if the AF2 average pAE for a design 
		is greater than this threshold, then the design is classified as "stable"). Default is 0.43373787 (taken from the
		alphafold_finetune github repo: https://github.com/phbradley/alphafold_finetune).

	plot_confidence_distributions: Plot the distributions of the confidence scores for the classic and fine-tuned models
		at each training step.  Requires a outfolder_parent/job_name/plotting_dfs folder to exist with outputs from the 
		--stability_classifier outputs.

	precision_recall: Generate Precision vs. Recall curves for classifying stable designs at different pLDDT and 
		pAE thresholds.

	reshaped_residue_stability: Analyze the Precision and Recall of the AF2 confidence scores in predicting designs as stable
			and unstable using the pAE values between the bb_remodeled_residues and non-bb_remodeled_residues and
			pLDDT values of the bb_remodeled_residues (this residue set is found in the bb_remodeled_residues column
			of the stability_data dataframe(s)). Uses the default pLDDT and pAE thresholds or the provided thresholds.


Outputs:

	Figures and data files written to the outfolder_parent/job_name.

"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted

def sns_plot(x_vals, y_vals, x_name, y_name, job_name, training_step, outfolder, title_append, filename_append):
	"""
	Plot the RMSDs to the ground-truth structure for the classic and fine-tuned AF2 predictions.
	"""
	# sns.set_theme()
	sns.set_theme(style="whitegrid")
	fig = plt.figure()
	ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
	ax.set_xlabel('%s\nRMSD to Rosetta' % x_name) #, fontsize=15)
	ax.set_ylabel('%s RMSD to Rosetta' % y_name) #, fontsize=15)
	ax.set_title('RMSD to Rosetta for Classic and\nFine-Tuned AF2 models, %s Training Steps%s'%(training_step, title_append), fontsize=15)
	values = np.vstack([x_vals, y_vals])
	kernel = stats.gaussian_kde(values)(values)
	sns.scatterplot(x=x_vals, y=y_vals, c=kernel, cmap='viridis')
	lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
	        np.max([ax.get_xlim(), ax.get_ylim()])]
	ax.plot(lims, lims, '--', alpha=0.5, zorder=100)
	ax.set_aspect('equal')
	# plt.tight_layout() # not compatible with these fig axes.
	plt.savefig(os.path.join(outfolder,'%s_step_%s_rmsds%s.png'%(job_name, training_step, filename_append)), dpi=300)
	print('Saved figure: %s.png' % (os.path.join(outfolder,'%s_step_%s_rmsds%s.png'%(job_name, training_step, filename_append))))
	plt.close()

def sns_violinplot(data, x, y, hue, split, title='', outpath=''):

	sns.set_theme(style="whitegrid")
	sns.set(font_scale=1)
	fig = plt.figure()
	ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
	ax.set_xlabel(x) #, fontsize=15)
	ax.set_ylabel(y) #, fontsize=15)
	ax.set_title(title, fontsize=15)

	if 'uncertain' in data[x].tolist():
		vp = sns.violinplot(data=data, x=x, y=y, hue=hue, split=split, order=['stable','unstable','uncertain'])
	else:
		vp = sns.violinplot(data=data, x=x, y=y, hue=hue, split=split) #, order=['stable','unstable'])

	plt.savefig(outpath, dpi=300)
	print('Saved figure: %s.png' % outpath)
	plt.close()

def compare_rmsds(args, df, job_name, training_step, outfolder, rmsd_thresh, title_append, filename_append, rmsd_col_ending='_gt_rmsd'):
	"""
	Compare the RMSDs between the classic and fine-tuned AF2 predictions and the
	ground truth structure. Plot the RMSDs against each other.

	Model 1 should always be the classic AF2 model.
	"""
	model1_name = df.iloc[0]['classic_name']
	model2_name = df.iloc[0]['ft_name']
	num_model1_better, num_model2_better = 0, 0
	num_model1_sub_thresh, num_model2_sub_thresh = 0, 0
	model1_rmsds, model2_rmsds = [], []
	for idx, row in df.iterrows():
		model1_rmsd = row['classic'+rmsd_col_ending]
		model2_rmsd = row['ft'+rmsd_col_ending]

		model1_rmsds.append(model1_rmsd)
		model2_rmsds.append(model2_rmsd)

		if model1_rmsd < model2_rmsd:
			num_model1_better+=1
		else:
			num_model2_better+=1
		if model1_rmsd < rmsd_thresh:
			num_model1_sub_thresh+=1
		if model2_rmsd < rmsd_thresh:
			num_model2_sub_thresh+=1

	if args.verbose:
		print('Training Step: %s:'%training_step)
		print('%s has lower RMSD to Rosetta for %d structures' % (model1_name, num_model1_better))
		print('%s has lower RMSD to Rosetta for %d structures' % (model2_name, num_model2_better))

		print('%s has sub-%.2f RMSD to Rosetta for %d structures' % (model1_name, rmsd_thresh, num_model1_sub_thresh))
		print('%s has sub-%.2f RMSD to Rosetta for %d structures' % (model2_name, rmsd_thresh, num_model2_sub_thresh))

	sns_plot(model2_rmsds, model1_rmsds, model2_name, model1_name, job_name, training_step, outfolder, title_append, filename_append)

	return num_model1_better, num_model2_better, num_model1_sub_thresh, num_model2_sub_thresh

def plot_rmsd_frac_over_training_step(args, training_steps, fracs, labels='', title='', outfile=None):
	"""
	Plot the fractions of designs with lower RMSDs to the ground truth structure than the other model
	(classic or fine-tuned), or the fractions of designs predicted to sub-<RMSD threshold> RMSD to the 
	ground truth model, over training steps.
	"""
	fig = plt.figure()
	ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
	ax.set_xlabel('Training Step')
	ax.set_ylabel('Fraction')
	ax.set_title(title)
	if args.verbose:
		print('labels:')
		print(labels)
		print('training steps:')
		print(training_steps)
		print('fracs:')
		print(fracs)
	if labels != '':
		for t, f, l in zip(training_steps, fracs, labels):
			ax.plot(t, f, label=l)
		ax.legend()
	else:
		for t, f in zip(training_steps, fracs):
			ax.plot(t, f)
	if outfile != None:
		plt.savefig(outfile, dpi=300)
		print('Saved figure: %s'%outfile)
		plt.close()

def make_all_rmsd_frac_over_training_step_figures(args, df, title_append='', filename_append='', outfolder='', rmsd_col_ending='_gt_rmsd'):
	"""
	Make four plots: two with the fractions of designs predicted to lower RMSD to the ground truth
	structures for both classic and fine-tuned AF2, over fine-tuning steps, and two with the 
	fractions of designs predicted to lower than args.rmsd_thresh RMSD to the ground truth structures.
	"""
	ground_truth_parents = natsorted(df['ground_truth_parent'].unique().tolist())

	dataset_names = [ground_truth_parent.split('/')[-2] for ground_truth_parent in ground_truth_parents]

	# Load these parent arrays with data for each of the ground_truth_parents
	all_training_steps, all_frac_classic_lower_rmsd, all_frac_ft_lower_rmsd = [], [], []
	all_frac_classic_sub_thresh, all_frac_ft_sub_thresh = [], []

	for ground_truth_parent in ground_truth_parents:
		# Slice the df to include only examples from the current ground_truth_parent
		sub_df = df.loc[df['ground_truth_parent'] == ground_truth_parent]
		training_steps, frac_classic_lower_rmsd, frac_ft_lower_rmsd = [], [], []
		frac_classic_sub_thresh, frac_ft_sub_thresh = [], []
		for training_step in natsorted(sub_df['training_step'].unique().tolist()):
			sub_df_2 = sub_df.loc[sub_df['training_step']==training_step]
			if args.verbose:
				print('Processing Training Step %d'%training_step)
				print('Number of examples predicted by this checkpointed model: %d'%len(sub_df_2))
				print('Number of %s examples predicted by this checkpointed model: %d'%(ground_truth_parent,len(sub_df_2)))
			num_classic_better, num_ft_better, num_classic_sub_thresh, num_ft_sub_thresh = \
				compare_rmsds(args, sub_df_2, args.job_name, training_step, outfolder, args.rmsd_thresh, title_append, filename_append, rmsd_col_ending=rmsd_col_ending)
			training_steps.append(int(training_step))
			frac_classic_lower_rmsd.append(num_classic_better/len(sub_df_2))
			frac_ft_lower_rmsd.append(num_ft_better/len(sub_df_2))
			frac_classic_sub_thresh.append(num_classic_sub_thresh/len(sub_df_2))
			frac_ft_sub_thresh.append(num_ft_sub_thresh/len(sub_df_2))

		all_training_steps.append(training_steps)
		all_frac_classic_lower_rmsd.append(frac_classic_lower_rmsd)
		all_frac_ft_lower_rmsd.append(frac_ft_lower_rmsd)
		all_frac_classic_sub_thresh.append(frac_classic_sub_thresh)
		all_frac_ft_sub_thresh.append(frac_ft_sub_thresh)

	# Plot the fractions of designs predicted with lower RMSDs over training steps.
	plot_rmsd_frac_over_training_step(args, all_training_steps, all_frac_classic_lower_rmsd, labels=dataset_names, 
		title='Fraction of designs Classic AF2\npredicted lower RMSD to Rosetta%s'%title_append,
		outfile=os.path.join(outfolder, '%s_classic_lower_rmsd%s.png'%(args.job_name, filename_append)))
	plot_rmsd_frac_over_training_step(args, all_training_steps, all_frac_ft_lower_rmsd, labels=dataset_names,
		title='Fraction of designs Fine-tuned AF2\npredicted lower RMSD to Rosetta%s'%title_append,
		outfile=os.path.join(outfolder, '%s_ft_lower_rmsd%s.png'%(args.job_name, filename_append)))
	plot_rmsd_frac_over_training_step(args, all_training_steps, all_frac_classic_sub_thresh, labels=dataset_names,
		title='Fraction of designs Classic AF2\npredicted <%.2f RMSD to Rosetta%s'%(args.rmsd_thresh, title_append),
		outfile=os.path.join(outfolder, '%s_classic_sub%s_rmsd%s.png'%\
			(args.job_name, str(args.rmsd_thresh).replace('.','p'), filename_append)))
	plot_rmsd_frac_over_training_step(args, all_training_steps, all_frac_ft_sub_thresh, labels=dataset_names,
		title='Fraction of designs Fine-tuned AF2\npredicted <%.2f RMSD to Rosetta%s'%(args.rmsd_thresh, title_append),
		outfile=os.path.join(outfolder, '%s_ft_sub%s_rmsd%s.png'%\
			(args.job_name, str(args.rmsd_thresh).replace('.','p'), filename_append)))

def load_plddts(plddt_string):
	return np.array([float(v) for v in \
		plddt_string.replace('\n','').replace('[','').replace(']','').split()])

def load_paes(pae_string):
	paes = []
	for row in pae_string.split('] ['):
		row = row.replace('\n','').replace('[','').replace(']','')
		paes.append([float(v) for v in row.split()])
	return np.array(paes)

def calc_precision_recall_for_one_example(row, thresh,
	val, tp, fp, tn, fn, higher_better=True):
	"""
	Add to the number of true/false positives/negatives
	and return their values.
	"""
	# If lower (positive) values are considered higher confidence
	# (like with pAE), then multiply the pAE values by -1 to use
	# the same logic below as for pLDDT.
	if not higher_better:
		val = -1 * val
		thresh = -1 * thresh
	if row['stability_label'] == 'stable':
		if val > thresh:
			tp += 1
		else:
			fn += 1
	elif row['stability_label'] == 'unstable':
		if val > thresh:
			fp += 1
		else:
			tn += 1
	return tp, fp, tn, fn

def get_precision_value(tp, fp):
	if (tp + fp) == 0:
		return 1
	else:
		return tp / (tp + fp)

def get_recall_value(tp, fn):
	if (tp + fn) == 0:
		return 0
	else:
		return tp / (tp + fn)

def get_false_positive_rate(fp, tn):
	if (fp + tn) == 0:
		return 0
	else:
		return fp / (fp + tn)

def count_confusion_matrix_values(args, thresh, sub_plot_df):
	"""
	Calculate values in the pLDDT and pAE confusion matrices for a given threshold.
	"""

	## For the classic AF2 model:
	# True/False Positive/Negative for calculating Precision and Recall
	# These values are used when classifying by both pLDDT and pAE
	c_tp, c_fp, c_tn, c_fn = 0, 0, 0, 0
	# These values are used when classifying by only pLDDT
	c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn = 0, 0, 0, 0
	# These values are used when classifying by only pAE
	c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn = 0, 0, 0, 0
	## For the fine-tuned AF2 model:
	# True/False Positive/Negative for calculating Precision and Recall
	# These values are used when classifying by both pLDDT and pAE
	f_tp, f_fp, f_tn, f_fn = 0, 0, 0, 0
	# These values are used when classifying by only pLDDT
	f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn = 0, 0, 0, 0
	# These values are used when classifying by only pAE
	f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn = 0, 0, 0, 0

	for idx, row in sub_plot_df.iterrows():

		model_type = row['model_type']
		plddt = row[plddt_metric]
		pae = row[pae_metric]

		if model_type=='classic':
			c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn = calc_precision_recall_for_one_example(row,
				thresh, plddt, c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn,
				higher_better=True)
			c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn = calc_precision_recall_for_one_example(row,
				thresh, pae, c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn,
				higher_better=False)

		elif model_type=='ft':
			f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn = calc_precision_recall_for_one_example(row,
				thresh, plddt, f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn,
				higher_better=True)
			f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn = calc_precision_recall_for_one_example(row,
				thresh, pae, f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn,
				higher_better=False)

	if args.verbose:
		print('Precision and Recall for checkpointed model after %d training steps for %s with threshold %d'%\
				(sub_plot_df.iloc[0]['training_step'], dataset, thresh))
		print('c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn')
		print('%d %d %d %d'%(c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn))
		print('f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn')
		print('%d %d %d %d'%(f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn))
		print('c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn')
		print('%d %d %d %d'%(c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn))
		print('f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn')
		print('%d %d %d %d'%(f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn))

	return c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn, \
			f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn, \
			c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn, \
			f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn


def plot_precision_recall_curve(df, metric_type=None, outpath=None):
	"""
	Plot a Precision-Recall curve for the provided df, which
	contains 'precision' and 'recall' values.
	"""
	# sns.set_theme()
	sns.set_theme(style="whitegrid")
	fig = plt.figure()
	ax = fig.add_axes([0.04, 0.2, 0.6, 0.6])
	ax.set_xlabel('Recall', fontsize=15)
	ax.set_ylabel('Precision', fontsize=15)
	ax.set_title('Precision vs. Recall for Checkpointed\nModel at %d Training Steps\nClassify by %s'%(df.iloc[0]['training_step'],metric_type), fontsize=12)

	# df = df.sort_values(by='precision')

	sub_df = df.loc[df['metric_type']==metric_type]

	## If plotting multiple datasets
	# lp = sns.lineplot(data=df, x='recall', y='precision', hue=df[['model_type','dataset']].apply(tuple, axis=1))
	## If plotting only one dataset
	lp = sns.lineplot(data=sub_df, x='recall', y='precision', hue='model_type', ci=None)

	plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

	# Change legend fontsize
	plt.setp(lp.get_legend().get_texts(), fontsize=20) 

	# Add a line from (0,1) to (1,0)
	# lims = [[0,1],[1,0]]
	# ax.plot(lims[0], lims[1], '--', alpha=0.5, zorder=100)
	# plt.tight_layout() # not compatible with these fig axes.
	plt.xlim(-0.05, 1.05)
	plt.ylim(-0.05, 1.05)
	ax.set_aspect('equal')
	plt.savefig(outpath, dpi=300)
	print('Saved figure: %s.png' % outpath)
	plt.close()

def plot_tpr_fpr_curve(df, metric_type=None, outpath=None):
	"""
	Plot a True Positive Rate vs. False Positive Rate curve 
	for the provided df, which contains 'recall' and 'FPR' values.
	"""
	sns.set(font_scale=1.5)
	sns.set_theme(style="whitegrid")

	# sns.set_theme()
	fig = plt.figure()
	ax = fig.add_axes([0.1, 0.2, 0.6, 0.6])
	ax.set_xlabel('False Positive Rate', fontsize=15)
	ax.set_ylabel('True Positive Rate', fontsize=15)
	ax.set_title('TPR vs. FPR for Checkpointed\nModel at %d Training Steps.\n%s. Classify by %s'%(df.iloc[0]['training_step'],df.iloc[0]['dataset'],metric_type), fontsize=12)

	# Only plot one metric
	df = df.loc[df['metric_type']==metric_type]

	# df = df.sort_values(by='FPR')

	# ci=None turns off confidence intervals.
	lp = sns.lineplot(data=df, x='FPR', y='recall', hue='model_type', hue_order=['classic', 'ft'], ci=None) # hue=df[['model_type','dataset']].apply(tuple, axis=1)

	# Calculate AUROC
	df = df.sort_values(by='FPR')
	xc = df.loc[df['model_type']=='classic']['FPR'].tolist()
	yc = df.loc[df['model_type']=='classic']['recall'].tolist()
	auc_c = auc(xc, yc)

	xf = df.loc[df['model_type']=='ft']['FPR'].tolist()
	yf = df.loc[df['model_type']=='ft']['recall'].tolist()
	auc_f = auc(xf, yf)

	plt.text(0.2, 0.14, 'Classic AUROC = %.3f'%auc_c, fontsize=12)
	plt.text(0.2, 0.06, 'Fine-tuned AUROC = %.3f'%auc_f, fontsize=12)

	plt.yticks(fontsize=16)
	plt.xticks(fontsize=16)

	plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

	# Change legend fontsize
	plt.setp(lp.get_legend().get_texts(), fontsize='20') 

	# Add a line from (0,1) to (1,0)
	# lims = [[0,1],[1,0]]
	# ax.plot(lims[0], lims[1], '--', alpha=0.5, zorder=100)
	# plt.tight_layout() # not compatible with these fig axes.
	plt.xlim(-0.05, 1.05)
	plt.ylim(-0.05, 1.05)
	ax.set_aspect('equal')
	plt.savefig(outpath, dpi=300)
	print('Saved figure: %s.png' % outpath)
	plt.close()

def plot_rmsds_for_each_dataset(df, args, rmsd_col_ending='_gt_rmsd', title_append='', filename_append='', outfolder=''):
	"""
	Plot RMSD to Rosetta values for each example over dataset name, for both the classic and
	fine-tuned AF2 predictions.
	"""
	# sns.set_theme()
	sns.set_theme(style="whitegrid")
	sns.set(font_scale=2)

	for training_step in natsorted(df['training_step'].unique().tolist()):
		
		sub_df = df.loc[df['training_step']==training_step]
		
		first_df = True
		
		for ground_truth_parent in natsorted(df['ground_truth_parent'].unique().tolist()):

			# Second to last subfolder in the ground_truth_parent is the name of the de novo protein dataset
			# For example, /wynton/group/kortemme/borr/alphafold_finetune/data/abba_rocklin_rd6/pdbs
			dataset_name = ground_truth_parent.split('/')[-2]

			# Slice the df to include only examples from the current ground_truth_parent
			sub_df_2 = sub_df.loc[sub_df['ground_truth_parent'] == ground_truth_parent]
			
			classic_tuples = list(zip([dataset_name]*len(sub_df_2), ['Classic']*len(sub_df_2), sub_df_2['classic'+rmsd_col_ending]))
			ft_tuples = list(zip([dataset_name]*len(sub_df_2), ['Fine-tuned']*len(sub_df_2), sub_df_2['ft'+rmsd_col_ending]))
			list_of_tuples = classic_tuples + ft_tuples

			# Make a new df in an sns-friendly format.
			temp_df = pd.DataFrame(list_of_tuples,
						columns=['Dataset', 'Model', 'RMSD to Rosetta'])

			if not first_df:
				plot_df = pd.concat([plot_df, temp_df], ignore_index=True)
				plot_df.reset_index()
			else:
				plot_df = temp_df
				first_df = False

		if len(df['ground_truth_parent'].unique().tolist()) == 1:
			# I should 
			fig = plt.figure(figsize=[12,8])
			ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
			sns.set_theme(style="whitegrid")
			ax.set_xlabel('Model', fontsize=25)
			ax.set_ylabel('RMSD to Rosetta', fontsize=25)
			ax.set_title('RMSD to Rosetta for Classic and\nFine-Tuned AF2 models, %s Training Steps%s'%(training_step, title_append), fontsize=25)
			sp = sns.stripplot(y='RMSD to Rosetta', x='Model', data=plot_df, dodge=True, jitter=True, alpha=0.5)
		else:
			fig = plt.figure(figsize=[16,9])
			ax = fig.add_axes([0.08, 0.2, 0.6, 0.6])
			sns.set_theme(style="whitegrid")
			ax.set_xlabel('Model', fontsize=25)
			ax.set_ylabel('RMSD to Rosetta', fontsize=25)
			ax.set_title('RMSD to Rosetta for Classic and\nFine-Tuned AF2 models, %s Training Steps%s'%(training_step, title_append), fontsize=25)
			sp = sns.stripplot(y='RMSD to Rosetta', x='Model', data=plot_df, hue='Dataset', dodge=True, jitter=True, alpha=0.5)
			plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
			# Change legend fontsize
			plt.setp(sp.get_legend().get_texts(), fontsize='20')

		plt.savefig(os.path.join(outfolder,'%s_step_%s_rmsd%s_points.png'%(args.job_name, training_step, filename_append)), dpi=300)
		print('Saved figure: %s' % (os.path.join(outfolder,'%s_step_%s_rmsd%s_points.png'%(args.job_name, training_step, filename_append))))
		plt.show()
		plt.close()


def violinplot_rmsds_for_each_dataset(df, args, rmsd_col_ending='_gt_rmsd', title_append='', filename_append='', outfolder=''):
	"""
	Plot violin plots of RMSD to Rosetta values for each example over dataset name, for both the classic and
	fine-tuned AF2 predictions.
	"""
	sns.set_theme(style="whitegrid")
	sns.set(font_scale=2)

	for training_step in natsorted(df['training_step'].unique().tolist()):

		sub_df = df.loc[df['training_step']==training_step]
		
		first_df = True
		
		for ground_truth_parent in natsorted(df['ground_truth_parent'].unique().tolist()):

			# Second to last subfolder in the ground_truth_parent is the name of the de novo protein dataset
			# For example, /wynton/group/kortemme/borr/alphafold_finetune/data/abba_rocklin_rd6/pdbs
			dataset_name = ground_truth_parent.split('/')[-2]

			# Slice the df to include only examples from the current ground_truth_parent
			sub_df_2 = sub_df.loc[sub_df['ground_truth_parent'] == ground_truth_parent]

			classic_tuples = list(zip([dataset_name]*len(sub_df_2), ['Classic']*len(sub_df_2), sub_df_2['classic'+rmsd_col_ending], sub_df_2['stability_label']))
			ft_tuples = list(zip([dataset_name]*len(sub_df_2), ['Fine-tuned']*len(sub_df_2), sub_df_2['ft'+rmsd_col_ending], sub_df_2['stability_label']))
			list_of_tuples = classic_tuples + ft_tuples

			# Make a new df in an sns-friendly format.
			temp_df = pd.DataFrame(list_of_tuples,
						columns=['Dataset', 'Model', 'RMSD to Rosetta', 'stability_label'])

			if not first_df:
				plot_df = pd.concat([plot_df, temp_df], ignore_index=True)
				plot_df.reset_index()
			else:
				plot_df = temp_df
				first_df = False

		if len(df['ground_truth_parent'].unique().tolist()) == 1:
			fig = plt.figure(figsize=[12,7])
			ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
			sns.set_theme(style="whitegrid")
			ax.set_xlabel('Model', fontsize=25)
			ax.set_ylabel('RMSD to Rosetta', fontsize=25)
			ax.set_title('RMSD to Rosetta for Classic and\nFine-Tuned AF2 models, %s Training Steps%s'%(training_step, title_append), fontsize=25)
			vp = sns.violinplot(y='RMSD to Rosetta', x='Model', data=plot_df, hue='stability_label', split=True)

		else:
			fig = plt.figure(figsize=[16,9])
			ax = fig.add_axes([0.08, 0.2, 0.6, 0.6])
			sns.set_theme(style="whitegrid")
			ax.set_xlabel('Model', fontsize=25)
			ax.set_ylabel('RMSD to Rosetta', fontsize=25)
			ax.set_title('RMSD to Rosetta for Classic and\nFine-Tuned AF2 models, %s Training Steps%s'%(training_step, title_append), fontsize=25)
			vp = sns.violinplot(y='RMSD to Rosetta', x='Model', data=plot_df, hue='stability_label', split=True) # hue='Dataset')
			plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
			# Change legend fontsize
			plt.setp(vp.get_legend().get_texts(), fontsize='20')

		plt.savefig(os.path.join(outfolder,'%s_step_%s_rmsd%s_distributions.png'%(args.job_name, training_step, filename_append)), dpi=300)
		print('Saved figure: %s' % (os.path.join(outfolder,'%s_step_%s_rmsd%s_distributions.png'%(args.job_name, training_step, filename_append))))
		plt.show()
		plt.close()

def get_confidence_values(args, new_df, row, stab_row, possible_names):
	"""
	Load pLDDT and pAE values for the current design.
	"""

	# Get the sequence length of each example, and only analyze the first num_seq
	# positions in the pae and plddt arrays. I believe this is from a bug in the
	# alphafold_finetune code, must pad its input representations, and outputs
	# pLDDT and pAE matrices that include low values for padded positions.
	try:
		seq_len = len(stab_row['sequence'])
	except: # The Rocklin dataframes use this column for protein sequence
		seq_len = len(stab_row['protein_sequence_c'])

	# Select the first seq_len residues from which to calculate
	# average pLDDT and pAE scores.
	# alphafold_finetune thresholds are pLDDT / 100 and pAE / 10 (I believe).
	
	# The arrays of per-residue confidence values, saved as strings,
	# need to be reformatted
	c_plddts = load_plddts(row['classic_plddt'])
	f_plddts = load_plddts(row['ft_plddt'])					

	c_paes = load_paes(row['classic_pae'])
	f_paes = load_paes(row['ft_pae'])

	## That these values should be divided by 100 and 10 may be a faulty assumption.
	## I need to see how the binder_intercepts are used in alphafold_finetune!
	## For now, it may make more sense to simply plot the distributions of confidence values
	## for stable and unstable designs, before and after AF2 fine-tuning.
	c_plddt = np.mean(c_plddts[:seq_len])
	f_plddt = np.mean(f_plddts[:seq_len])

	## The output confidence arrays from alphafold_finetune appear to have padded values,
	## so just use the array indices that correspond with actual residue indices.
	c_pae = np.mean(c_paes[:seq_len,:seq_len])
	f_pae = np.mean(f_paes[:seq_len,:seq_len])

	if args.verbose:
		print('Classic and Fine-tuned pLDDTs')
		print(c_plddt)
		print(f_plddt)

		print('Classic and Fine-tuned pAEs')
		print(c_pae)
		print(f_pae)

	return c_pae, f_pae, c_plddt, f_plddt


=======
Analyzes metrics for AlphaFold checkpointed models across training.

Author: Ben Orr
Date: 4.4.23

This module provides comprehensive analysis tools for comparing classic and fine-tuned
AlphaFold2 (AF2) predictions against ground truth structures. It evaluates both
structural accuracy (RMSD) and confidence metrics (pLDDT and pAE) to assess model
performance across training checkpoints.

Usage
-----
analyze_checkpointed_models_metrics.py --job_name <job_name> \
    --checkpoint_model_data_folder <checkpoint_model_data_folder> \
    --outfolder_parent <output_folder_parent> [ --rmsd_analysis \
    --stability_data <stability_dataframe1> [ <stability_dataframe1> ... ] \
    --rmsd_thresh <rmsd_threshold> --plot_rmsd_distributions \
    --reshaped_helix_rmsd --stability_classifier \
    --plddt_thresh <pLDDT_threshold> --pae_thresh <pAE_threshold> \
    --plot_confidence_distributions --precision_recall --reshaped_residue_stability]

Parameters
----------
job_name : str
    Name for the job which will be used in the output .png files.

checkpoint_model_data_folder : str
    Folder containing dataframe(s) output by calc_checkpointed_models_metrics.py.

outfolder_parent : str
    Output files are written to outfolder_parent.

rmsd_analysis : bool, optional
    Analyze the RMSDs to the ground truth structures for each checkpointed model's
    predictions.

stability_data : list of str, optional
    Dataframe(s) containing a 'stability_label' column with 'stable,' 'unstable,'
    and 'uncertain' values, as well as a 'sequence' column, which is used to get
    the sequence length.

rmsd_thresh : float, optional
    Threshold for AF2 to ground truth RMSD values for plotting the fraction of
    designs predicted to sub-<rmsd_thresh> RMSD to the ground truth structures.
    Default: 1.5

plot_rmsd_distributions : bool, optional
    Plot distributions of RMSD values for stable vs. unstable designs.

reshaped_helix_rmsd : bool, optional
    Plot Reshaped Helix RMSD to Rosetta values for LUCS designs.

stability_classifier : bool, optional
    Analyze the Precision and Recall of the AF2 confidence scores in predicting
    designs as stable and unstable. Uses the default pLDDT and pAE thresholds or
    the provided thresholds.

plddt_thresh : float, optional
    Threshold for AF2 pLDDT values in classifying a design as stable (if the AF2
    average pLDDT for a design is greater than this threshold, then the design is
    classified as "stable"). Default is 0.80367635 (taken from the alphafold_finetune
    github repo: https://github.com/phbradley/alphafold_finetune).

pae_thresh : float, optional
    Threshold for AF2 pAE values in classifying a design as stable (if the AF2
    average pAE for a design is greater than this threshold, then the design is
    classified as "stable"). Default is 0.43373787 (taken from the alphafold_finetune
    github repo: https://github.com/phbradley/alphafold_finetune).

plot_confidence_distributions : bool, optional
    Plot the distributions of the confidence scores for the classic and fine-tuned
    models at each training step. Requires a outfolder_parent/job_name/plotting_dfs
    folder to exist with outputs from the --stability_classifier outputs.

precision_recall : bool, optional
    Generate Precision vs. Recall curves for classifying stable designs at different
    pLDDT and pAE thresholds.

reshaped_residue_stability : bool, optional
    Analyze the Precision and Recall of the AF2 confidence scores in predicting
    designs as stable and unstable using the pAE values between the bb_remodeled_residues
    and non-bb_remodeled_residues and pLDDT values of the bb_remodeled_residues (this
    residue set is found in the bb_remodeled_residues column of the stability_data
    dataframe(s)). Uses the default pLDDT and pAE thresholds or the provided thresholds.

Outputs
-------
Figures and data files written to the outfolder_parent/job_name.

Notes
-----
This module processes checkpoint data from fine-tuned AlphaFold2 models to evaluate
how model predictions improve over training steps. It provides statistical and
visual comparisons between classic and fine-tuned model performance.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from natsort import natsorted
from scipy import stats
from sklearn.metrics import auc


def sns_plot(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_name: str,
    y_name: str,
    job_name: str,
    training_step: int,
    outfolder: str,
    title_append: str,
    filename_append: str
) -> None:
    """
    Plot the RMSDs to the ground-truth structure for the classic and fine-tuned AF2 predictions.

    Creates a density scatter plot comparing RMSD values between two models with
    a diagonal reference line for visual comparison.

    Parameters
    ----------
    x_vals : np.ndarray
        RMSD values for the x-axis model.
    y_vals : np.ndarray
        RMSD values for the y-axis model.
    x_name : str
        Label for the x-axis model.
    y_name : str
        Label for the y-axis model.
    job_name : str
        Name of the job for output file naming.
    training_step : int
        Number of training steps for the fine-tuned model.
    outfolder : str
        Directory path where the figure will be saved.
    title_append : str
        Additional text to append to the plot title.
    filename_append : str
        Additional text to append to the output filename.

    Returns
    -------
    None
        Saves the plot to disk.

    Notes
    -----
    The plot uses a Gaussian kernel density estimation to color points by density,
    making it easier to identify regions with many overlapping data points.
    """
    # sns.set_theme()
    sns.set_theme(style="whitegrid")
    fig = plt.figure()
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    ax.set_xlabel('%s\nRMSD to Rosetta' % x_name)  # , fontsize=15)
    ax.set_ylabel('%s RMSD to Rosetta' % y_name)  # , fontsize=15)
    ax.set_title(
        'RMSD to Rosetta for Classic and\nFine-Tuned AF2 models, %s Training Steps%s' % (
            training_step, title_append
        ),
        fontsize=15
    )
    values = np.vstack([x_vals, y_vals])
    kernel = stats.gaussian_kde(values)(values)
    sns.scatterplot(x=x_vals, y=y_vals, c=kernel, cmap='viridis')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, '--', alpha=0.5, zorder=100)
    ax.set_aspect('equal')
    # plt.tight_layout() # not compatible with these fig axes.
    plt.savefig(
        os.path.join(outfolder, '%s_step_%s_rmsds%s.png' % (
            job_name, training_step, filename_append
        )),
        dpi=300
    )
    print('Saved figure: %s.png' % (
        os.path.join(outfolder, '%s_step_%s_rmsds%s.png' % (
            job_name, training_step, filename_append
        ))
    ))
    plt.close()


def sns_violinplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    split: bool,
    title: str = '',
    outpath: str = ''
) -> None:
    """
    Create a violin plot for visualizing data distributions.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to plot.
    x : str
        Column name for the x-axis categorical variable.
    y : str
        Column name for the y-axis continuous variable.
    hue : str
        Column name for grouping variable that will split the violins.
    split : bool
        Whether to split the violins by the hue variable.
    title : str, optional
        Title for the plot. Default is ''.
    outpath : str, optional
        File path where the figure will be saved. Default is ''.

    Returns
    -------
    None
        Saves the plot to disk.

    Notes
    -----
    This function automatically orders categories as 'stable', 'unstable', 'uncertain'
    if the 'uncertain' category is present in the data.
    """
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1)
    fig = plt.figure()
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    ax.set_xlabel(x)  # , fontsize=15)
    ax.set_ylabel(y)  # , fontsize=15)
    ax.set_title(title, fontsize=15)

    if 'uncertain' in data[x].tolist():
        vp = sns.violinplot(
            data=data, x=x, y=y, hue=hue, split=split,
            order=['stable', 'unstable', 'uncertain']
        )
    else:
        vp = sns.violinplot(
            data=data, x=x, y=y, hue=hue, split=split
        )  # , order=['stable','unstable'])

    plt.savefig(outpath, dpi=300)
    print('Saved figure: %s.png' % outpath)
    plt.close()


def compare_rmsds(
    args: argparse.Namespace,
    df: pd.DataFrame,
    job_name: str,
    training_step: int,
    outfolder: str,
    rmsd_thresh: float,
    title_append: str,
    filename_append: str,
    rmsd_col_ending: str = '_gt_rmsd'
) -> Tuple[int, int, int, int]:
    """
    Compare the RMSDs between the classic and fine-tuned AF2 predictions and the ground truth structure.

    Calculates statistics comparing how many designs each model predicts more
    accurately and generates a visualization comparing their performance.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration options.
    df : pd.DataFrame
        DataFrame containing RMSD data for both models.
    job_name : str
        Name of the job for output file naming.
    training_step : int
        Number of training steps for the fine-tuned model.
    outfolder : str
        Directory path where outputs will be saved.
    rmsd_thresh : float
        RMSD threshold for classification.
    title_append : str
        Additional text to append to plot titles.
    filename_append : str
        Additional text to append to output filenames.
    rmsd_col_ending : str, optional
        Suffix for RMSD column names. Default is '_gt_rmsd'.

    Returns
    -------
    num_model1_better : int
        Number of structures where model 1 (classic) has lower RMSD.
    num_model2_better : int
        Number of structures where model 2 (fine-tuned) has lower RMSD.
    num_model1_sub_thresh : int
        Number of structures where model 1 has RMSD below threshold.
    num_model2_sub_thresh : int
        Number of structures where model 2 has RMSD below threshold.

    Notes
    -----
    Model 1 should always be the classic AF2 model.
    """
    model1_name = df.iloc[0]['classic_name']
    model2_name = df.iloc[0]['ft_name']
    num_model1_better, num_model2_better = 0, 0
    num_model1_sub_thresh, num_model2_sub_thresh = 0, 0
    model1_rmsds, model2_rmsds = [], []
    for idx, row in df.iterrows():
        model1_rmsd = row['classic' + rmsd_col_ending]
        model2_rmsd = row['ft' + rmsd_col_ending]

        model1_rmsds.append(model1_rmsd)
        model2_rmsds.append(model2_rmsd)

        if model1_rmsd < model2_rmsd:
            num_model1_better += 1
        else:
            num_model2_better += 1
        if model1_rmsd < rmsd_thresh:
            num_model1_sub_thresh += 1
        if model2_rmsd < rmsd_thresh:
            num_model2_sub_thresh += 1

    if args.verbose:
        print('Training Step: %s:' % training_step)
        print('%s has lower RMSD to Rosetta for %d structures' % (
            model1_name, num_model1_better
        ))
        print('%s has lower RMSD to Rosetta for %d structures' % (
            model2_name, num_model2_better
        ))

        print('%s has sub-%.2f RMSD to Rosetta for %d structures' % (
            model1_name, rmsd_thresh, num_model1_sub_thresh
        ))
        print('%s has sub-%.2f RMSD to Rosetta for %d structures' % (
            model2_name, rmsd_thresh, num_model2_sub_thresh
        ))

    sns_plot(
        model2_rmsds, model1_rmsds, model2_name, model1_name, job_name,
        training_step, outfolder, title_append, filename_append
    )

    return num_model1_better, num_model2_better, num_model1_sub_thresh, num_model2_sub_thresh


def plot_rmsd_frac_over_training_step(
    args: argparse.Namespace,
    training_steps: List[List[int]],
    fracs: List[List[float]],
    labels: Union[str, List[str]] = '',
    title: str = '',
    outfile: Optional[str] = None
) -> None:
    """
    Plot fractions of designs with specified RMSD characteristics over training steps.

    Creates line plots showing how model performance changes across training steps,
    such as fractions of designs with lower RMSDs or sub-threshold RMSDs.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration options.
    training_steps : list of list of int
        Training step values for each data series.
    fracs : list of list of float
        Fraction values corresponding to each training step.
    labels : str or list of str, optional
        Labels for each data series. Default is ''.
    title : str, optional
        Title for the plot. Default is ''.
    outfile : str or None, optional
        Path where the figure will be saved. If None, plot is not saved.
        Default is None.

    Returns
    -------
    None
        Optionally saves the plot to disk if outfile is provided.

    Notes
    -----
    If verbose mode is enabled, prints the labels, training steps, and fractions
    for debugging purposes.
    """
    fig = plt.figure()
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Fraction')
    ax.set_title(title)
    if args.verbose:
        print('labels:')
        print(labels)
        print('training steps:')
        print(training_steps)
        print('fracs:')
        print(fracs)
    if labels != '':
        for t, f, l in zip(training_steps, fracs, labels):
            ax.plot(t, f, label=l)
        ax.legend()
    else:
        for t, f in zip(training_steps, fracs):
            ax.plot(t, f)
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        print('Saved figure: %s' % outfile)
        plt.close()


def make_all_rmsd_frac_over_training_step_figures(
    args: argparse.Namespace,
    df: pd.DataFrame,
    title_append: str = '',
    filename_append: str = '',
    outfolder: str = '',
    rmsd_col_ending: str = '_gt_rmsd'
) -> None:
    """
    Make four plots tracking RMSD statistics across training steps.

    Generates plots showing: (1) fraction of designs where classic AF2 has lower
    RMSD, (2) fraction where fine-tuned AF2 has lower RMSD, (3) fraction where
    classic AF2 achieves sub-threshold RMSD, and (4) fraction where fine-tuned
    AF2 achieves sub-threshold RMSD.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration options including job_name
        and rmsd_thresh.
    df : pd.DataFrame
        DataFrame containing RMSD data with 'ground_truth_parent' and
        'training_step' columns.
    title_append : str, optional
        Additional text to append to plot titles. Default is ''.
    filename_append : str, optional
        Additional text to append to output filenames. Default is ''.
    outfolder : str, optional
        Directory path where outputs will be saved. Default is ''.
    rmsd_col_ending : str, optional
        Suffix for RMSD column names. Default is '_gt_rmsd'.

    Returns
    -------
    None
        Saves four plots to disk.

    Notes
    -----
    This function processes data for each unique ground truth parent directory
    separately to allow comparison across different datasets.
    """
    ground_truth_parents = natsorted(df['ground_truth_parent'].unique().tolist())

    dataset_names = [
        ground_truth_parent.split('/')[-2]
        for ground_truth_parent in ground_truth_parents
    ]

    # Load these parent arrays with data for each of the ground_truth_parents
    all_training_steps, all_frac_classic_lower_rmsd, all_frac_ft_lower_rmsd = [], [], []
    all_frac_classic_sub_thresh, all_frac_ft_sub_thresh = [], []

    for ground_truth_parent in ground_truth_parents:
        # Slice the df to include only examples from the current ground_truth_parent
        sub_df = df.loc[df['ground_truth_parent'] == ground_truth_parent]
        training_steps, frac_classic_lower_rmsd, frac_ft_lower_rmsd = [], [], []
        frac_classic_sub_thresh, frac_ft_sub_thresh = [], []
        for training_step in natsorted(sub_df['training_step'].unique().tolist()):
            sub_df_2 = sub_df.loc[sub_df['training_step'] == training_step]
            if args.verbose:
                print('Processing Training Step %d' % training_step)
                print('Number of examples predicted by this checkpointed model: %d' % len(sub_df_2))
                print('Number of %s examples predicted by this checkpointed model: %d' % (
                    ground_truth_parent, len(sub_df_2)
                ))
            num_classic_better, num_ft_better, num_classic_sub_thresh, num_ft_sub_thresh = \
                compare_rmsds(
                    args, sub_df_2, args.job_name, training_step, outfolder,
                    args.rmsd_thresh, title_append, filename_append,
                    rmsd_col_ending=rmsd_col_ending
                )
            training_steps.append(int(training_step))
            frac_classic_lower_rmsd.append(num_classic_better / len(sub_df_2))
            frac_ft_lower_rmsd.append(num_ft_better / len(sub_df_2))
            frac_classic_sub_thresh.append(num_classic_sub_thresh / len(sub_df_2))
            frac_ft_sub_thresh.append(num_ft_sub_thresh / len(sub_df_2))

        all_training_steps.append(training_steps)
        all_frac_classic_lower_rmsd.append(frac_classic_lower_rmsd)
        all_frac_ft_lower_rmsd.append(frac_ft_lower_rmsd)
        all_frac_classic_sub_thresh.append(frac_classic_sub_thresh)
        all_frac_ft_sub_thresh.append(frac_ft_sub_thresh)

    # Plot the fractions of designs predicted with lower RMSDs over training steps.
    plot_rmsd_frac_over_training_step(
        args, all_training_steps, all_frac_classic_lower_rmsd, labels=dataset_names,
        title='Fraction of designs Classic AF2\npredicted lower RMSD to Rosetta%s' % title_append,
        outfile=os.path.join(
            outfolder, '%s_classic_lower_rmsd%s.png' % (args.job_name, filename_append)
        )
    )
    plot_rmsd_frac_over_training_step(
        args, all_training_steps, all_frac_ft_lower_rmsd, labels=dataset_names,
        title='Fraction of designs Fine-tuned AF2\npredicted lower RMSD to Rosetta%s' % title_append,
        outfile=os.path.join(
            outfolder, '%s_ft_lower_rmsd%s.png' % (args.job_name, filename_append)
        )
    )
    plot_rmsd_frac_over_training_step(
        args, all_training_steps, all_frac_classic_sub_thresh, labels=dataset_names,
        title='Fraction of designs Classic AF2\npredicted <%.2f RMSD to Rosetta%s' % (
            args.rmsd_thresh, title_append
        ),
        outfile=os.path.join(
            outfolder, '%s_classic_sub%s_rmsd%s.png' % (
                args.job_name, str(args.rmsd_thresh).replace('.', 'p'), filename_append
            )
        )
    )
    plot_rmsd_frac_over_training_step(
        args, all_training_steps, all_frac_ft_sub_thresh, labels=dataset_names,
        title='Fraction of designs Fine-tuned AF2\npredicted <%.2f RMSD to Rosetta%s' % (
            args.rmsd_thresh, title_append
        ),
        outfile=os.path.join(
            outfolder, '%s_ft_sub%s_rmsd%s.png' % (
                args.job_name, str(args.rmsd_thresh).replace('.', 'p'), filename_append
            )
        )
    )


def load_plddts(plddt_string: str) -> np.ndarray:
    """
    Parse a string representation of pLDDT values into a NumPy array.

    Parameters
    ----------
    plddt_string : str
        String representation of pLDDT values, formatted with brackets and spaces.

    Returns
    -------
    np.ndarray
        Array of pLDDT values as floats.

    Notes
    -----
    This function removes newlines, brackets, and parses space-separated values.
    """
    return np.array([
        float(v) for v in
        plddt_string.replace('\n', '').replace('[', '').replace(']', '').split()
    ])


def load_paes(pae_string: str) -> np.ndarray:
    """
    Parse a string representation of pAE values into a 2D NumPy array.

    Parameters
    ----------
    pae_string : str
        String representation of pAE matrix values, formatted with nested brackets.

    Returns
    -------
    np.ndarray
        2D array of pAE values as floats.

    Notes
    -----
    This function handles the nested bracket structure of pAE matrices by splitting
    on '] [' delimiters and parsing each row separately.
    """
    paes = []
    for row in pae_string.split('] ['):
        row = row.replace('\n', '').replace('[', '').replace(']', '')
        paes.append([float(v) for v in row.split()])
    return np.array(paes)


def calc_precision_recall_for_one_example(
    row: pd.Series,
    thresh: float,
    val: float,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    higher_better: bool = True
) -> Tuple[int, int, int, int]:
    """
    Add to the number of true/false positives/negatives and return their values.

    Classifies a single example based on a threshold and updates confusion matrix
    counts accordingly.

    Parameters
    ----------
    row : pd.Series
        DataFrame row containing the 'stability_label' field.
    thresh : float
        Threshold value for classification.
    val : float
        Metric value to compare against threshold.
    tp : int
        Current count of true positives.
    fp : int
        Current count of false positives.
    tn : int
        Current count of true negatives.
    fn : int
        Current count of false negatives.
    higher_better : bool, optional
        If True, values above threshold are considered positive predictions.
        If False (e.g., for pAE), values below threshold are positive. Default is True.

    Returns
    -------
    tp : int
        Updated count of true positives.
    fp : int
        Updated count of false positives.
    tn : int
        Updated count of true negatives.
    fn : int
        Updated count of false negatives.

    Notes
    -----
    If lower (positive) values are considered higher confidence (like with pAE),
    then multiply the pAE values by -1 to use the same logic below as for pLDDT.
    """
    # If lower (positive) values are considered higher confidence
    # (like with pAE), then multiply the pAE values by -1 to use
    # the same logic below as for pLDDT.
    if not higher_better:
        val = -1 * val
        thresh = -1 * thresh
    if row['stability_label'] == 'stable':
        if val > thresh:
            tp += 1
        else:
            fn += 1
    elif row['stability_label'] == 'unstable':
        if val > thresh:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def get_precision_value(tp: int, fp: int) -> float:
    """
    Calculate precision from true positive and false positive counts.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.

    Returns
    -------
    float
        Precision value. Returns 1 if no predictions were made (tp + fp == 0).

    Notes
    -----
    Precision = TP / (TP + FP)
    """
    if (tp + fp) == 0:
        return 1
    else:
        return tp / (tp + fp)


def get_recall_value(tp: int, fn: int) -> float:
    """
    Calculate recall from true positive and false negative counts.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    float
        Recall value. Returns 0 if no positive examples exist (tp + fn == 0).

    Notes
    -----
    Recall = TP / (TP + FN)
    """
    if (tp + fn) == 0:
        return 0
    else:
        return tp / (tp + fn)


def get_false_positive_rate(fp: int, tn: int) -> float:
    """
    Calculate false positive rate from false positive and true negative counts.

    Parameters
    ----------
    fp : int
        Number of false positives.
    tn : int
        Number of true negatives.

    Returns
    -------
    float
        False positive rate. Returns 0 if no negative examples exist (fp + tn == 0).

    Notes
    -----
    FPR = FP / (FP + TN)
    """
    if (fp + tn) == 0:
        return 0
    else:
        return fp / (fp + tn)


def count_confusion_matrix_values(
    args: argparse.Namespace,
    thresh: float,
    sub_plot_df: pd.DataFrame
) -> Tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]:
    """
    Calculate values in the pLDDT and pAE confusion matrices for a given threshold.

    Computes true positives, false positives, true negatives, and false negatives
    for both classic and fine-tuned models using both pLDDT and pAE metrics.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration options.
    thresh : float
        Threshold value for classification.
    sub_plot_df : pd.DataFrame
        DataFrame containing model predictions and stability labels.

    Returns
    -------
    c_plddt_tp : int
        Classic model pLDDT true positives.
    c_plddt_fp : int
        Classic model pLDDT false positives.
    c_plddt_tn : int
        Classic model pLDDT true negatives.
    c_plddt_fn : int
        Classic model pLDDT false negatives.
    f_plddt_tp : int
        Fine-tuned model pLDDT true positives.
    f_plddt_fp : int
        Fine-tuned model pLDDT false positives.
    f_plddt_tn : int
        Fine-tuned model pLDDT true negatives.
    f_plddt_fn : int
        Fine-tuned model pLDDT false negatives.
    c_pae_tp : int
        Classic model pAE true positives.
    c_pae_fp : int
        Classic model pAE false positives.
    c_pae_tn : int
        Classic model pAE true negatives.
    c_pae_fn : int
        Classic model pAE false negatives.
    f_pae_tp : int
        Fine-tuned model pAE true positives.
    f_pae_fp : int
        Fine-tuned model pAE false positives.
    f_pae_tn : int
        Fine-tuned model pAE true negatives.
    f_pae_fn : int
        Fine-tuned model pAE false negatives.

    Notes
    -----
    Requires 'plddt_metric' and 'pae_metric' to be defined in the calling scope.
    """
    ## For the classic AF2 model:
    # True/False Positive/Negative for calculating Precision and Recall
    # These values are used when classifying by both pLDDT and pAE
    c_tp, c_fp, c_tn, c_fn = 0, 0, 0, 0
    # These values are used when classifying by only pLDDT
    c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn = 0, 0, 0, 0
    # These values are used when classifying by only pAE
    c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn = 0, 0, 0, 0
    ## For the fine-tuned AF2 model:
    # True/False Positive/Negative for calculating Precision and Recall
    # These values are used when classifying by both pLDDT and pAE
    f_tp, f_fp, f_tn, f_fn = 0, 0, 0, 0
    # These values are used when classifying by only pLDDT
    f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn = 0, 0, 0, 0
    # These values are used when classifying by only pAE
    f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn = 0, 0, 0, 0

    for idx, row in sub_plot_df.iterrows():

        model_type = row['model_type']
        plddt = row[plddt_metric]
        pae = row[pae_metric]

        if model_type == 'classic':
            c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn = calc_precision_recall_for_one_example(
                row, thresh, plddt, c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn,
                higher_better=True
            )
            c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn = calc_precision_recall_for_one_example(
                row, thresh, pae, c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn,
                higher_better=False
            )

        elif model_type == 'ft':
            f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn = calc_precision_recall_for_one_example(
                row, thresh, plddt, f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn,
                higher_better=True
            )
            f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn = calc_precision_recall_for_one_example(
                row, thresh, pae, f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn,
                higher_better=False
            )

    if args.verbose:
        print('Precision and Recall for checkpointed model after %d training steps for %s with threshold %d' % (
            sub_plot_df.iloc[0]['training_step'], dataset, thresh
        ))
        print('c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn')
        print('%d %d %d %d' % (c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn))
        print('f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn')
        print('%d %d %d %d' % (f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn))
        print('c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn')
        print('%d %d %d %d' % (c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn))
        print('f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn')
        print('%d %d %d %d' % (f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn))

    return c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn, \
        f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn, \
        c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn, \
        f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn


def plot_precision_recall_curve(
    df: pd.DataFrame,
    metric_type: Optional[str] = None,
    outpath: Optional[str] = None
) -> None:
    """
    Plot a Precision-Recall curve for the provided df.

    Creates a visualization showing the trade-off between precision and recall
    for different classification thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'precision', 'recall', 'metric_type', and 'model_type'
        columns.
    metric_type : str or None, optional
        Type of metric to plot (e.g., 'plddt', 'pae'). Default is None.
    outpath : str or None, optional
        Path where the figure will be saved. Default is None.

    Returns
    -------
    None
        Saves the plot to disk if outpath is provided.

    Notes
    -----
    The plot displays precision vs. recall curves for both classic and fine-tuned
    models for the specified metric type.
    """
    # sns.set_theme()
    sns.set_theme(style="whitegrid")
    fig = plt.figure()
    ax = fig.add_axes([0.04, 0.2, 0.6, 0.6])
    ax.set_xlabel('Recall', fontsize=15)
    ax.set_ylabel('Precision', fontsize=15)
    ax.set_title(
        'Precision vs. Recall for Checkpointed\nModel at %d Training Steps\nClassify by %s' % (
            df.iloc[0]['training_step'], metric_type
        ),
        fontsize=12
    )

    # df = df.sort_values(by='precision')

    sub_df = df.loc[df['metric_type'] == metric_type]

    ## If plotting multiple datasets
    # lp = sns.lineplot(data=df, x='recall', y='precision', hue=df[['model_type','dataset']].apply(tuple, axis=1))
    ## If plotting only one dataset
    lp = sns.lineplot(data=sub_df, x='recall', y='precision', hue='model_type', ci=None)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # Change legend fontsize
    plt.setp(lp.get_legend().get_texts(), fontsize=20)

    # Add a line from (0,1) to (1,0)
    # lims = [[0,1],[1,0]]
    # ax.plot(lims[0], lims[1], '--', alpha=0.5, zorder=100)
    # plt.tight_layout() # not compatible with these fig axes.
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    plt.savefig(outpath, dpi=300)
    print('Saved figure: %s.png' % outpath)
    plt.close()


def plot_tpr_fpr_curve(
    df: pd.DataFrame,
    metric_type: Optional[str] = None,
    outpath: Optional[str] = None
) -> None:
    """
    Plot a True Positive Rate vs. False Positive Rate curve (ROC curve).

    Creates a visualization showing the trade-off between TPR and FPR for different
    classification thresholds, and computes the AUROC for both models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'recall' (TPR), 'FPR', 'metric_type', 'model_type',
        'training_step', and 'dataset' columns.
    metric_type : str or None, optional
        Type of metric to plot (e.g., 'plddt', 'pae'). Default is None.
    outpath : str or None, optional
        Path where the figure will be saved. Default is None.

    Returns
    -------
    None
        Saves the plot to disk if outpath is provided.

    Notes
    -----
    The function calculates and displays AUROC values for both classic and
    fine-tuned models directly on the plot.
    """
    sns.set(font_scale=1.5)
    sns.set_theme(style="whitegrid")

    # sns.set_theme()
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.2, 0.6, 0.6])
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.set_ylabel('True Positive Rate', fontsize=15)
    ax.set_title(
        'TPR vs. FPR for Checkpointed\nModel at %d Training Steps.\n%s. Classify by %s' % (
            df.iloc[0]['training_step'], df.iloc[0]['dataset'], metric_type
        ),
        fontsize=12
    )

    # Only plot one metric
    df = df.loc[df['metric_type'] == metric_type]

    # df = df.sort_values(by='FPR')

    # ci=None turns off confidence intervals.
    lp = sns.lineplot(
        data=df, x='FPR', y='recall', hue='model_type',
        hue_order=['classic', 'ft'], ci=None
    )  # hue=df[['model_type','dataset']].apply(tuple, axis=1)

    # Calculate AUROC
    df = df.sort_values(by='FPR')
    xc = df.loc[df['model_type'] == 'classic']['FPR'].tolist()
    yc = df.loc[df['model_type'] == 'classic']['recall'].tolist()
    auc_c = auc(xc, yc)

    xf = df.loc[df['model_type'] == 'ft']['FPR'].tolist()
    yf = df.loc[df['model_type'] == 'ft']['recall'].tolist()
    auc_f = auc(xf, yf)

    plt.text(0.2, 0.14, 'Classic AUROC = %.3f' % auc_c, fontsize=12)
    plt.text(0.2, 0.06, 'Fine-tuned AUROC = %.3f' % auc_f, fontsize=12)

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # Change legend fontsize
    plt.setp(lp.get_legend().get_texts(), fontsize='20')

    # Add a line from (0,1) to (1,0)
    # lims = [[0,1],[1,0]]
    # ax.plot(lims[0], lims[1], '--', alpha=0.5, zorder=100)
    # plt.tight_layout() # not compatible with these fig axes.
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    plt.savefig(outpath, dpi=300)
    print('Saved figure: %s.png' % outpath)
    plt.close()


def plot_rmsds_for_each_dataset(
    df: pd.DataFrame,
    args: argparse.Namespace,
    rmsd_col_ending: str = '_gt_rmsd',
    title_append: str = '',
    filename_append: str = '',
    outfolder: str = ''
) -> None:
    """
    Plot RMSD to Rosetta values for each example over dataset name.

    Creates strip plots showing RMSD distributions for both classic and fine-tuned
    AF2 predictions, organized by dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing RMSD data with 'training_step' and 'ground_truth_parent'
        columns.
    args : argparse.Namespace
        Command-line arguments containing configuration options.
    rmsd_col_ending : str, optional
        Suffix for RMSD column names. Default is '_gt_rmsd'.
    title_append : str, optional
        Additional text to append to plot titles. Default is ''.
    filename_append : str, optional
        Additional text to append to output filenames. Default is ''.
    outfolder : str, optional
        Directory path where outputs will be saved. Default is ''.

    Returns
    -------
    None
        Saves plots to disk and displays them.

    Notes
    -----
    The function creates different plot layouts depending on whether data from
    single or multiple datasets is being visualized.
    """
    # sns.set_theme()
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    for training_step in natsorted(df['training_step'].unique().tolist()):

        sub_df = df.loc[df['training_step'] == training_step]

        first_df = True

        for ground_truth_parent in natsorted(df['ground_truth_parent'].unique().tolist()):

            # Second to last subfolder in the ground_truth_parent is the name of the de novo protein dataset
            # For example, /wynton/group/kortemme/borr/alphafold_finetune/data/abba_rocklin_rd6/pdbs
            dataset_name = ground_truth_parent.split('/')[-2]

            # Slice the df to include only examples from the current ground_truth_parent
            sub_df_2 = sub_df.loc[sub_df['ground_truth_parent'] == ground_truth_parent]

            classic_tuples = list(zip(
                [dataset_name] * len(sub_df_2),
                ['Classic'] * len(sub_df_2),
                sub_df_2['classic' + rmsd_col_ending]
            ))
            ft_tuples = list(zip(
                [dataset_name] * len(sub_df_2),
                ['Fine-tuned'] * len(sub_df_2),
                sub_df_2['ft' + rmsd_col_ending]
            ))
            list_of_tuples = classic_tuples + ft_tuples

            # Make a new df in an sns-friendly format.
            temp_df = pd.DataFrame(
                list_of_tuples,
                columns=['Dataset', 'Model', 'RMSD to Rosetta']
            )

            if not first_df:
                plot_df = pd.concat([plot_df, temp_df], ignore_index=True)
                plot_df.reset_index()
            else:
                plot_df = temp_df
                first_df = False

        if len(df['ground_truth_parent'].unique().tolist()) == 1:
            # I should
            fig = plt.figure(figsize=[12, 8])
            ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
            sns.set_theme(style="whitegrid")
            ax.set_xlabel('Model', fontsize=25)
            ax.set_ylabel('RMSD to Rosetta', fontsize=25)
            ax.set_title(
                'RMSD to Rosetta for Classic and\nFine-Tuned AF2 models, %s Training Steps%s' % (
                    training_step, title_append
                ),
                fontsize=25
            )
            sp = sns.stripplot(
                y='RMSD to Rosetta', x='Model', data=plot_df,
                dodge=True, jitter=True, alpha=0.5
            )
        else:
            fig = plt.figure(figsize=[16, 9])
            ax = fig.add_axes([0.08, 0.2, 0.6, 0.6])
            sns.set_theme(style="whitegrid")
            ax.set_xlabel('Model', fontsize=25)
            ax.set_ylabel('RMSD to Rosetta', fontsize=25)
            ax.set_title(
                'RMSD to Rosetta for Classic and\nFine-Tuned AF2 models, %s Training Steps%s' % (
                    training_step, title_append
                ),
                fontsize=25
            )
            sp = sns.stripplot(
                y='RMSD to Rosetta', x='Model', data=plot_df, hue='Dataset',
                dodge=True, jitter=True, alpha=0.5
            )
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            # Change legend fontsize
            plt.setp(sp.get_legend().get_texts(), fontsize='20')

        plt.savefig(
            os.path.join(
                outfolder, '%s_step_%s_rmsd%s_points.png' % (
                    args.job_name, training_step, filename_append
                )
            ),
            dpi=300
        )
        print('Saved figure: %s' % (
            os.path.join(
                outfolder, '%s_step_%s_rmsd%s_points.png' % (
                    args.job_name, training_step, filename_append
                )
            )
        ))
        plt.show()
        plt.close()


def violinplot_rmsds_for_each_dataset(
    df: pd.DataFrame,
    args: argparse.Namespace,
    rmsd_col_ending: str = '_gt_rmsd',
    title_append: str = '',
    filename_append: str = '',
    outfolder: str = ''
) -> None:
    """
    Plot violin plots of RMSD to Rosetta values for each example over dataset name.

    Creates violin plots showing RMSD distributions for both classic and fine-tuned
    AF2 predictions, split by stability label.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing RMSD data with 'training_step', 'ground_truth_parent',
        and 'stability_label' columns.
    args : argparse.Namespace
        Command-line arguments containing configuration options.
    rmsd_col_ending : str, optional
        Suffix for RMSD column names. Default is '_gt_rmsd'.
    title_append : str, optional
        Additional text to append to plot titles. Default is ''.
    filename_append : str, optional
        Additional text to append to output filenames. Default is ''.
    outfolder : str, optional
        Directory path where outputs will be saved. Default is ''.

    Returns
    -------
    None
        Saves plots to disk and displays them.

    Notes
    -----
    The function creates different plot layouts depending on whether data from
    single or multiple datasets is being visualized. Violins are split by
    stability label to show distributions for stable vs. unstable designs.
    """
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    for training_step in natsorted(df['training_step'].unique().tolist()):

        sub_df = df.loc[df['training_step'] == training_step]

        first_df = True

        for ground_truth_parent in natsorted(df['ground_truth_parent'].unique().tolist()):

            # Second to last subfolder in the ground_truth_parent is the name of the de novo protein dataset
            # For example, /wynton/group/kortemme/borr/alphafold_finetune/data/abba_rocklin_rd6/pdbs
            dataset_name = ground_truth_parent.split('/')[-2]

            # Slice the df to include only examples from the current ground_truth_parent
            sub_df_2 = sub_df.loc[sub_df['ground_truth_parent'] == ground_truth_parent]

            classic_tuples = list(zip(
                [dataset_name] * len(sub_df_2),
                ['Classic'] * len(sub_df_2),
                sub_df_2['classic' + rmsd_col_ending],
                sub_df_2['stability_label']
            ))
            ft_tuples = list(zip(
                [dataset_name] * len(sub_df_2),
                ['Fine-tuned'] * len(sub_df_2),
                sub_df_2['ft' + rmsd_col_ending],
                sub_df_2['stability_label']
            ))
            list_of_tuples = classic_tuples + ft_tuples

            # Make a new df in an sns-friendly format.
            temp_df = pd.DataFrame(
                list_of_tuples,
                columns=['Dataset', 'Model', 'RMSD to Rosetta', 'stability_label']
            )

            if not first_df:
                plot_df = pd.concat([plot_df, temp_df], ignore_index=True)
                plot_df.reset_index()
            else:
                plot_df = temp_df
                first_df = False

        if len(df['ground_truth_parent'].unique().tolist()) == 1:
            fig = plt.figure(figsize=[12, 7])
            ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
            sns.set_theme(style="whitegrid")
            ax.set_xlabel('Model', fontsize=25)
            ax.set_ylabel('RMSD to Rosetta', fontsize=25)
            ax.set_title(
                'RMSD to Rosetta for Classic and\nFine-Tuned AF2 models, %s Training Steps%s' % (
                    training_step, title_append
                ),
                fontsize=25
            )
            vp = sns.violinplot(
                y='RMSD to Rosetta', x='Model', data=plot_df,
                hue='stability_label', split=True
            )

        else:
            fig = plt.figure(figsize=[16, 9])
            ax = fig.add_axes([0.08, 0.2, 0.6, 0.6])
            sns.set_theme(style="whitegrid")
            ax.set_xlabel('Model', fontsize=25)
            ax.set_ylabel('RMSD to Rosetta', fontsize=25)
            ax.set_title(
                'RMSD to Rosetta for Classic and\nFine-Tuned AF2 models, %s Training Steps%s' % (
                    training_step, title_append
                ),
                fontsize=25
            )
            vp = sns.violinplot(
                y='RMSD to Rosetta', x='Model', data=plot_df,
                hue='stability_label', split=True
            )  # hue='Dataset')
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            # Change legend fontsize
            plt.setp(vp.get_legend().get_texts(), fontsize='20')

        plt.savefig(
            os.path.join(
                outfolder, '%s_step_%s_rmsd%s_distributions.png' % (
                    args.job_name, training_step, filename_append
                )
            ),
            dpi=300
        )
        print('Saved figure: %s' % (
            os.path.join(
                outfolder, '%s_step_%s_rmsd%s_distributions.png' % (
                    args.job_name, training_step, filename_append
                )
            )
        ))
        plt.show()
        plt.close()


def get_confidence_values(
    args: argparse.Namespace,
    new_df: pd.DataFrame,
    row: pd.Series,
    stab_row: pd.Series
) -> Tuple[float, float, float, float]:
    """
    Load pLDDT and pAE values for the current design.

    Extracts and processes confidence metrics (pLDDT and pAE) from stored string
    representations, handling potential padding artifacts from AlphaFold predictions.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration options including verbose flag.
    new_df : pd.DataFrame
        DataFrame for storing processed confidence values (not modified in this function).
    row : pd.Series
        DataFrame row containing 'classic_plddt', 'ft_plddt', 'classic_pae', and
        'ft_pae' columns.
    stab_row : pd.Series
        DataFrame row from stability data containing 'sequence' or 'protein_sequence_c'
        column.

    Returns
    -------
    c_pae : float
        Mean pAE value for classic model over actual sequence length.
    f_pae : float
        Mean pAE value for fine-tuned model over actual sequence length.
    c_plddt : float
        Mean pLDDT value for classic model over actual sequence length.
    f_plddt : float
        Mean pLDDT value for fine-tuned model over actual sequence length.

    Notes
    -----
    Get the sequence length of each example, and only analyze the first num_seq
    positions in the pae and plddt arrays. I believe this is from a bug in the
    alphafold_finetune code, must pad its input representations, and outputs
    pLDDT and pAE matrices that include low values for padded positions.

    Select the first seq_len residues from which to calculate average pLDDT and
    pAE scores. alphafold_finetune thresholds are pLDDT / 100 and pAE / 31

    That these values should be divided by 100 and 10 may be a faulty assumption.
    I need to see how the binder_intercepts are used in alphafold_finetune!
    For now, it may make more sense to simply plot the distributions of confidence
    values for stable and unstable designs, before and after AF2 fine-tuning.

    The output confidence arrays from alphafold_finetune appear to have padded values,
    so just use the array indices that correspond with actual residue indices.
    """
    # Get the sequence length of each example, and only analyze the first num_seq
    # positions in the pae and plddt arrays. I believe this is from a bug in the
    # alphafold_finetune code, must pad its input representations, and outputs
    # pLDDT and pAE matrices that include low values for padded positions.
    try:
        seq_len = len(stab_row['sequence'])
    except:  # The Rocklin dataframes use this column for protein sequence
        seq_len = len(stab_row['protein_sequence_c'])

    # Select the first seq_len residues from which to calculate
    # average pLDDT and pAE scores.
    # alphafold_finetune thresholds are pLDDT / 100 and pAE / 31

    # The arrays of per-residue confidence values, saved as strings,
    # need to be reformatted
    c_plddts = load_plddts(row['classic_plddt'])
    f_plddts = load_plddts(row['ft_plddt'])

    c_paes = load_paes(row['classic_pae'])
    f_paes = load_paes(row['ft_pae'])

    ## That these values should be divided by 100 and 10 may be a faulty assumption.
    ## I need to see how the binder_intercepts are used in alphafold_finetune!
    ## For now, it may make more sense to simply plot the distributions of confidence values
    ## for stable and unstable designs, before and after AF2 fine-tuning.
    c_plddt = np.mean(c_plddts[:seq_len])
    f_plddt = np.mean(f_plddts[:seq_len])

    ## The output confidence arrays from alphafold_finetune appear to have padded values,
    ## so just use the array indices that correspond with actual residue indices.
    c_pae = np.mean(c_paes[:seq_len, :seq_len])
    f_pae = np.mean(f_paes[:seq_len, :seq_len])

    if args.verbose:
        print('Classic and Fine-tuned pLDDTs')
        print(c_plddt)
        print(f_plddt)

        print('Classic and Fine-tuned pAEs')
        print(c_pae)
        print(f_pae)

    return c_pae, f_pae, c_plddt, f_plddt
>>>>>>> be02a1e (lucs_af refactor and cleanup)


##########


if __name__ == '__main__':

<<<<<<< HEAD
	# Parse command line arguments.
	parser = argparse.ArgumentParser(
		description='Analyze RMSDs between classic and fine-tuned AF2 predictions to their ground truth structures,\
					as well as the classic and fine-tuned AF2 confidence metrics.',
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('--job_name', type=str, required=True,
			help='Name for the job which will be used in the output .png files.')
	parser.add_argument('--checkpoint_model_data_folder', type=str, required=True,
			help='Folder containing dataframe(s) output by calc_checkpointed_models_metrics.py.')
	parser.add_argument('--outfolder_parent', type=str, required=True,
			help='Output files are written to outfolder_parent.')

	parser.add_argument('--rmsd_analysis', action='store_true',
			help="Analyze the RMSDs to the ground truth structures for each checkpointed model's predictions.")
	parser.add_argument('--stability_data', nargs='*', type=str, required=False, default=None,
			help='Dataframe(s) containing a "stability_label" column with "stable," "unstable," and "uncertain" values,\
			as well as a "sequence" column, which is used to get the sequence length.')
	parser.add_argument('--rmsd_thresh', type=float, required=False, default=1.5,
			help='Threshold for AF2 to ground truth RMSD values for plotting the fraction of designs predicted to\
			sub-<rmsd_thresh> RMSD to the ground truth structures.')
	parser.add_argument('--plot_rmsd_distributions', action='store_true',
			help='Plot distributions of RMSD values for stable vs. unstable designs.')
	parser.add_argument('--reshaped_helix_rmsd', action='store_true',
			help='Plot Reshaped Helix RMSD to Rosetta values for LUCS designs.')

	parser.add_argument('--stability_classifier', action='store_true',
			help='Analyze the Precision and Recall of the AF2 confidence scores in predicting designs as stable\
			and unstable.  Uses the default pLDDT and pAE thresholds or the provided thresholds.')
	parser.add_argument('--plddt_thresh', type=float, required=False, default=0.80367635,
			help='Threshold for AF2 pLDDT values in classifying a design as stable (if the AF2 average pLDDT for a design \
			is greater than this threshold, then the design is classified as "stable"). Default is 0.80367635 (taken from the\
			alphafold_finetune github repo: https://github.com/phbradley/alphafold_finetune).')
	parser.add_argument('--pae_thresh', type=float, required=False, default=0.43373787,
			help='Threshold for AF2 pAE values in classifying a design as stable (if the AF2 average pAE for a design \
			is greater than this threshold, then the design is classified as "stable"). Default is 0.43373787 (taken from the\
			alphafold_finetune github repo: https://github.com/phbradley/alphafold_finetune).')
	parser.add_argument('--plot_confidence_distributions', action='store_true',
			help='Plot the distributions of the confidence scores for the classic and fine-tuned models\
			at each training step.  Requires a outfolder_parent/job_name/plotting_dfs folder to exist with outputs from the \
			--stability_classifier outputs.')
	parser.add_argument('--precision_recall', action='store_true',
			help='Generate Precision vs. Recall curves for classifying stable designs at different pLDDT and \
			pAE thresholds.')
	parser.add_argument('--reshaped_residue_stability', action='store_true',
			help='Analyze the Precision and Recall of the AF2 confidence scores in predicting designs as stable\
			and unstable using the pAE values between the bb_remodeled_residues and non-bb_remodeled_residues and\
			pLDDT values of the bb_remodeled_residues (this residue set is found in the bb_remodeled_residues column\
			of the stability_data dataframe(s)). Uses the default pLDDT and pAE thresholds or the provided thresholds.')

	parser.add_argument('--verbose', action='store_true',
			help='Print verbose output to the command line.')

	args = parser.parse_args()

	outfolder = os.path.join(args.outfolder_parent)
	os.makedirs(outfolder, exist_ok=True)

	# Combine the checkpoint_model_dataframes into one df
	df_paths = [os.path.join(args.checkpoint_model_data_folder, f) for \
				f in os.listdir(args.checkpoint_model_data_folder) if '.csv' in f and 'prediction_analysis' not in f]
	dfs = [pd.read_csv(p) for p in df_paths]
	df = pd.concat(dfs)

	# If stability data is provided, add this data to the df.
	if args.stability_data != None:

		# Load the stability_data df(s)
		stab_dfs = [pd.read_csv(sdf) for sdf in args.stability_data]
		stab_df = pd.concat(stab_dfs)

		for idx, row in df.iterrows():

			# Half of the Basanta NTF2s' ground truth files include the string '_BasicBeNTF2_designed_0000' after the design name
			name = row['ground_truth_path'].rsplit('/',1)[1].replace('.pdb','').replace('_BasicBeNTF2_designed_0000','')

			# Names to check for in the 'name' column of the stab_df
			possible_names = [name, name+'.pdb', row['ground_truth_path'].rsplit('/',1)[1].replace('.pdb','')]
			try:
				possible_names.append(int(name))
			except:
				pass

			# The rocklin_miniproteins_2017 stability df has .pdb at the end of the design names
			# In case the stab_df 'name' column contains int values, check for int(name)
			df.loc[idx, 'stability_label'] = stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label']

	# Analyze the RMSDs to the ground truth structures for each checkpointed model's predictions
	if args.rmsd_analysis:

		# For each unique dataset, plot the fractions of designs with lower RMSDs to the ground 
		# truth structure than the other model (classic or fine-tuned) for each training step 
		# for both models.
		for idx, row in df.iterrows():
			df.loc[idx, 'ground_truth_parent'] = row['ground_truth_path'].rsplit('/',1)[0]

		rmsd_col_endings = ['_gt_rmsd']
		rmsd_col_names = ['\n(Whole Structure RMSD)']
		if args.reshaped_helix_rmsd:
			# Calculate RMSDs for both whole structures and reshaped helices.
			rmsd_col_endings.append('_helix_rmsd')
			rmsd_col_names.append('\n(Reshaped Helix RMSD)')

		for rmsd_col_ending, rmsd_col_name in zip(rmsd_col_endings, rmsd_col_names):

			## Only plot RMSDs over training steps if there are dataframes for multiple training steps.
			if len(df_paths) != 1:
				# Plot the fractions of designs with lower RMSDs to the ground truth structure than the
				# other model (classic or fine-tuned) for each training step for both models.
				make_all_rmsd_frac_over_training_step_figures(args, df, title_append=rmsd_col_name, 
					filename_append=rmsd_col_ending, outfolder=outfolder, rmsd_col_ending=rmsd_col_ending)

			plot_rmsds_for_each_dataset(df, args, rmsd_col_ending=rmsd_col_ending,  title_append=rmsd_col_name, filename_append=rmsd_col_ending, outfolder=outfolder)

			# This violinplot uses stability labels from args.stability_data
			if args.stability_data != None:
				violinplot_rmsds_for_each_dataset(df, args, rmsd_col_ending=rmsd_col_ending, title_append=rmsd_col_name, filename_append=rmsd_col_ending, outfolder=outfolder)		

			for ground_truth_parent in natsorted(df['ground_truth_parent'].unique().tolist()):

				# Slice the df to include only examples from the current ground_truth_parent
				sub_df = df.loc[df['ground_truth_parent'] == ground_truth_parent]

				# Second to last subfolder in the ground_truth_parent is the name of the de novo protein dataset
				# For example, /wynton/group/kortemme/borr/alphafold_finetune/data/abba_rocklin_rd6/pdbs
				dataset_name = ground_truth_parent.split('/')[-2]

				if len(df_paths) != 1:
					## Only plot RMSDs over training steps if there are dataframes for multiple training steps.
					make_all_rmsd_frac_over_training_step_figures(args, sub_df
							title_append=' for %s%s'%(dataset_name, rmsd_col_name), filename_append='_'+dataset_name+'_'+rmsd_col_ending, 
							outfolder=outfolder)

	# Analyze the Precision and Recall of the AF2 confidence scores in predicting designs as stable\
	# and unstable.  Uses the default pLDDT and pAE thresholds or the provided thresholds.
	if args.stability_classifier:

		# Ensure the a path to a stability df has been provided
		# There's a better way to do this, with argparse dependent arguments.
		if args.stability_data == None:
			print('If using the --stability_classifier flag, you must use the --stability_data flag to provide \
				a path to a dataframe (.csv file) with "stability_label" columns' )
		else:
			# Iterate over each checkpointed model's RMSD+confidence score dataframe.
			for df_path in natsorted(df_paths):

				# New df formatted for plotting with seaborn
				if args.reshaped_helix_rmsd:
					new_df = pd.DataFrame(columns=['name', 'dataset','training_step','plddt','pae','stability_label','model_type','rmsd_to_gt','helix_rmsd_to_gt','reshaped_plddt','reshaped_pae'])
				else:
					new_df = pd.DataFrame(columns=['name', 'dataset','training_step','plddt','pae','stability_label','model_type','rmsd_to_gt'])

				t_step_df = pd.read_csv(df_path)

				training_step = t_step_df.iloc[0]['training_step']

				# Load the stability_data df(s)
				stab_dfs = [pd.read_csv(sdf) for sdf in args.stability_data]
				stab_df = pd.concat(stab_dfs)

				# Add columns to the new_df (which is used for plotting confidence score distributions for
				# stable and unstable designs) with data from the checkpointed model's df.
				# Calculate Precision and Recall for predicting stable designs, based on the classic and fine-tuned AF2 
				# models' confidence metrics and the args.plddt_thresh and args.pae_thresh (average confidence values
				# greater than these thresholds for a sequence are considered to be a prediction of that sequence
				# folding into a stable protein).
				# Split Precision and Recall analysis by ground truth folder (dataset).
				# Temporarily add ground_truth_parent, name, and stability_label columns to the t_step_df
				new_idx = 0
				for idx, row in t_step_df.iterrows():

					t_step_df.loc[idx, 'ground_truth_parent'] = row['ground_truth_path'].rsplit('/',1)[0]
					name = row['ground_truth_path'].rsplit('/',1)[1].replace('.pdb','')

					# The Basanta NTF2s' ground truth files include the string '_BasicBeNTF2_designed_0000' after the design name
					name = name.replace('_BasicBeNTF2_designed_0000','')

					# Names to check for in the 'name' column of the stab_df
					possible_names = [name, name+'.pdb', row['ground_truth_path'].rsplit('/',1)[1].replace('.pdb','')]
					try:
						possible_names.append(int(name))
					except:
						pass

					t_step_df.loc[idx, 'name'] = name

					# The rocklin_miniproteins_2017 stability df has .pdb at the end of the design names
					# In case the stab_df 'name' column contains int values, check for int(name)
					t_step_df.loc[idx, 'stability_label'] = stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label']

					# Add columns to the new dataframe (in a format friendly for seaborn plotting)
					# Columns of new_df are columns=['name','dataset','training_step','plddt','pae','stability_label','model_type','rmsd_to_gt']
					## If including reshaped residue metrics, then the df has the additional columns: 'helix_rmsd_to_gt','reshaped_plddt','reshaped_pae']
					if args.reshaped_helix_rmsd:
						new_df.loc[new_idx] = [name, '', training_step, 0, 0, stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label'],'classic', 0, 0, 0, 0]
						new_df.loc[new_idx+1] = [name, '', training_step, 0, 0, stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label'],'ft', 0, 0, 0, 0]
						new_idx+=2
					else:
						new_df.loc[new_idx] = [name, '', training_step, 0, 0, stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label'],'classic', 0]
						new_df.loc[new_idx+1] = [name, '', training_step, 0, 0, stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label'],'ft', 0]
						new_idx+=2

				for ground_truth_parent in natsorted(t_step_df['ground_truth_parent'].unique().tolist()):

					# Slice the df to include only examples from the current ground_truth_parent
					sub_df = t_step_df.loc[t_step_df['ground_truth_parent'] == ground_truth_parent]

					# Second to last subfolder in the ground_truth_parent is the name of the de novo protein dataset
					# For example, /wynton/group/kortemme/borr/alphafold_finetune/data/abba_rocklin_rd6/pdbs
					dataset_name = ground_truth_parent.split('/')[-2]

					# Iterate over the rows in the dataset-specific checkpointed model df
					for idx, row in sub_df.iterrows():

						# Names to check for in the 'name' column of the stab_df
						possible_names = [row['name'], row['name']+'.pdb', row['ground_truth_path'].rsplit('/',1)[1].replace('.pdb','')]
						try:
							possible_names.append(int(row['name']))
						except:
							pass

						# To select the corresponding row in the stability df
						stab_row = stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]

						# Add the dataset name to the new_df (this df is used for plotting with seaborn)
						new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='classic'), 'dataset'] = dataset_name
						new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='ft'), 'dataset'] = dataset_name

						# Add the predicted structure to ground truth RMSD values
						new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='classic'), 'rmsd_to_gt'] = row['classic_gt_rmsd']
						new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='ft'), 'rmsd_to_gt'] = row['ft_gt_rmsd']

						if args.reshaped_helix_rmsd:
							# Add the predicted structure to ground truth helix RMSD values
							new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='classic'), 'helix_rmsd_to_gt'] = row['classic_helix_rmsd']
							new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='ft'), 'helix_rmsd_to_gt'] = row['ft_helix_rmsd']
						
						c_pae, f_pae, c_plddt, f_plddt = get_confidence_values(args, new_df, row, stab_row, possible_names)

						# Add the average pLDDT values to the new_df (used for plotting with seaborn)
						new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='classic'), 'plddt'] = c_plddt
						new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='ft'), 'plddt'] = f_plddt

						# Add the average pAE values to the new_df (used for plotting with seaborn)
						new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='classic'), 'pae'] = c_pae
						new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='ft'), 'pae'] = f_pae

						if args.reshaped_residue_stability:
							# Add the average reshaped residue pLDDT values to the new_df (used for plotting with seaborn)
							new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='classic'), 'reshaped_plddt'] = row['classic_reshaped_plddt']
							new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='ft'), 'reshaped_plddt'] = row['ft_reshaped_plddt']

							# Add the average reshaped residue pAE values to the new_df (used for plotting with seaborn)
							new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='classic'), 'reshaped_pae'] = row['classic_reshaped_pae']
							new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type']=='ft'), 'reshaped_pae'] = row['ft_reshaped_pae']
							

				# Saving thec new_df (which is in a format that makes plotting with
				# seaborn convenient).
				os.makedirs(os.path.join(outfolder, 'plotting_dfs'), exist_ok=True)
				new_df.to_csv(os.path.join(outfolder, 'plotting_dfs','plotting_df_%s_trainstep_%d.csv'%(args.job_name, training_step)))
				print('Saved dataframe %s'%os.path.join(outfolder, 'plotting_dfs','plotting_df_%s_trainstep_%d.csv'%(args.job_name, training_step)))

	# Plot the distributions of confidence values for classic and fine-tuned AF2, for each training step, and
	# for each dataset (as well as all the datasets at once).
	if args.plot_confidence_distributions:

		for df_file in natsorted([f for f in os.listdir(os.path.join(outfolder, 'plotting_dfs')) if '.csv' in f]):

			training_step = df_file.split('.')[0].split('_')[-1]

			plot_df = pd.read_csv(os.path.join(outfolder, 'plotting_dfs', df_file))

			print('Number of examples in %s: %d'%(df_file, len(plot_df)))

			# Make an output folder for the generated figures.
			os.makedirs(os.path.join(outfolder, 'plotting_dfs', 'figures'), exist_ok=True)

			if args.reshaped_helix_rmsd:
				x_y_hue = [('stability_label', 'plddt', 'model_type'), ('stability_label', 'pae', 'model_type'),
								('stability_label', 'reshaped_plddt', 'model_type'), ('stability_label', 'reshaped_pae', 'model_type')]
			else:
				x_y_hue = [('stability_label', 'plddt', 'model_type'), ('stability_label', 'pae', 'model_type')]

			for x, y, hue in x_y_hue:

				sns_violinplot(data=plot_df, x=x, y=y, hue=hue, split=True,
					title='%s for Stable vs. Unstable designs\nat Training Step %s'%(y, training_step), outpath=os.path.join(outfolder, 'plotting_dfs', 'figures',
					'%s_vs_%s_%s_%s'%(y,x,args.job_name,training_step)))

				for dataset in natsorted(plot_df['dataset'].unique().tolist()):

					sub_plot_df = plot_df.loc[plot_df['dataset']==dataset]

					sns_violinplot(data=sub_plot_df, x=x, y=y, hue=hue, split=True,
						title='%s for Stable vs. Unstable designs\nat Training Step %s\nin %s'%(y, training_step, dataset), 
						outpath=os.path.join(outfolder, 'plotting_dfs', 'figures',
						'%s_%s_vs_%s_%s_%s'%(dataset, y,x,args.job_name,training_step)))

	# Analyze the Precision and Recall of the AF2 confidence scores in predicting designs as stable\
	# and unstable.  Uses the default pLDDT and pAE thresholds or the provided thresholds.
	if args.precision_recall:
		
		for df_file in natsorted([f for f in os.listdir(os.path.join(outfolder, 'plotting_dfs')) if '.csv' in f]):

			training_step = df_file.split('.')[0].split('_')[-1]

			plot_df = pd.read_csv(os.path.join(outfolder, 'plotting_dfs', df_file))

			print('Number of examples in %s: %d'%(df_file, len(plot_df)))

			# Make an output folder for the generated Precision Recall dataframes.
			pr_df_outfolder = os.path.join(outfolder, 'precision_recall')
			os.makedirs(os.path.join(outfolder, 'precision_recall'), exist_ok=True)

			pr_fig_outfolder = os.path.join(outfolder, 'precision_recall', 'figures')
			os.makedirs(os.path.join(outfolder, 'precision_recall', 'figures'), exist_ok=True)

			pr_df = pd.DataFrame(columns=['thresh','training_step','dataset','model_type','metric_type','precision','recall','FPR'])

			pr_df_idx = 0

			for dataset in natsorted(plot_df['dataset'].unique().tolist()):

				sub_plot_df = plot_df.loc[plot_df['dataset']==dataset]

				# Iterate over both whole protein confidence metrics and reshaped region confidence metrics.
				if args.reshaped_helix_rmsd:
					conf_metrics = [('plddt', 'pae'), ('reshaped_plddt','reshaped_pae')]
				else:
					conf_metrics = [('plddt', 'pae')]


				for plddt_metric, pae_metric in conf_metrics:
					for thresh in list(range(0,101)):

						c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn, \
							f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn, \
							c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn, \
							f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn = count_confusion_matrix_values(args, thresh, sub_plot_df)

						# Calculate precision and recall for classic/fine-tuned plddt/pae
						c_plddt_precision = get_precision_value(c_plddt_tp, c_plddt_fp)
						c_plddt_recall = get_recall_value(c_plddt_tp, c_plddt_fn)
						c_plddt_FPR = get_false_positive_rate(c_plddt_fp, c_plddt_tn)

						f_plddt_precision = get_precision_value(f_plddt_tp, f_plddt_fp)
						f_plddt_recall = get_recall_value(f_plddt_tp, f_plddt_fn)
						f_plddt_FPR = get_false_positive_rate(f_plddt_fp, f_plddt_tn)

						c_pae_precision = get_precision_value(c_pae_tp, c_pae_fp)
						c_pae_recall = get_recall_value(c_pae_tp, c_pae_fn)
						c_pae_FPR = get_false_positive_rate(c_pae_fp, c_pae_tn)

						f_pae_precision = get_precision_value(f_pae_tp, f_pae_fp)
						f_pae_recall = get_recall_value(f_pae_tp, f_pae_fn)
						f_pae_FPR = get_false_positive_rate(f_pae_fp, f_pae_tn)

						# Columns are ordered ['thresh','training_step','dataset','model_type','metric_type','precision','recall','FPR']
						pr_df.loc[pr_df_idx] = [thresh, sub_plot_df.iloc[0]['training_step'], sub_plot_df.iloc[0]['dataset'], 'classic',
							plddt_metric, c_plddt_precision, c_plddt_recall, c_plddt_FPR]
						pr_df.loc[pr_df_idx+1] = [thresh, sub_plot_df.iloc[0]['training_step'], sub_plot_df.iloc[0]['dataset'], 'classic',
							pae_metric, c_pae_precision, c_pae_recall, c_pae_FPR]
						pr_df.loc[pr_df_idx+2] = [thresh, sub_plot_df.iloc[0]['training_step'], sub_plot_df.iloc[0]['dataset'], 'ft',
							plddt_metric, f_plddt_precision, f_plddt_recall, f_plddt_FPR]
						pr_df.loc[pr_df_idx+3] = [thresh, sub_plot_df.iloc[0]['training_step'], sub_plot_df.iloc[0]['dataset'], 'ft',
							pae_metric, f_pae_precision, f_pae_recall, f_pae_FPR]
						pr_df_idx += 4

					# Plot Precision-Recall curves for each dataset at each checkpointed model training step.
					sub_pr_df = pr_df.loc[pr_df['dataset'] == dataset]
					plot_precision_recall_curve(sub_pr_df, metric_type=plddt_metric,
						outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_%s_%s'%(plddt_metric, dataset, training_step)))
					plot_precision_recall_curve(sub_pr_df, metric_type=pae_metric,
						outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_%s_%s'%(pae_metric, dataset, training_step)))
					plot_tpr_fpr_curve(sub_pr_df, metric_type=plddt_metric,
						outpath=os.path.join(pr_fig_outfolder, 'tpr_fpr_%s_%s_%s'%(plddt_metric, dataset, training_step)))
					plot_tpr_fpr_curve(sub_pr_df, metric_type=pae_metric,
						outpath=os.path.join(pr_fig_outfolder, 'tpr_fpr_%s_%s_%s'%(pae_metric, dataset, training_step)))

			pr_df.to_csv(os.path.join(pr_df_outfolder, 'precision_recall_df_%s_%s.csv'%(args.job_name, training_step)))
			print('Saved dataframe: %s'%os.path.join(pr_df_outfolder, 'precision_recall_df_%s_%s.csv'%(args.job_name, training_step)))

			# Plot a Precision-Recall curve for all datasets at each checkpointed model training step.
			plot_precision_recall_curve(pr_df, metric_type='plddt',
				outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_alldata_%s'%('plddt', training_step)))
			plot_precision_recall_curve(pr_df, metric_type='pae',
				outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_alldata_%s'%('pae', training_step)))
			plot_precision_recall_curve(pr_df, metric_type='reshaped_plddt',
				outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_alldata_%s'%('reshaped_plddt', training_step)))
			plot_precision_recall_curve(pr_df, metric_type='reshaped_pae',
				outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_alldata_%s'%('reshaped_pae', training_step)))

	# Plot violin plots of RMSD to Ground Truth split by stability_label
	if args.plot_rmsd_distributions:

		for df_file in natsorted([f for f in os.listdir(os.path.join(outfolder, 'plotting_dfs')) if '.csv' in f]):

			training_step = df_file.split('.')[0].split('_')[-1]

			plot_df = pd.read_csv(os.path.join(outfolder, 'plotting_dfs', df_file))

			print('Number of examples in %s: %d'%(df_file, len(plot_df)))

			# Make an output folder for the generated figures.
			os.makedirs(os.path.join(outfolder, 'plotting_dfs', 'figures'), exist_ok=True)

			for x, y, hue in [('stability_label', 'rmsd_to_gt', 'model_type')]:

				sns_violinplot(data=plot_df, x=x, y=y, hue=hue, split=True,
					title='%s for Stable vs. Unstable designs\nat Training Step %s'%(y, training_step), outpath=os.path.join(outfolder, 'plotting_dfs', 'figures',
					'%s_vs_%s_%s_%s'%(y,x,args.job_name,training_step)))

				for dataset in natsorted(plot_df['dataset'].unique().tolist()):

					sub_plot_df = plot_df.loc[plot_df['dataset']==dataset]

					sns_violinplot(data=sub_plot_df, x=x, y=y, hue=hue, split=True,
						title='%s for Stable vs. Unstable designs\nat Training Step %s\nin %s'%(y, training_step, dataset), 
						outpath=os.path.join(outfolder, 'plotting_dfs', 'figures',
						'%s_%s_vs_%s_%s_%s'%(dataset, y, x, args.job_name,training_step)))


















		






=======
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description='Analyze RMSDs between classic and fine-tuned AF2 predictions to their ground truth structures, '
                    'as well as the classic and fine-tuned AF2 confidence metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--job_name', type=str, required=True,
        help='Name for the job which will be used in the output .png files.'
    )
    parser.add_argument(
        '--checkpoint_model_data_folder', type=str, required=True,
        help='Folder containing dataframe(s) output by calc_checkpointed_models_metrics.py.'
    )
    parser.add_argument(
        '--outfolder_parent', type=str, required=True,
        help='Output files are written to outfolder_parent.'
    )

    parser.add_argument(
        '--rmsd_analysis', action='store_true',
        help="Analyze the RMSDs to the ground truth structures for each checkpointed model's predictions."
    )
    parser.add_argument(
        '--stability_data', nargs='*', type=str, required=False, default=None,
        help='Dataframe(s) containing a "stability_label" column with "stable," "unstable," and "uncertain" values, '
             'as well as a "sequence" column, which is used to get the sequence length.'
    )
    parser.add_argument(
        '--rmsd_thresh', type=float, required=False, default=1.5,
        help='Threshold for AF2 to ground truth RMSD values for plotting the fraction of designs predicted to '
             'sub-<rmsd_thresh> RMSD to the ground truth structures.'
    )
    parser.add_argument(
        '--plot_rmsd_distributions', action='store_true',
        help='Plot distributions of RMSD values for stable vs. unstable designs.'
    )
    parser.add_argument(
        '--reshaped_helix_rmsd', action='store_true',
        help='Plot Reshaped Helix RMSD to Rosetta values for LUCS designs.'
    )

    parser.add_argument(
        '--stability_classifier', action='store_true',
        help='Analyze the Precision and Recall of the AF2 confidence scores in predicting designs as stable '
             'and unstable. Uses the default pLDDT and pAE thresholds or the provided thresholds.'
    )
    parser.add_argument(
        '--plddt_thresh', type=float, required=False, default=0.80367635,
        help='Threshold for AF2 pLDDT values in classifying a design as stable (if the AF2 average pLDDT for a design '
             'is greater than this threshold, then the design is classified as "stable"). Default is 0.80367635 (taken from the '
             'alphafold_finetune github repo: https://github.com/phbradley/alphafold_finetune).'
    )
    parser.add_argument(
        '--pae_thresh', type=float, required=False, default=0.43373787,
        help='Threshold for AF2 pAE values in classifying a design as stable (if the AF2 average pAE for a design '
             'is greater than this threshold, then the design is classified as "stable"). Default is 0.43373787 (taken from the '
             'alphafold_finetune github repo: https://github.com/phbradley/alphafold_finetune).'
    )
    parser.add_argument(
        '--plot_confidence_distributions', action='store_true',
        help='Plot the distributions of the confidence scores for the classic and fine-tuned models '
             'at each training step. Requires a outfolder_parent/job_name/plotting_dfs folder to exist with outputs from the '
             '--stability_classifier outputs.'
    )
    parser.add_argument(
        '--precision_recall', action='store_true',
        help='Generate Precision vs. Recall curves for classifying stable designs at different pLDDT and '
             'pAE thresholds.'
    )
    parser.add_argument(
        '--reshaped_residue_stability', action='store_true',
        help='Analyze the Precision and Recall of the AF2 confidence scores in predicting designs as stable '
             'and unstable using the pAE values between the bb_remodeled_residues and non-bb_remodeled_residues and '
             'pLDDT values of the bb_remodeled_residues (this residue set is found in the bb_remodeled_residues column '
             'of the stability_data dataframe(s)). Uses the default pLDDT and pAE thresholds or the provided thresholds.'
    )

    parser.add_argument(
        '--verbose', action='store_true',
        help='Print verbose output to the command line.'
    )

    args = parser.parse_args()

    outfolder = os.path.join(args.outfolder_parent)
    os.makedirs(outfolder, exist_ok=True)

    # Combine the checkpoint_model_dataframes into one df
    df_paths = [
        os.path.join(args.checkpoint_model_data_folder, f) for
        f in os.listdir(args.checkpoint_model_data_folder)
        if '.csv' in f and 'prediction_analysis' not in f
    ]
    dfs = [pd.read_csv(p) for p in df_paths]
    df = pd.concat(dfs)

    # If stability data is provided, add this data to the df.
    if args.stability_data is not None:

        # Load the stability_data df(s)
        stab_dfs = [pd.read_csv(sdf) for sdf in args.stability_data]
        stab_df = pd.concat(stab_dfs)

        for idx, row in df.iterrows():

            # Half of the Basanta NTF2s' ground truth files include the string '_BasicBeNTF2_designed_0000' after the design name
            name = row['ground_truth_path'].rsplit('/', 1)[1].replace('.pdb', '').replace('_BasicBeNTF2_designed_0000', '')

            # Names to check for in the 'name' column of the stab_df
            possible_names = [name, name + '.pdb', row['ground_truth_path'].rsplit('/', 1)[1].replace('.pdb', '')]
            try:
                possible_names.append(int(name))
            except:
                pass

            # The rocklin_miniproteins_2017 stability df has .pdb at the end of the design names
            # In case the stab_df 'name' column contains int values, check for int(name)
            df.loc[idx, 'stability_label'] = stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label']

    # Analyze the RMSDs to the ground truth structures for each checkpointed model's predictions
    if args.rmsd_analysis:

        # For each unique dataset, plot the fractions of designs with lower RMSDs to the ground
        # truth structure than the other model (classic or fine-tuned) for each training step
        # for both models.
        for idx, row in df.iterrows():
            df.loc[idx, 'ground_truth_parent'] = row['ground_truth_path'].rsplit('/', 1)[0]

        rmsd_col_endings = ['_gt_rmsd']
        rmsd_col_names = ['\n(Whole Structure RMSD)']
        if args.reshaped_helix_rmsd:
            # Calculate RMSDs for both whole structures and reshaped helices.
            rmsd_col_endings.append('_helix_rmsd')
            rmsd_col_names.append('\n(Reshaped Helix RMSD)')

        for rmsd_col_ending, rmsd_col_name in zip(rmsd_col_endings, rmsd_col_names):

            ## Only plot RMSDs over training steps if there are dataframes for multiple training steps.
            if len(df_paths) != 1:
                # Plot the fractions of designs with lower RMSDs to the ground truth structure than the
                # other model (classic or fine-tuned) for each training step for both models.
                make_all_rmsd_frac_over_training_step_figures(
                    args, df, title_append=rmsd_col_name,
                    filename_append=rmsd_col_ending, outfolder=outfolder,
                    rmsd_col_ending=rmsd_col_ending
                )

            plot_rmsds_for_each_dataset(
                df, args, rmsd_col_ending=rmsd_col_ending, title_append=rmsd_col_name,
                filename_append=rmsd_col_ending, outfolder=outfolder
            )

            # This violinplot uses stability labels from args.stability_data
            if args.stability_data is not None:
                violinplot_rmsds_for_each_dataset(
                    df, args, rmsd_col_ending=rmsd_col_ending, title_append=rmsd_col_name,
                    filename_append=rmsd_col_ending, outfolder=outfolder
                )

            for ground_truth_parent in natsorted(df['ground_truth_parent'].unique().tolist()):

                # Slice the df to include only examples from the current ground_truth_parent
                sub_df = df.loc[df['ground_truth_parent'] == ground_truth_parent]

                # Second to last subfolder in the ground_truth_parent is the name of the de novo protein dataset
                # For example, /wynton/group/kortemme/borr/alphafold_finetune/data/abba_rocklin_rd6/pdbs
                dataset_name = ground_truth_parent.split('/')[-2]

                if len(df_paths) != 1:
                    ## Only plot RMSDs over training steps if there are dataframes for multiple training steps.
                    make_all_rmsd_frac_over_training_step_figures(
                        args, sub_df,
                        title_append=' for %s%s' % (dataset_name, rmsd_col_name),
                        filename_append='_' + dataset_name + '_' + rmsd_col_ending,
                        outfolder=outfolder
                    )

    # Analyze the Precision and Recall of the AF2 confidence scores in predicting designs as stable
    # and unstable. Uses the default pLDDT and pAE thresholds or the provided thresholds.
    if args.stability_classifier:

        # Ensure the a path to a stability df has been provided
        # There's a better way to do this, with argparse dependent arguments.
        if args.stability_data is None:
            print('If using the --stability_classifier flag, you must use the --stability_data flag to provide '
                  'a path to a dataframe (.csv file) with "stability_label" columns')
        else:
            # Iterate over each checkpointed model's RMSD+confidence score dataframe.
            for df_path in natsorted(df_paths):

                # New df formatted for plotting with seaborn
                if args.reshaped_helix_rmsd:
                    new_df = pd.DataFrame(columns=[
                        'name', 'dataset', 'training_step', 'plddt', 'pae', 'stability_label',
                        'model_type', 'rmsd_to_gt', 'helix_rmsd_to_gt', 'reshaped_plddt', 'reshaped_pae'
                    ])
                else:
                    new_df = pd.DataFrame(columns=[
                        'name', 'dataset', 'training_step', 'plddt', 'pae', 'stability_label',
                        'model_type', 'rmsd_to_gt'
                    ])

                t_step_df = pd.read_csv(df_path)

                training_step = t_step_df.iloc[0]['training_step']

                # Load the stability_data df(s)
                stab_dfs = [pd.read_csv(sdf) for sdf in args.stability_data]
                stab_df = pd.concat(stab_dfs)

                # Add columns to the new_df (which is used for plotting confidence score distributions for
                # stable and unstable designs) with data from the checkpointed model's df.
                # Calculate Precision and Recall for predicting stable designs, based on the classic and fine-tuned AF2
                # models' confidence metrics and the args.plddt_thresh and args.pae_thresh (average confidence values
                # greater than these thresholds for a sequence are considered to be a prediction of that sequence
                # folding into a stable protein).
                # Split Precision and Recall analysis by ground truth folder (dataset).
                # Temporarily add ground_truth_parent, name, and stability_label columns to the t_step_df
                new_idx = 0
                for idx, row in t_step_df.iterrows():

                    t_step_df.loc[idx, 'ground_truth_parent'] = row['ground_truth_path'].rsplit('/', 1)[0]
                    name = row['ground_truth_path'].rsplit('/', 1)[1].replace('.pdb', '')

                    # The Basanta NTF2s' ground truth files include the string '_BasicBeNTF2_designed_0000' after the design name
                    name = name.replace('_BasicBeNTF2_designed_0000', '')

                    # Names to check for in the 'name' column of the stab_df
                    possible_names = [name, name + '.pdb', row['ground_truth_path'].rsplit('/', 1)[1].replace('.pdb', '')]
                    try:
                        possible_names.append(int(name))
                    except:
                        pass

                    t_step_df.loc[idx, 'name'] = name

                    # The rocklin_miniproteins_2017 stability df has .pdb at the end of the design names
                    # In case the stab_df 'name' column contains int values, check for int(name)
                    t_step_df.loc[idx, 'stability_label'] = stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label']

                    # Add columns to the new dataframe (in a format friendly for seaborn plotting)
                    # Columns of new_df are columns=['name','dataset','training_step','plddt','pae','stability_label','model_type','rmsd_to_gt']
                    ## If including reshaped residue metrics, then the df has the additional columns: 'helix_rmsd_to_gt','reshaped_plddt','reshaped_pae']
                    if args.reshaped_helix_rmsd:
                        new_df.loc[new_idx] = [
                            name, '', training_step, 0, 0,
                            stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label'],
                            'classic', 0, 0, 0, 0
                        ]
                        new_df.loc[new_idx + 1] = [
                            name, '', training_step, 0, 0,
                            stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label'],
                            'ft', 0, 0, 0, 0
                        ]
                        new_idx += 2
                    else:
                        new_df.loc[new_idx] = [
                            name, '', training_step, 0, 0,
                            stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label'],
                            'classic', 0
                        ]
                        new_df.loc[new_idx + 1] = [
                            name, '', training_step, 0, 0,
                            stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]['stability_label'],
                            'ft', 0
                        ]
                        new_idx += 2

                for ground_truth_parent in natsorted(t_step_df['ground_truth_parent'].unique().tolist()):

                    # Slice the df to include only examples from the current ground_truth_parent
                    sub_df = t_step_df.loc[t_step_df['ground_truth_parent'] == ground_truth_parent]

                    # Second to last subfolder in the ground_truth_parent is the name of the de novo protein dataset
                    # For example, /wynton/group/kortemme/borr/alphafold_finetune/data/abba_rocklin_rd6/pdbs
                    dataset_name = ground_truth_parent.split('/')[-2]

                    # Iterate over the rows in the dataset-specific checkpointed model df
                    for idx, row in sub_df.iterrows():

                        # Names to check for in the 'name' column of the stab_df
                        possible_names = [row['name'], row['name'] + '.pdb', row['ground_truth_path'].rsplit('/', 1)[1].replace('.pdb', '')]
                        try:
                            possible_names.append(int(row['name']))
                        except:
                            pass

                        # To select the corresponding row in the stability df
                        stab_row = stab_df.loc[stab_df['name'].isin(possible_names)].iloc[0]

                        # Add the dataset name to the new_df (this df is used for plotting with seaborn)
                        new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'classic'), 'dataset'] = dataset_name
                        new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'ft'), 'dataset'] = dataset_name

                        # Add the predicted structure to ground truth RMSD values
                        new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'classic'), 'rmsd_to_gt'] = row['classic_gt_rmsd']
                        new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'ft'), 'rmsd_to_gt'] = row['ft_gt_rmsd']

                        if args.reshaped_helix_rmsd:
                            # Add the predicted structure to ground truth helix RMSD values
                            new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'classic'), 'helix_rmsd_to_gt'] = row['classic_helix_rmsd']
                            new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'ft'), 'helix_rmsd_to_gt'] = row['ft_helix_rmsd']

                        c_pae, f_pae, c_plddt, f_plddt = get_confidence_values(args, new_df, row, stab_row, possible_names)

                        # Add the average pLDDT values to the new_df (used for plotting with seaborn)
                        new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'classic'), 'plddt'] = c_plddt
                        new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'ft'), 'plddt'] = f_plddt

                        # Add the average pAE values to the new_df (used for plotting with seaborn)
                        new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'classic'), 'pae'] = c_pae
                        new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'ft'), 'pae'] = f_pae

                        if args.reshaped_residue_stability:
                            # Add the average reshaped residue pLDDT values to the new_df (used for plotting with seaborn)
                            new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'classic'), 'reshaped_plddt'] = row['classic_reshaped_plddt']
                            new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'ft'), 'reshaped_plddt'] = row['ft_reshaped_plddt']

                            # Add the average reshaped residue pAE values to the new_df (used for plotting with seaborn)
                            new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'classic'), 'reshaped_pae'] = row['classic_reshaped_pae']
                            new_df.loc[(new_df['name'].isin(possible_names)) & (new_df['model_type'] == 'ft'), 'reshaped_pae'] = row['ft_reshaped_pae']

                # Saving thec new_df (which is in a format that makes plotting with
                # seaborn convenient).
                os.makedirs(os.path.join(outfolder, 'plotting_dfs'), exist_ok=True)
                new_df.to_csv(os.path.join(outfolder, 'plotting_dfs', 'plotting_df_%s_trainstep_%d.csv' % (args.job_name, training_step)))
                print('Saved dataframe %s' % os.path.join(outfolder, 'plotting_dfs', 'plotting_df_%s_trainstep_%d.csv' % (args.job_name, training_step)))

    # Plot the distributions of confidence values for classic and fine-tuned AF2, for each training step, and
    # for each dataset (as well as all the datasets at once).
    if args.plot_confidence_distributions:

        for df_file in natsorted([f for f in os.listdir(os.path.join(outfolder, 'plotting_dfs')) if '.csv' in f]):

            training_step = df_file.split('.')[0].split('_')[-1]

            plot_df = pd.read_csv(os.path.join(outfolder, 'plotting_dfs', df_file))

            print('Number of examples in %s: %d' % (df_file, len(plot_df)))

            # Make an output folder for the generated figures.
            os.makedirs(os.path.join(outfolder, 'plotting_dfs', 'figures'), exist_ok=True)

            if args.reshaped_helix_rmsd:
                x_y_hue = [
                    ('stability_label', 'plddt', 'model_type'),
                    ('stability_label', 'pae', 'model_type'),
                    ('stability_label', 'reshaped_plddt', 'model_type'),
                    ('stability_label', 'reshaped_pae', 'model_type')
                ]
            else:
                x_y_hue = [
                    ('stability_label', 'plddt', 'model_type'),
                    ('stability_label', 'pae', 'model_type')
                ]

            for x, y, hue in x_y_hue:

                sns_violinplot(
                    data=plot_df, x=x, y=y, hue=hue, split=True,
                    title='%s for Stable vs. Unstable designs\nat Training Step %s' % (y, training_step),
                    outpath=os.path.join(
                        outfolder, 'plotting_dfs', 'figures',
                        '%s_vs_%s_%s_%s' % (y, x, args.job_name, training_step)
                    )
                )

                for dataset in natsorted(plot_df['dataset'].unique().tolist()):

                    sub_plot_df = plot_df.loc[plot_df['dataset'] == dataset]

                    sns_violinplot(
                        data=sub_plot_df, x=x, y=y, hue=hue, split=True,
                        title='%s for Stable vs. Unstable designs\nat Training Step %s\nin %s' % (y, training_step, dataset),
                        outpath=os.path.join(
                            outfolder, 'plotting_dfs', 'figures',
                            '%s_%s_vs_%s_%s_%s' % (dataset, y, x, args.job_name, training_step)
                        )
                    )

    # Analyze the Precision and Recall of the AF2 confidence scores in predicting designs as stable
    # and unstable. Uses the default pLDDT and pAE thresholds or the provided thresholds.
    if args.precision_recall:

        for df_file in natsorted([f for f in os.listdir(os.path.join(outfolder, 'plotting_dfs')) if '.csv' in f]):

            training_step = df_file.split('.')[0].split('_')[-1]

            plot_df = pd.read_csv(os.path.join(outfolder, 'plotting_dfs', df_file))

            print('Number of examples in %s: %d' % (df_file, len(plot_df)))

            # Make an output folder for the generated Precision Recall dataframes.
            pr_df_outfolder = os.path.join(outfolder, 'precision_recall')
            os.makedirs(os.path.join(outfolder, 'precision_recall'), exist_ok=True)

            pr_fig_outfolder = os.path.join(outfolder, 'precision_recall', 'figures')
            os.makedirs(os.path.join(outfolder, 'precision_recall', 'figures'), exist_ok=True)

            pr_df = pd.DataFrame(columns=[
                'thresh', 'training_step', 'dataset', 'model_type', 'metric_type',
                'precision', 'recall', 'FPR'
            ])

            pr_df_idx = 0

            for dataset in natsorted(plot_df['dataset'].unique().tolist()):

                sub_plot_df = plot_df.loc[plot_df['dataset'] == dataset]

                # Iterate over both whole protein confidence metrics and reshaped region confidence metrics.
                if args.reshaped_helix_rmsd:
                    conf_metrics = [('plddt', 'pae'), ('reshaped_plddt', 'reshaped_pae')]
                else:
                    conf_metrics = [('plddt', 'pae')]

                for plddt_metric, pae_metric in conf_metrics:
                    for thresh in list(range(0, 101)):

                        c_plddt_tp, c_plddt_fp, c_plddt_tn, c_plddt_fn, \
                            f_plddt_tp, f_plddt_fp, f_plddt_tn, f_plddt_fn, \
                            c_pae_tp, c_pae_fp, c_pae_tn, c_pae_fn, \
                            f_pae_tp, f_pae_fp, f_pae_tn, f_pae_fn = count_confusion_matrix_values(args, thresh, sub_plot_df)

                        # Calculate precision and recall for classic/fine-tuned plddt/pae
                        c_plddt_precision = get_precision_value(c_plddt_tp, c_plddt_fp)
                        c_plddt_recall = get_recall_value(c_plddt_tp, c_plddt_fn)
                        c_plddt_FPR = get_false_positive_rate(c_plddt_fp, c_plddt_tn)

                        f_plddt_precision = get_precision_value(f_plddt_tp, f_plddt_fp)
                        f_plddt_recall = get_recall_value(f_plddt_tp, f_plddt_fn)
                        f_plddt_FPR = get_false_positive_rate(f_plddt_fp, f_plddt_tn)

                        c_pae_precision = get_precision_value(c_pae_tp, c_pae_fp)
                        c_pae_recall = get_recall_value(c_pae_tp, c_pae_fn)
                        c_pae_FPR = get_false_positive_rate(c_pae_fp, c_pae_tn)

                        f_pae_precision = get_precision_value(f_pae_tp, f_pae_fp)
                        f_pae_recall = get_recall_value(f_pae_tp, f_pae_fn)
                        f_pae_FPR = get_false_positive_rate(f_pae_fp, f_pae_tn)

                        # Columns are ordered ['thresh','training_step','dataset','model_type','metric_type','precision','recall','FPR']
                        pr_df.loc[pr_df_idx] = [
                            thresh, sub_plot_df.iloc[0]['training_step'], sub_plot_df.iloc[0]['dataset'], 'classic',
                            plddt_metric, c_plddt_precision, c_plddt_recall, c_plddt_FPR
                        ]
                        pr_df.loc[pr_df_idx + 1] = [
                            thresh, sub_plot_df.iloc[0]['training_step'], sub_plot_df.iloc[0]['dataset'], 'classic',
                            pae_metric, c_pae_precision, c_pae_recall, c_pae_FPR
                        ]
                        pr_df.loc[pr_df_idx + 2] = [
                            thresh, sub_plot_df.iloc[0]['training_step'], sub_plot_df.iloc[0]['dataset'], 'ft',
                            plddt_metric, f_plddt_precision, f_plddt_recall, f_plddt_FPR
                        ]
                        pr_df.loc[pr_df_idx + 3] = [
                            thresh, sub_plot_df.iloc[0]['training_step'], sub_plot_df.iloc[0]['dataset'], 'ft',
                            pae_metric, f_pae_precision, f_pae_recall, f_pae_FPR
                        ]
                        pr_df_idx += 4

                    # Plot Precision-Recall curves for each dataset at each checkpointed model training step.
                    sub_pr_df = pr_df.loc[pr_df['dataset'] == dataset]
                    plot_precision_recall_curve(
                        sub_pr_df, metric_type=plddt_metric,
                        outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_%s_%s' % (plddt_metric, dataset, training_step))
                    )
                    plot_precision_recall_curve(
                        sub_pr_df, metric_type=pae_metric,
                        outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_%s_%s' % (pae_metric, dataset, training_step))
                    )
                    plot_tpr_fpr_curve(
                        sub_pr_df, metric_type=plddt_metric,
                        outpath=os.path.join(pr_fig_outfolder, 'tpr_fpr_%s_%s_%s' % (plddt_metric, dataset, training_step))
                    )
                    plot_tpr_fpr_curve(
                        sub_pr_df, metric_type=pae_metric,
                        outpath=os.path.join(pr_fig_outfolder, 'tpr_fpr_%s_%s_%s' % (pae_metric, dataset, training_step))
                    )

            pr_df.to_csv(os.path.join(pr_df_outfolder, 'precision_recall_df_%s_%s.csv' % (args.job_name, training_step)))
            print('Saved dataframe: %s' % os.path.join(pr_df_outfolder, 'precision_recall_df_%s_%s.csv' % (args.job_name, training_step)))

            # Plot a Precision-Recall curve for all datasets at each checkpointed model training step.
            plot_precision_recall_curve(
                pr_df, metric_type='plddt',
                outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_alldata_%s' % ('plddt', training_step))
            )
            plot_precision_recall_curve(
                pr_df, metric_type='pae',
                outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_alldata_%s' % ('pae', training_step))
            )
            plot_precision_recall_curve(
                pr_df, metric_type='reshaped_plddt',
                outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_alldata_%s' % ('reshaped_plddt', training_step))
            )
            plot_precision_recall_curve(
                pr_df, metric_type='reshaped_pae',
                outpath=os.path.join(pr_fig_outfolder, 'prec_rec_%s_alldata_%s' % ('reshaped_pae', training_step))
            )

    # Plot violin plots of RMSD to Ground Truth split by stability_label
    if args.plot_rmsd_distributions:

        for df_file in natsorted([f for f in os.listdir(os.path.join(outfolder, 'plotting_dfs')) if '.csv' in f]):

            training_step = df_file.split('.')[0].split('_')[-1]

            plot_df = pd.read_csv(os.path.join(outfolder, 'plotting_dfs', df_file))

            print('Number of examples in %s: %d' % (df_file, len(plot_df)))

            # Make an output folder for the generated figures.
            os.makedirs(os.path.join(outfolder, 'plotting_dfs', 'figures'), exist_ok=True)

            for x, y, hue in [('stability_label', 'rmsd_to_gt', 'model_type')]:

                sns_violinplot(
                    data=plot_df, x=x, y=y, hue=hue, split=True,
                    title='%s for Stable vs. Unstable designs\nat Training Step %s' % (y, training_step),
                    outpath=os.path.join(
                        outfolder, 'plotting_dfs', 'figures',
                        '%s_vs_%s_%s_%s' % (y, x, args.job_name, training_step)
                    )
                )

                for dataset in natsorted(plot_df['dataset'].unique().tolist()):

                    sub_plot_df = plot_df.loc[plot_df['dataset'] == dataset]

                    sns_violinplot(
                        data=sub_plot_df, x=x, y=y, hue=hue, split=True,
                        title='%s for Stable vs. Unstable designs\nat Training Step %s\nin %s' % (y, training_step, dataset),
                        outpath=os.path.join(
                            outfolder, 'plotting_dfs', 'figures',
                            '%s_%s_vs_%s_%s_%s' % (dataset, y, x, args.job_name, training_step)
                        )
                    )
>>>>>>> be02a1e (lucs_af refactor and cleanup)
