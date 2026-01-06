# Instructions for running fine-tuned AF2 models

Benjamin Orr

For "An improved model for prediction of de novo designed proteins with diverse geometries"

Authors: Benjamin Orr\*, Stephanie E. Crilly\*, Deniz Akpinaroglu, Eleanor Zhu, Michael J. Keiser, Tanja Kortemme

---

## Overview

The simplest way to run inference with the fine-tuned AF2 models is by using Google Colab ([FineTunedAlphaFold2.ipynb](https://colab.research.google.com/drive/1R8S4jZLcpDOcwBALfXUR-bI37ELkrFkH?usp=sharing)). Alternatively, one can use a local installation of [localcolabfold](https://github.com/YoshitakaMo/localcolabfold).

---

## Running fine-tuned AF2 using Google Colab (recommended)

[FineTunedAlphaFold2.ipynb](https://colab.research.google.com/drive/1R8S4jZLcpDOcwBALfXUR-bI37ELkrFkH?usp=sharing) is a modified version of [AlphaFold2.ipynb](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb#scrollTo=11l8k--10q0C) (Mirdita et al. 2022). This modified version downloads the fine-tuned AF2 parameters and runs AF2 inference using these parameters.

Instructions for running inference with the fine-tuned AF2 models using Google Colab can be found in the python notebook [FineTunedAlphaFold2.ipynb](https://colab.research.google.com/drive/1R8S4jZLcpDOcwBALfXUR-bI37ELkrFkH?usp=sharing).

## Running fine-tuned AF2 using localcolabfold

Alternatively, inference can be run with the fine-tuned AF2 models using a local installation of [localcolabfold](https://github.com/YoshitakaMo/localcolabfold).

<<<<<<< HEAD
First, download the model parameters from [zenodo](https://zenodo.org/records/15580792) or using the command line with the following commands:
=======
First, download the model parameters from [zenodo](https://zenodo.org/records/15580792) (or from the publication data [repo](https://zenodo.org/records/16938925)) or using the command line with the following commands:
>>>>>>> be02a1e (lucs_af refactor and cleanup)

```
wget https://zenodo.org/records/15580792/files/StableStructSplit_params_model_2_ptm.npz
wget https://zenodo.org/records/15580792/files/Stable_params_model_2_ptm.npz
wget https://zenodo.org/records/15580792/files/StableUnstable_params_model_2_ptm.npz
```

The fine-tuned AF2 models are fine-tuned versions of AF2 model_2_ptm. To run inference with a fine-tuned AF2 model using localcolabfold, rename the fine-tuned parameters to model_2_ptm.npz (from a name like StableStructSplit_params_model_2_ptm.npz; this is because AF2 expects parameter files with names like model_2_ptm.npz), move this renamed file to the --data_dir (localcolabfold installs AF2 parameters into a subdirectory named "params"), and pass this --data_dir argument to localcolabfold. To run the fine-tuned model alone (and not models 1, 3, 4, or 5), provide the arguments --num_models 1 --model_order [2]. These steps are described in more detail below:

1. Install [localcolabfold](https://github.com/YoshitakaMo/localcolabfold)

2. Rename the fine-tuned AF2 parameter file to model_2_ptm.npz and copy this file to subdirectory containing AF2 params (AF2 expects this subdirectory to be named "params"; localcolabfold installs AF2 parameters into a subdirectory named "params").

```
# copy the fine-tuned AF2 parameters to params_model_2_ptm.npz
cp StableStructSplit_params_model_2_ptm.npz params_model_2_ptm.npz

# move the renamed fine-tuned AF2 parameters to the params/ directory
mv params_model_2_ptm.npz DATA_DIR/params
```

3. Run localcolabfold with the following arguments (note that --data_dir takes the path to DATA_DIR, and not DATA_DIR/params):

```
# "--model_type alphafold2_ptm" ensures that params_model_2_ptm.npz is used
# "--msa_mode single_sequence" skips the MSA and template generation steps (this is commonly used for predicting de novo protein sequences)
colabfold_batch --data_dir DATA_DIR --num_models 1 --model_order [2] --model_type alphafold2_ptm --msa_mode single_sequence input outputdir/
```

---

## Typical run times on Google Colab

Installation:
 - Installing the necessary packages for running inference with the fine-tuned AF2 models takes only ~5 minutes on Google Colab using "Runtime type" = Python 3, "Hardware accelerator" = T4 GPU, and "High-RAM" = ON.

Inference:
 - When predicting a ~100-residue protein sequence WITHOUT multiple sequence alignment or template inputs (as is common for predicting de novo-designed proteins; also called "single_sequence" mode), inference takes ~1 minute.
 - When predicting a ~100-residue protein sequence WITH multiple sequence alignment and template inputs (as is highly recommended for predicting natural proteins, using msa_mode="mmseqs2_uniref_env" and template_mode="pdb100" in Google Colab), inference takes ~5 minutes.

---

## Expected outputs

Expected outputs for a de novo-designed protein (PDB ID: 8tnm) and a natural protein (PDB ID: 2e12, CATH 4.2 Domain ID: A00) can be found under examples/expected_outputs/StableStructSplit_model2_8tnm and StableStructSplit_model2_2e12A00, respectively.

Prediction settings for StableStructSplit_model2_8tnm:
- query_sequence = 'SNDYVNQMISQMTDLAKSLNVDVTELITSVTQALEALLEEYRREGRLTDQVEKMASSVALQLAAELLAQKALEEGHDKKQTTAKRNQISNSYSSEAMSHARAWAASRHSEEEAEKLAEELYKDMKESLKQRIDTEQ'
- fine_tuned_model = "Stable Structure-Split"
- num_relax = 0
- template_mode = "none"
- msa_mode = "single_sequence"
- pair_mode = "unpaired_paired"
- model_type = "alphafold2_ptm"
- num_recycles = "3"
- recycle_early_stop_tolerance = "auto"
- relax_max_iterations = 200
- pairing_strategy = "greedy"
- calc_extra_ptm = False
- max_msa = "auto"
- num_seeds = 1
- random_seed = 0
- use_dropout = False

Prediction settings for StableStructSplit_model2_2e12A00:
- query_sequence = 'MPKYAPHVYTEQAQIATLEHWVKLLDGQERVRIELDDGSMIAGTVAVRPTIQTYRDEQEREGSNGQLRIDHLDASQEPQWIWMDRIVAVHPMP'
- fine_tuned_model = "Stable Structure-Split"
- num_relax = 0
- template_mode = "pdb100"
- msa_mode = "mmseqs2_uniref_env"
- pair_mode = "unpaired_paired"
- model_type = "alphafold2_ptm"
- num_recycles = "3"
- recycle_early_stop_tolerance = "auto"
- relax_max_iterations = 200
- pairing_strategy = "greedy"
- calc_extra_ptm = False
- max_msa = "auto"
- num_seeds = 1
- random_seed = 0
- use_dropout = False

---

## Software versions

The fine-tuned AF2 models were tested on the following software versions. These packages need not be installed individually for running inference with the fine-tuned AF2 models, as they are installed by running the code cells in [FineTunedAlphaFold2.ipynb](https://colab.research.google.com/drive/1R8S4jZLcpDOcwBALfXUR-bI37ELkrFkH?usp=sharing) or by installing [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) (for ColabFold v1.1.5).

Google Colab (these versions are all specified in [FineTunedAlphaFold2.ipynb](https://colab.research.google.com/drive/1R8S4jZLcpDOcwBALfXUR-bI37ELkrFkH?usp=sharing)):
 - Colabfold v1.5.5: colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold
 - python=3.11 
 - dm-haiku==0.0.10 
 - jax[cuda12_pip]==0.3.25
 - tensorflow==2.11
 - silence_tensorflow==1.2.3
 - kalign2=2.04 
 - hhsuite=3.3.0 
 - openmm=7.7.0 
 - pdbfixer=1.11
 - numpy==2.2.5
 - mmseqs2 (Release 17-b804f)

localcolabfold (these versions are all specified in [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) for ColabFold v1.1.5):
 - ColabFold v1.5.5: colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold
 - python=3.10 
 - dm-haiku==0.0.10 
 - jax[cuda12]==0.5.3
 - tensorflow==2.11
 - silence_tensorflow==1.2.3
 - kalign2=2.04
 - hhsuite=3.3.0
 - openmm==8.2.0 
 - pdbfixer=1.11
 - numpy==2.2.5
 - mmseqs2 (Release 17-b804f)

The fine-tuned AF2 models were developed on the UCSF HPC (Wynton), with the following operating system:
 - NAME="Rocky Linux"
 - VERSION="8.10 (Green Obsidian)"
 - ID="rocky"
 - ID_LIKE="rhel centos fedora"
 - VERSION_ID="8.10"
 - PLATFORM_ID="platform:el8"
 - PRETTY_NAME="Rocky Linux 8.10 (Green Obsidian)"
 - ANSI_COLOR="0;32"
 - LOGO="fedora-logo-icon"
 - CPE_NAME="cpe:/o:rocky:rocky:8:GA"
 - HOME_URL="https://rockylinux.org/"
 - BUG_REPORT_URL="https://bugs.rockylinux.org/"
 - SUPPORT_END="2029-05-31"
 - ROCKY_SUPPORT_PRODUCT="Rocky-Linux-8"
 - ROCKY_SUPPORT_PRODUCT_VERSION="8.10"
 - REDHAT_SUPPORT_PRODUCT="Rocky Linux"
 - REDHAT_SUPPORT_PRODUCT_VERSION="8.10"
