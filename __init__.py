"""
lucs_af

LUCS (Loop-Helix-Loop Unit Combinatorial Sampling) structural analysis 
and AlphaFold2 fine-tuning and inference.

LUCS structural analysis:
- Calculating protein structural diversity and similarity (RMSD, TM-score, helix RMSD, helix displacement and angle)
- Analyzing and visualizing helix geometry/diversity, e.g., the extent to which LUCS helices were reshaped from their starting positions
- Analyzing predicted structures' confidence metrics (pLDDT, pAE, pTM), structure-based metrics, and Rosetta metrics

AlphaFold fine-tuning:
- Fine-tuning AlphaFold2 models on custom datasets
- Fine-tuning classifier heads on top of AlphaFold2 for protein property prediction
- Running classic and fine-tuned AlphaFold2 predictions with and without MSA and templates
- Evaluating predictions for structural and classification accuracy

Author: Ben Orr
"""

__version__ = "2.0.0"
__author__ = "Ben Orr"
