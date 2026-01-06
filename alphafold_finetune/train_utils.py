"""
train_utils.py

Ben Orr
3.28.24
Adapted from https://github.com/phbradley/alphafold_finetune

Utility functions for AlphaFold fine-tuning and training.

Contains protein structure manipulation functions, coordinate transformations,
and JAX implementations of AlphaFold confidence metrics (pLDDT, PAE).
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from alphafold.common import residue_constants


# Feature keys used in PDB structure processing
pdb_key_list = [
    'atom14_atom_exists',
    'residx_atom14_to_atom37',
    'residx_atom37_to_atom14',
    'atom37_atom_exists',
    'pseudo_beta',
    'pseudo_beta_mask',
    'all_atom_mask',
    'chi_mask',
    'chi_angles',
    'all_atom_positions',
    'atom14_gt_exists',
    'atom14_gt_positions',
    'atom14_alt_gt_positions',
    'atom14_alt_gt_exists',
    'atom14_atom_is_ambiguous',
    'rigidgroups_gt_frames',
    'rigidgroups_gt_exists',
    'rigidgroups_group_exists',
    'rigidgroups_group_is_ambiguous',
    'rigidgroups_alt_gt_frames',
    'backbone_translation',
    'backbone_rotation',
    'backbone_affine_mask'
]

pdb_key_list_int = [
    'residx_atom14_to_atom37',
    'residx_atom37_to_atom14'
]

# Feature list A: atomic and backbone features
list_a = [
    'atom14_atom_exists',
    'residx_atom14_to_atom37',
    'residx_atom37_to_atom14',
    'atom37_atom_exists',
    'pseudo_beta',
    'pseudo_beta_mask',
    'all_atom_mask',
    'resolution',
    'all_atom_positions',
    'atom14_gt_exists',
    'atom14_gt_positions',
    'atom14_alt_gt_positions',
    'atom14_alt_gt_exists',
    'atom14_atom_is_ambiguous',
    'backbone_translation',
    'backbone_rotation',
    'backbone_affine_mask'
]

# Template features lists
list_a_templates = [
    'template_aatype',
    'template_all_atom_masks',
    'template_all_atom_positions',
    'template_pseudo_beta',
    'template_pseudo_beta_mask',
    'template_sum_probs',
]

list_b_templates = [
    'template_mask',
]

# Feature list B: sequence and MSA features
list_b = [
    'aatype',
    'residue_index',
    'seq_length',
    'is_distillation',
    'seq_mask',
    'msa_mask',
    'msa_row_mask',
    'random_crop_to_size_seed',
    'extra_msa',
    'extra_msa_mask',
    'extra_msa_row_mask',
    'bert_mask',
    'true_msa',
    'extra_has_deletion',
    'extra_deletion_value',
    'msa_feat',
    'target_feat'
]

# Feature list C: chi angles and rigid group features
list_c = [
    'chi_mask',
    'chi_angles',
    'rigidgroups_gt_frames',
    'rigidgroups_gt_exists',
    'rigidgroups_group_exists',
    'rigidgroups_group_is_ambiguous',
    'rigidgroups_alt_gt_frames'
]


def pseudo_beta_fn_np(
    aatype: npt.NDArray[np.int32],
    all_atom_positions: npt.NDArray[np.float32],
    all_atom_masks: Optional[npt.NDArray[np.float32]] = None
) -> Union[npt.NDArray[np.float32], Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
    """
    Create pseudo beta features.

    For glycine (which lacks CB), uses CA position as pseudo-beta.
    For all other residues, uses CB position.

    Parameters
    ----------
    aatype : np.ndarray
        Amino acid types, shape [...].
    all_atom_positions : np.ndarray
        Atom positions, shape [..., 37, 3].
    all_atom_masks : np.ndarray, optional
        Atom masks, shape [..., 37]. If None, only positions returned.

    Returns
    -------
    np.ndarray or tuple
        If all_atom_masks is None: pseudo_beta positions, shape [..., 3].
        Otherwise: (pseudo_beta, pseudo_beta_mask), both [..., 3] and [...].
    """
    is_gly = np.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = np.where(
        np.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :]
    )

    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(
            is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx]
        )
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def apply_rot_to_vec(
    rot: npt.NDArray[np.float32],
    vec: Union[npt.NDArray[np.float32], List[npt.NDArray[np.float32]]],
    unstack: bool = False
) -> List[npt.NDArray[np.float32]]:
    """
    Multiply rotation matrix by a vector.

    Parameters
    ----------
    rot : np.ndarray
        Rotation matrix, shape [3, 3, ...].
    vec : np.ndarray or list
        Vector to rotate. If unstack=True, shape [..., 3].
        If unstack=False, list of 3 arrays.
    unstack : bool, optional
        Whether to unstack vec first (default: False).

    Returns
    -------
    list of np.ndarray
        Rotated vector as list [x, y, z].
    """
    if unstack:
        x, y, z = [vec[:, i] for i in range(3)]
    else:
        x, y, z = vec
    return [
        rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
        rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
        rot[2][0] * x + rot[2][1] * y + rot[2][2] * z
    ]


def _multiply(
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Multiply two 3x3 rotation matrices.

    Parameters
    ----------
    a : np.ndarray
        First rotation matrix, shape [3, 3, ...].
    b : np.ndarray
        Second rotation matrix, shape [3, 3, ...].

    Returns
    -------
    np.ndarray
        Matrix product a @ b, shape [3, 3, ...].
    """
    return np.stack([
        np.array([
            a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0],
            a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1],
            a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2]
        ]),
        np.array([
            a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0],
            a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1],
            a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2]
        ]),
        np.array([
            a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0],
            a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1],
            a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2]
        ])
    ])


def make_canonical_transform(
    n_xyz: npt.NDArray[np.float32],
    ca_xyz: npt.NDArray[np.float32],
    c_xyz: npt.NDArray[np.float32]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Returns translation and rotation matrices to canonicalize residue atoms.

    Note that this method does not take care of symmetries. If you provide the
    atom positions in the non-standard way, the N atom will end up not at
    [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
    need to take care of such cases in your code.

    Parameters
    ----------
    n_xyz : np.ndarray
        Nitrogen xyz coordinates, shape [batch, 3].
    ca_xyz : np.ndarray
        Carbon alpha xyz coordinates, shape [batch, 3].
    c_xyz : np.ndarray
        Carbon xyz coordinates, shape [batch, 3].

    Returns
    -------
    tuple
        (translation, rotation) where:
        - translation: array of shape [batch, 3] defining the translation.
        - rotation: array of shape [batch, 3, 3] defining the rotation.

        After applying the translation and rotation to all atoms in a residue:
        * All atoms will be shifted so that CA is at the origin.
        * All atoms will be rotated so that C is at the x-axis.
        * All atoms will be shifted so that N is in the xy plane.
    """
    assert len(n_xyz.shape) == 2, n_xyz.shape
    assert n_xyz.shape[-1] == 3, n_xyz.shape
    assert n_xyz.shape == ca_xyz.shape == c_xyz.shape, (
        n_xyz.shape, ca_xyz.shape, c_xyz.shape
    )

    # Place CA at the origin.
    translation = -ca_xyz
    n_xyz = n_xyz + translation
    c_xyz = c_xyz + translation

    # Place C on the x-axis.
    c_x, c_y, c_z = [c_xyz[:, i] for i in range(3)]
    # Rotate by angle c1 in the x-y plane (around the z-axis).
    sin_c1 = -c_y / np.sqrt(1e-20 + c_x**2 + c_y**2)
    cos_c1 = c_x / np.sqrt(1e-20 + c_x**2 + c_y**2)
    zeros = np.zeros_like(sin_c1)
    ones = np.ones_like(sin_c1)
    c1_rot_matrix = np.stack([
        np.array([cos_c1, -sin_c1, zeros]),
        np.array([sin_c1,  cos_c1, zeros]),
        np.array([zeros,    zeros,  ones])
    ])

    # Rotate by angle c2 in the x-z plane (around the y-axis).
    sin_c2 = c_z / np.sqrt(1e-20 + c_x**2 + c_y**2 + c_z**2)
    cos_c2 = np.sqrt(c_x**2 + c_y**2) / np.sqrt(
        1e-20 + c_x**2 + c_y**2 + c_z**2
    )
    c2_rot_matrix = np.stack([
        np.array([cos_c2,  zeros, sin_c2]),
        np.array([zeros,    ones,  zeros]),
        np.array([-sin_c2, zeros, cos_c2])
    ])

    c_rot_matrix = _multiply(c2_rot_matrix, c1_rot_matrix)
    n_xyz = np.stack(apply_rot_to_vec(c_rot_matrix, n_xyz, unstack=True)).T

    # Place N in the x-y plane.
    _, n_y, n_z = [n_xyz[:, i] for i in range(3)]
    # Rotate by angle alpha in the y-z plane (around the x-axis).
    sin_n = -n_z / np.sqrt(1e-20 + n_y**2 + n_z**2)
    cos_n = n_y / np.sqrt(1e-20 + n_y**2 + n_z**2)
    n_rot_matrix = np.stack([
        np.array([ones,  zeros,  zeros]),
        np.array([zeros, cos_n, -sin_n]),
        np.array([zeros, sin_n,  cos_n])
    ])

    return (
        translation,
        np.transpose(_multiply(n_rot_matrix, c_rot_matrix), [2, 0, 1])
    )


def make_transform_from_reference_np(
    n_xyz: npt.NDArray[np.float32],
    ca_xyz: npt.NDArray[np.float32],
    c_xyz: npt.NDArray[np.float32]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Returns rotation and translation matrices to convert from reference.

    Note that this method does not take care of symmetries. If you provide the
    atom positions in the non-standard way, the N atom will end up not at
    [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
    need to take care of such cases in your code.

    Parameters
    ----------
    n_xyz : np.ndarray
        Nitrogen xyz coordinates, shape [batch, 3].
    ca_xyz : np.ndarray
        Carbon alpha xyz coordinates, shape [batch, 3].
    c_xyz : np.ndarray
        Carbon xyz coordinates, shape [batch, 3].

    Returns
    -------
    tuple
        (rotation, translation) where:
        - rotation: array of shape [batch, 3, 3] defining the rotation.
        - translation: array of shape [batch, 3] defining the translation.

        After applying the translation and rotation to the reference backbone,
        the coordinates will approximately equal to the input coordinates.
        The order of translation and rotation differs from make_canonical_transform
        because the rotation from this function should be applied before the
        translation, unlike make_canonical_transform.
    """
    translation, rotation = make_canonical_transform(n_xyz, ca_xyz, c_xyz)
    return np.transpose(rotation, (0, 2, 1)), -translation


def make_atom14_positions(prot: Dict[str, npt.NDArray]) -> Dict[str, npt.NDArray]:
    """
    Constructs denser atom positions (14 dimensions instead of 37).

    Hack: including this function here so we don't have to import alphafold.relax
    since it has some extra dependencies we don't have/need.

    Parameters
    ----------
    prot : dict
        Protein dictionary with keys:
        - 'aatype': amino acid types
        - 'all_atom_positions': atom37 positions
        - 'all_atom_mask': atom37 mask

    Returns
    -------
    dict
        Updated protein dictionary with additional keys:
        - 'atom14_atom_exists': atom14 existence mask
        - 'atom14_gt_exists': atom14 ground truth existence mask
        - 'atom14_gt_positions': atom14 ground truth positions
        - 'residx_atom14_to_atom37': mapping from atom14 to atom37
        - 'residx_atom37_to_atom14': mapping from atom37 to atom14
        - 'atom37_atom_exists': atom37 existence mask
        - 'atom14_alt_gt_positions': alternative atom14 positions
        - 'atom14_alt_gt_exists': alternative atom14 existence mask
        - 'atom14_atom_is_ambiguous': ambiguous atom mask
    """
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[
            residue_constants.restype_1to3[rt]
        ]

        restype_atom14_to_atom37.append([
            (residue_constants.atom_order[name] if name else 0)
            for name in atom_names
        ])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in residue_constants.atom_types
        ])

        restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    # Add dummy mapping for restype 'UNK'.
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.] * 14)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

    # Create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein.
    residx_atom14_to_atom37 = restype_atom14_to_atom37[prot["aatype"]]
    residx_atom14_mask = restype_atom14_mask[prot["aatype"]]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
        prot["all_atom_mask"], residx_atom14_to_atom37, axis=1
    ).astype(np.float32)

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[:, :, None] * (
        np.take_along_axis(
            prot["all_atom_positions"],
            residx_atom14_to_atom37[..., None],
            axis=1
        )
    )

    prot["atom14_atom_exists"] = residx_atom14_mask
    prot["atom14_gt_exists"] = residx_atom14_gt_mask
    prot["atom14_gt_positions"] = residx_atom14_gt_positions

    prot["residx_atom14_to_atom37"] = residx_atom14_to_atom37

    # Create the gather indices for mapping back.
    residx_atom37_to_atom14 = restype_atom37_to_atom14[prot["aatype"]]
    prot["residx_atom37_to_atom14"] = residx_atom37_to_atom14

    # Create the corresponding mask.
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[prot["aatype"]]
    prot["atom37_atom_exists"] = residx_atom37_mask

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [
        residue_constants.restype_1to3[res] for res in residue_constants.restypes
    ]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        correspondences = np.arange(14)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = residue_constants.restype_name_to_atom14_names[
                resname
            ].index(source_atom_swap)
            target_index = residue_constants.restype_name_to_atom14_names[
                resname
            ].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = np.zeros((14, 14), dtype=np.float32)
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.
        all_matrices[resname] = renaming_matrix.astype(np.float32)
    renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[prot["aatype"]]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = np.einsum(
        "rac,rab->rbc",
        residx_atom14_gt_positions,
        renaming_transform
    )
    prot["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = np.einsum(
        "ra,rab->rb",
        residx_atom14_gt_mask,
        renaming_transform
    )

    prot["atom14_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = residue_constants.restype_order[
                residue_constants.restype_3to1[resname]
            ]
            atom_idx1 = residue_constants.restype_name_to_atom14_names[
                resname
            ].index(atom_name1)
            atom_idx2 = residue_constants.restype_name_to_atom14_names[
                resname
            ].index(atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    prot["atom14_atom_is_ambiguous"] = (
        restype_atom14_is_ambiguous[prot["aatype"]]
    )

    return prot


# JAX versions of AlphaFold confidence metric functions
# These are needed since we compute pLDDT/PAE within the binder model


def compute_plddt_jax(logits: jnp.ndarray) -> jnp.ndarray:
    """
    Computes per-residue pLDDT from logits (JAX version).

    Parameters
    ----------
    logits : jnp.ndarray
        Logits from PredictedLDDTHead, shape [num_res, num_bins].

    Returns
    -------
    jnp.ndarray
        Per-residue pLDDT scores (0-100), shape [num_res].
    """
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = jnp.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
    probs = jax.nn.softmax(logits, axis=-1)
    predicted_lddt_ca = jnp.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100


def _calculate_bin_centers_jax(breaks: jnp.ndarray) -> jnp.ndarray:
    """
    Gets the bin centers from the bin edges (JAX version).

    Parameters
    ----------
    breaks : jnp.ndarray
        Error bin edges, shape [num_bins - 1].

    Returns
    -------
    jnp.ndarray
        Error bin centers, shape [num_bins].
    """
    step = (breaks[1] - breaks[0])

    # Add half-step to get the center
    bin_centers = breaks + step / 2
    # Add a catch-all bin at the end.
    bin_centers = jnp.concatenate(
        [bin_centers, jnp.array([bin_centers[-1] + step])],
        axis=0
    )
    return bin_centers


def _calculate_expected_aligned_error_jax(
    alignment_confidence_breaks: jnp.ndarray,
    aligned_distance_error_probs: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculates expected aligned distance errors for every pair of residues (JAX version).

    Parameters
    ----------
    alignment_confidence_breaks : jnp.ndarray
        Error bin edges, shape [num_bins - 1].
    aligned_distance_error_probs : jnp.ndarray
        Predicted probs for each error bin, for each pair of residues,
        shape [num_res, num_res, num_bins].

    Returns
    -------
    tuple
        (predicted_aligned_error, max_predicted_aligned_error) where:
        - predicted_aligned_error: expected aligned distance error for each
          pair of residues, shape [num_res, num_res].
        - max_predicted_aligned_error: maximum predicted error possible (scalar).
    """
    bin_centers = _calculate_bin_centers_jax(alignment_confidence_breaks)

    # Tuple of expected aligned distance error and max possible error.
    return (
        jnp.sum(aligned_distance_error_probs * bin_centers, axis=-1),
        jnp.asarray(bin_centers[-1])
    )


def compute_predicted_aligned_error_jax(
    logits: jnp.ndarray,
    breaks: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """
    Computes aligned confidence metrics from logits (JAX version).

    Parameters
    ----------
    logits : jnp.ndarray
        Logits from PredictedAlignedErrorHead,
        shape [num_res, num_res, num_bins].
    breaks : jnp.ndarray
        Error bin edges, shape [num_bins - 1].

    Returns
    -------
    dict
        Dictionary with keys:
        - 'aligned_confidence_probs': predicted aligned error probabilities
          over bins for each residue pair, shape [num_res, num_res, num_bins].
        - 'predicted_aligned_error': expected aligned distance error for each
          pair of residues, shape [num_res, num_res].
        - 'max_predicted_aligned_error': maximum predicted error possible.
    """
    aligned_confidence_probs = jax.nn.softmax(logits, axis=-1)
    predicted_aligned_error, max_predicted_aligned_error = (
        _calculate_expected_aligned_error_jax(
            alignment_confidence_breaks=breaks,
            aligned_distance_error_probs=aligned_confidence_probs
        )
    )
    return {
        'aligned_confidence_probs': aligned_confidence_probs,
        'predicted_aligned_error': predicted_aligned_error,
        'max_predicted_aligned_error': max_predicted_aligned_error,
    }
