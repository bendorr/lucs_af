"""
train_utils.py

Ben Orr
3.28.24
Adapted from https://github.com/phbradley/alphafold_finetune

AF training utilities - structure manipulation, coordinate transforms, JAX confidence metrics (pLDDT, PAE).
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from alphafold.common import residue_constants


# Feature keys for PDB structure processing
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

# Atomic and backbone features
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

# Template features
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

# Sequence and MSA features
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

# Chi angles and rigid group features
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
    Pseudo-beta features. Gly uses CA, others CB.

    Parameters
    ----------
    aatype : np.ndarray
        AA types, shape [...]
    all_atom_positions : np.ndarray
        Atom positions, [..., 37, 3]
    all_atom_masks : np.ndarray, optional
        Atom masks, [..., 37] (None = positions only)

    Returns
    -------
    np.ndarray or tuple
        If all_atom_masks None: pseudo_beta [..., 3]
        Else: (pseudo_beta, pseudo_beta_mask)
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
    Rotation matrix × vector multiplication.

    Parameters
    ----------
    rot : np.ndarray
        Rotation matrix, [3, 3, ...]
    vec : np.ndarray or list
        Vector (unstack=True: [..., 3], False: list of 3)
    unstack : bool
        Unstack vec first, default False

    Returns
    -------
    list of np.ndarray
        Rotated vector [x, y, z]
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
    3x3 rotation matrix multiplication.

    Parameters
    ----------
    a : np.ndarray
        First rotation matrix, [3, 3, ...]
    b : np.ndarray
        Second rotation matrix, [3, 3, ...]

    Returns
    -------
    np.ndarray
        Product a @ b, [3, 3, ...]
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
    Canonicalize residue atoms via translation/rotation.
    CA at origin, C on x-axis, N in xy plane.

    Note: Doesn't handle symmetries. Non-standard positions
    may place N at [-0.527250, -1.359329, 0.0] vs [-, +1.359329, -].

    Parameters
    ----------
    n_xyz : np.ndarray
        N coords, [batch, 3]
    ca_xyz : np.ndarray
        CA coords, [batch, 3]
    c_xyz : np.ndarray
        C coords, [batch, 3]

    Returns
    -------
    tuple
        (translation, rotation)
        translation: [batch, 3]
        rotation: [batch, 3, 3]
    """
    assert len(n_xyz.shape) == 2, n_xyz.shape
    assert n_xyz.shape[-1] == 3, n_xyz.shape
    assert n_xyz.shape == ca_xyz.shape == c_xyz.shape, (
        n_xyz.shape, ca_xyz.shape, c_xyz.shape
    )

    # Place CA at origin
    translation = -ca_xyz
    n_xyz = n_xyz + translation
    c_xyz = c_xyz + translation

    # Place C on x-axis
    c_x, c_y, c_z = [c_xyz[:, i] for i in range(3)]
    # Rotate by angle c1 in xy plane (around z-axis)
    sin_c1 = -c_y / np.sqrt(1e-20 + c_x**2 + c_y**2)
    cos_c1 = c_x / np.sqrt(1e-20 + c_x**2 + c_y**2)
    zeros = np.zeros_like(sin_c1)
    ones = np.ones_like(sin_c1)
    c1_rot_matrix = np.stack([
        np.array([cos_c1, -sin_c1, zeros]),
        np.array([sin_c1,  cos_c1, zeros]),
        np.array([zeros,    zeros,  ones])
    ])

    # Rotate by angle c2 in xz plane (around y-axis)
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

    # Place N in xy plane
    _, n_y, n_z = [n_xyz[:, i] for i in range(3)]
    # Rotate by angle alpha in yz plane (around x-axis)
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
    Convert from reference coords via rotation/translation.
    Inverse of make_canonical_transform - apply rotation before translation.

    Note: Doesn't handle symmetries (see make_canonical_transform).

    Parameters
    ----------
    n_xyz : np.ndarray
        N coords, [batch, 3]
    ca_xyz : np.ndarray
        CA coords, [batch, 3]
    c_xyz : np.ndarray
        C coords, [batch, 3]

    Returns
    -------
    tuple
        (rotation, translation)
        rotation: [batch, 3, 3]
        translation: [batch, 3]
    """
    translation, rotation = make_canonical_transform(n_xyz, ca_xyz, c_xyz)
    return np.transpose(rotation, (0, 2, 1)), -translation


def make_atom14_positions(prot: Dict[str, npt.NDArray]) -> Dict[str, npt.NDArray]:
    """
    Denser atom positions (14 dims vs 37).

    Avoids alphafold.relax import (extra dependencies).

    Parameters
    ----------
    prot : dict
        Protein dict with 'aatype', 'all_atom_positions', 'all_atom_mask'

    Returns
    -------
    dict
        Updated protein dict with added keys:
        atom14_atom_exists, atom14_gt_exists, atom14_gt_positions,
        residx_atom14_to_atom37, residx_atom37_to_atom14, atom37_atom_exists,
        atom14_alt_gt_positions, atom14_alt_gt_exists, atom14_atom_is_ambiguous
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

    # Add dummy mapping for 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.] * 14)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

    # Create mapping (residx, atom14) --> atom37
    # Shape (num_res, 14) with atom37 indices
    residx_atom14_to_atom37 = restype_atom14_to_atom37[prot["aatype"]]
    residx_atom14_mask = restype_atom14_mask[prot["aatype"]]

    # Mask for known ground truth positions
    residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
        prot["all_atom_mask"], residx_atom14_to_atom37, axis=1
    ).astype(np.float32)

    # Gather ground truth positions
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

    # Gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[prot["aatype"]]
    prot["residx_atom37_to_atom14"] = residx_atom37_to_atom14

    # Create corresponding mask
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[prot["aatype"]]
    prot["atom37_atom_exists"] = residx_atom37_mask

    # Atom naming ambiguous for 7/20 AAs - provide alternative GT coords with swapped naming
    restype_3 = [
        residue_constants.restype_1to3[res] for res in residue_constants.restypes
    ]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms
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

    # Pick transformation matrices for given residue sequence, shape (num_res, 14, 14)
    renaming_transform = renaming_matrices[prot["aatype"]]

    # Apply to ground truth positions, shape (num_res, 14, 3)
    alternative_gt_positions = np.einsum(
        "rac,rab->rbc",
        residx_atom14_gt_positions,
        renaming_transform
    )
    prot["atom14_alt_gt_positions"] = alternative_gt_positions

    # Mask for alternative GT (differs if only one atom in ambiguous pair has GT position)
    alternative_gt_mask = np.einsum(
        "ra,rab->rb",
        residx_atom14_gt_mask,
        renaming_transform
    )

    prot["atom14_alt_gt_exists"] = alternative_gt_mask

    # Ambiguous atoms mask, shape (21, 14)
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

    # Ambiguous mask for given sequence
    prot["atom14_atom_is_ambiguous"] = (
        restype_atom14_is_ambiguous[prot["aatype"]]
    )

    return prot


# JAX versions of AF confidence metrics (needed for computing pLDDT/PAE within binder model)


def compute_plddt_jax(logits: jnp.ndarray) -> jnp.ndarray:
    """
    Per-residue pLDDT from logits (JAX).

    Parameters
    ----------
    logits : jnp.ndarray
        PredictedLDDTHead logits, [num_res, num_bins]

    Returns
    -------
    jnp.ndarray
        pLDDT scores (0-100), [num_res]
    """
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = jnp.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
    probs = jax.nn.softmax(logits, axis=-1)
    predicted_lddt_ca = jnp.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100


def _calculate_bin_centers_jax(breaks: jnp.ndarray) -> jnp.ndarray:
    """
    Bin centers from edges (JAX).

    Parameters
    ----------
    breaks : jnp.ndarray
        Error bin edges, [num_bins - 1]

    Returns
    -------
    jnp.ndarray
        Error bin centers, [num_bins]
    """
    step = (breaks[1] - breaks[0])

    # Add half-step for center
    bin_centers = breaks + step / 2
    # Add catch-all bin at end
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
    Expected aligned distance errors per residue pair (JAX).

    Parameters
    ----------
    alignment_confidence_breaks : jnp.ndarray
        Error bin edges, [num_bins - 1]
    aligned_distance_error_probs : jnp.ndarray
        Predicted probs per bin per pair, [num_res, num_res, num_bins]

    Returns
    -------
    tuple
        (predicted_aligned_error, max_predicted_aligned_error)
        predicted_aligned_error: [num_res, num_res]
        max_predicted_aligned_error: scalar
    """
    bin_centers = _calculate_bin_centers_jax(alignment_confidence_breaks)

    return (
        jnp.sum(aligned_distance_error_probs * bin_centers, axis=-1),
        jnp.asarray(bin_centers[-1])
    )


def compute_predicted_aligned_error_jax(
    logits: jnp.ndarray,
    breaks: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """
    Aligned confidence metrics from logits (JAX).

    Parameters
    ----------
    logits : jnp.ndarray
        PredictedAlignedErrorHead logits, [num_res, num_res, num_bins]
    breaks : jnp.ndarray
        Error bin edges, [num_bins - 1]

    Returns
    -------
    dict
        'aligned_confidence_probs': [num_res, num_res, num_bins]
        'predicted_aligned_error': [num_res, num_res]
        'max_predicted_aligned_error': scalar
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
