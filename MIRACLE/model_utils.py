import torch
import pdb
from typing import List, Tuple


def compute_max_atoms(scope: List[Tuple[int, int]]) -> int:
    """Given the scopes for a batch of molecules, computes max atoms."""
    max_atoms = 0
    for st, le in scope:
        if le > max_atoms:
            max_atoms = le
    return max_atoms


def convert_to_3D(input, scope, max_atoms, device, self_attn=True):
    """Converts the input to a 3D batch matrix

    Args:
        input: A tensor of shape [# atoms, # features]
        scope: A list of start/length indices for the molecules
        max_atoms: The maximum number of atoms for padding purposes
        device: For creating tensors
    Returns:
        A matrix of size [batch_size, max atoms, # features]
    """
    n_features = input.size()[1]

    batch_input = []
    batch_mask = []
    for (st, le) in scope:

        mol_input = input.narrow(0, st, le)

        n_atoms = le
        n_padding = max_atoms - le

        mask = torch.ones([n_atoms], device=device)

        if n_padding > 0:
            mask = torch.cat(
                [mask, torch.zeros([n_padding], device=device)])
            mol_input_padded = torch.cat(
                [mol_input, torch.zeros([n_padding, n_features], device=device)])
            batch_input.append(mol_input_padded)
        else:
            batch_input.append(mol_input)

        mask = mask.repeat([max_atoms, 1]) * mask.unsqueeze(1)
        if not self_attn:
            for i in range(max_atoms):
                mask[i, i] = 0
        batch_mask.append(mask)

    batch_input = torch.stack(batch_input, dim=0)
    batch_mask = torch.stack(batch_mask, dim=0).byte()
    return batch_input, batch_mask


def convert_to_2D(input, scope):
    """Convert back to 2D

    Args:
        input: A tensor of shape [batch size, max padding, # features]
        scope: A list of start/length indices for the molecules
    Returns:
        A matrix of size [# atoms, # features]
    """
    input_2D = []

    for idx, (_, le) in enumerate(scope):
        mol_input = input[idx].narrow(0, 0, le)
        input_2D.append(mol_input)

    input_2D = torch.cat(input_2D, dim=0)
    return input_2D
