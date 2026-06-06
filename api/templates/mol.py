# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List
from copy import deepcopy

import torch
import numpy as np

from data.file_loader import MolType, MolLoader
from data.bioparse.utils import recur_index, index_to_numerical_index
from data.bioparse.parser.sdf_to_complex import sdf_to_complex
from data.resample import SizeSamplerByPocketSpace
from data.bioparse.hierarchy import Atom, Block, Bond, BondType
import utils.register as R
from models.modules.adapter.model import ConditionConfig

from .base import BaseTemplate, ComplexDesc


@R.register('Molecule')
class Molecule(BaseTemplate):

    def __init__(self, size_min=None, size_max=None):
        super().__init__(size_min, size_max)
        self.size_sampler = SizeSamplerByPocketSpace(size_min, size_max)
    
    @property
    def moltype(self) -> MolType:
        return MolType.MOLECULE
    
    def default_filter_configs(self) -> List[dict]:
        return super().default_filter_configs() + [ { 'class': 'MolSMARTSFilter' } ]

    def sample_size(self, cplx_desc):
        pocket_pos = []
        for _id in cplx_desc.pocket_block_ids:
            block = recur_index(cplx_desc.cplx, _id)
            for atom in block: pocket_pos.append(atom.coordinate)
        
        num_blocks = self.size_sampler(1, pocket_pos)[0]

        return num_blocks
    
    def dummy_lig_block_bonds(self, cplx_desc: ComplexDesc, size: int) -> List[Block]:
        blocks = [Block(
            name='UNK',
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(1, str(i))  # the same residue
        ) for i in range(size)]
        bonds = []
        return blocks, bonds

    def validate(self, cplx_desc):
        return True

    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        loader = MolLoader(
            tgt_chains=cplx_desc.tgt_chains,
            lig_chains=cplx_desc.lig_chains,
        )
        return loader.cplx_to_data(
            cplx_desc.cplx,
            cplx_desc.pocket_block_ids,
            cplx_desc.lig_block_ids
        )


@R.register('MoleculeResample')
class MoleculeResample(Molecule):
    def __init__(self):
        super().__init__(size_min=None, size_max=None)

    ########## Override Basic Functions to Maintain the Original Ligand ##########
    @property
    def resample_mode(self):
        return True

    def remove_ref_lig(self, cplx_desc):
        return cplx_desc.cplx

    def add_dummy_lig(self, cplx_desc):
        return cplx_desc.cplx


def _copy_fragment_complex(fragment_cplx, mol_index):
    blocks, bonds, block_map = [], [], {}

    for old_mol_idx, mol in enumerate(fragment_cplx):
        for old_block_idx, block in enumerate(mol):
            new_block = deepcopy(block)
            new_block.id = (1, str(len(blocks)))
            new_block.properties = {}
            block_map[(old_mol_idx, old_block_idx)] = len(blocks)
            blocks.append(new_block)

    for bond in fragment_cplx.bonds:
        block_i = block_map[(bond.index1[0], bond.index1[1])]
        block_j = block_map[(bond.index2[0], bond.index2[1])]
        bonds.append(Bond(
            index1=(mol_index, block_i, bond.index1[2]),
            index2=(mol_index, block_j, bond.index2[2]),
            bond_type=bond.bond_type
        ))

    return blocks, bonds


def _resolve_fragment_atom(blocks, atom_label):
    """Resolve an atom in copied fragment blocks.

    atom_label should be an original SDF atom id (1-based)
    """
    for block_idx, block in enumerate(blocks):
        for atom_idx, atom in enumerate(block):
            if atom.id == str(atom_label): return (block_idx, atom_idx)
    return (None, None)

@R.register('MoleculeCompletion')
class MoleculeCompletion(Molecule):
    def __init__(self, fragment_sdf, add_block_size_min, add_block_size_max, block_centers = None, block_center_stds = None, w = 1.0):
        super().__init__(size_min=None, size_max=None)
        self.add_block_size_min = add_block_size_min
        self.add_block_size_max = add_block_size_max
        self.block_centers = [] if block_centers is None else block_centers
        self.block_center_stds = [1.0 for _ in self.block_centers] if block_center_stds is None else block_center_stds
        assert self.add_block_size_min < self.add_block_size_max, f'add_block_size_min ({self.add_block_size_min}) should be < add_block_size_max ({self.add_block_size_max})'
        assert len(self.block_centers) <= self.add_block_size_min, f'The number of specified block centers ({len(self.block_centers)}) should be <= add_block_size_min ({self.add_block_size_min})'
        assert len(self.block_centers) == len(self.block_center_stds), f'The number of specified block centers ({len(self.block_centers)}) should be equal to the number of specified block center stds ({len(self.block_center_stds)})'
        self.w = w
        self.wt_fragments = sdf_to_complex(fragment_sdf)
        self.num_fragment_blocks = sum(len(mol) for mol in self.wt_fragments)

    def sample_size(self, cplx_desc: ComplexDesc):
        return np.random.randint(self.add_block_size_min, self.add_block_size_max)

    def dummy_lig_block_bonds(self, cplx_desc: ComplexDesc, size: int) -> List[Block]:
        blocks, bonds = _copy_fragment_complex(self.wt_fragments, len(cplx_desc.cplx))
        for center, std in zip(self.block_centers, self.block_center_stds):
            center = (np.array(center) + std * np.random.randn(3)).tolist()
            blocks.append(Block(
                name='UNK',
                atoms=[Atom(name='C', coordinate=center, element='C', id=-1)],
                id=(1, str(len(blocks)))    # the same residue
            ))
        for _ in range(size - len(self.block_centers)):
            blocks.append(Block(
                name='UNK',
                atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
                id=(1, str(len(blocks)))    # the same residue
            ))
        return blocks, bonds

    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        # add topo conditioning
        fix_block = [0 for _ in cplx_desc.lig_block_ids]
        for i in range(self.num_fragment_blocks): fix_block[i] = 1 # the given fragments
        # add coord conditioning
        fix_block_center = [m for m in fix_block]
        for i in range(len(self.block_centers)):    # additional center control
            fix_block_center[self.num_fragment_blocks + i] = 1
        mask_2d = [0 for _ in cplx_desc.pocket_block_ids] + fix_block
        mask_3d = [0 for _ in cplx_desc.pocket_block_ids] + fix_block_center
        data['condition_config'] = ConditionConfig(
            mask_2d=torch.tensor(mask_2d, dtype=torch.bool),
            mask_3d=torch.tensor(mask_3d, dtype=torch.bool),
            mask_incomplete_2d=None,
            w=self.w
        )
        return data


@R.register('CovalentMolecule')
class CovalentMolecule(Molecule):
    def __init__(
            self,
            fragment_sdf,
            tgt_connect_atom_label,
            fragment_connect_atom_label,
            tgt_connect_bond_level=1,   # 1: single bond, 2: double bond, ...
            w=0.5):
        super().__init__()
        self.covalent_fragments = sdf_to_complex(fragment_sdf)
        self.num_fragment_blocks = sum(len(mol) for mol in self.covalent_fragments)
        self.tgt_connect_atom_label = tgt_connect_atom_label
        self.fragment_connect_atom_label = fragment_connect_atom_label
        self.tgt_connect_bond_level = tgt_connect_bond_level
        self.covalent_bond = BondType(tgt_connect_bond_level)
        self.w = w
        # dynamic
        self._covalent_bond_index = None

    @property
    def additional_sample_opt(self) -> dict:
        return { 'vae_disable_avoid_clash': True }

    def default_filter_configs(self)-> List[dict]:
        configs = []
        for config in super().default_filter_configs():
            if config['class'] == 'PhysicalValidityFilter':     # replace with faster detection
                configs.append({ 'class': 'PhysicalValidityFilter', 'skip_clash': True })
            else: configs.append(config)
        return configs

    def dummy_lig_block_bonds(self, cplx_desc: ComplexDesc, size: int) -> List[Block]:
        blocks, bonds = _copy_fragment_complex(self.covalent_fragments, len(cplx_desc.cplx))
        fragment_block_idx, fragment_atom_idx = _resolve_fragment_atom(
            blocks, self.fragment_connect_atom_label
        )
        assert fragment_block_idx is not None
        # add left blocks
        add_num_blocks = size - len(blocks)
        if add_num_blocks > 0:
            blocks.extend([Block(
                name='UNK',
                atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
                id=(1, str(i + len(blocks)))  # the same residue
            ) for i in range(add_num_blocks)])

        tgt_block_index = (
            self.tgt_connect_atom_label[0],
            (self.tgt_connect_atom_label[1], self.tgt_connect_atom_label[2])
        )
        tgt_block = recur_index(cplx_desc.cplx, tgt_block_index)
        tgt_block_index_numerical = index_to_numerical_index(cplx_desc.cplx, tgt_block_index)

        atom_index = None
        for i, atom in enumerate(tgt_block):
            if atom.name == self.tgt_connect_atom_label[3]:
                atom_index = i
                break
        assert atom_index is not None, f'Target atom {self.tgt_connect_atom_label} not found'

        self._covalent_bond_index = (
            (tgt_block_index_numerical[0], tgt_block_index_numerical[1], atom_index),
            (len(cplx_desc.cplx), fragment_block_idx, fragment_atom_idx),
            self.covalent_bond
        )
        bonds.append(Bond(*self._covalent_bond_index))
        return blocks, bonds

    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        tgt_block_id = (
            self.tgt_connect_atom_label[0],
            (self.tgt_connect_atom_label[1], self.tgt_connect_atom_label[2])
        )
        fixed_fragment_mask = [
            1 if i < self.num_fragment_blocks else 0
            for i in range(len(cplx_desc.lig_block_ids))
        ]
        mask_2d = [0 if id != tgt_block_id else 1 for id in cplx_desc.pocket_block_ids] + fixed_fragment_mask
        mask_3d = [0 for _ in cplx_desc.pocket_block_ids] + fixed_fragment_mask
        data['condition_config'] = ConditionConfig(
            mask_2d=torch.tensor(mask_2d, dtype=torch.bool),
            mask_3d=torch.tensor(mask_3d, dtype=torch.bool),
            mask_incomplete_2d=None,
            w=self.w
        )
        return data

    def set_desc_attr(self, cplx_desc: ComplexDesc):
        if cplx_desc.additional_props is None: cplx_desc.additional_props = {}
        cplx_desc.additional_props['covalent_modifications'] = [
            (self._covalent_bond_index[0], self._covalent_bond_index[1], self.tgt_connect_bond_level)
        ]
        return super().set_desc_attr(cplx_desc)
