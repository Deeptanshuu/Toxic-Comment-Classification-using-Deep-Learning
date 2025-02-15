from torch.utils.data import Sampler
import numpy as np
import logging
from collections import defaultdict
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

class MultilabelStratifiedSampler(Sampler):
    def __init__(self, labels, groups, batch_size, cached_size=None):
        super().__init__(None)
        self.labels = np.array(labels)
        self.groups = np.array(groups)
        self.batch_size = batch_size
        self.num_samples = len(labels)
        
        # Simple validation
        if len(self.labels) != len(self.groups):
            raise ValueError("Length mismatch between labels and groups")
        
        # Create indices per group
        self.group_indices = {}
        unique_groups = np.unique(self.groups)
        
        for group in unique_groups:
            indices = np.where(self.groups == group)[0]
            if len(indices) > 0:
                self.group_indices[group] = indices
        
        # Calculate group probabilities
        group_sizes = np.array([len(indices) for indices in self.group_indices.values()])
        self.group_probs = group_sizes / group_sizes.sum()
        self.valid_groups = list(self.group_indices.keys())
        
        # Calculate number of batches
        self.num_batches = self.num_samples // self.batch_size
        if self.num_batches == 0:
            self.num_batches = 1
        self.total_samples = self.num_batches * self.batch_size
    
    def __iter__(self):
        indices = []
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.batch_size):
                # Select group and sample from it
                group = np.random.choice(self.valid_groups, p=self.group_probs)
                idx = np.random.choice(self.group_indices[group])
                batch.append(idx)
            indices.extend(batch)
        
        return iter(indices)
    
    def __len__(self):
        return self.total_samples 