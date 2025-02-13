import random
from collections import defaultdict
from torch.utils.data import Sampler
import numpy as np
import torch

class MultilabelStratifiedSampler(Sampler):
    def __init__(self, labels, groups, batch_size, min_samples_per_lang=4):
        super().__init__(None)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        self.labels = labels
        self.groups = np.asarray(groups)
        self.batch_size = batch_size
        self.min_samples_per_lang = min_samples_per_lang
        self.unique_groups = np.unique(self.groups)
        
        if len(self.unique_groups) == 0:
            raise ValueError("No groups found in the dataset")
            
        self.num_classes = labels.shape[1]
        
        # Calculate minimum batch size needed
        min_batch_size = self.min_samples_per_lang * len(self.unique_groups)
        if self.batch_size < min_batch_size:
            raise ValueError(f"Batch size {batch_size} is too small to satisfy minimum {min_samples_per_lang} samples per language. Need at least {min_batch_size}.")
        
        # Pre-compute indices per group
        self.indices_per_group = self._get_indices_per_group()
        
        # Calculate number of batches
        total_samples = len(self.labels)
        self.num_batches = total_samples // self.batch_size + (1 if total_samples % self.batch_size != 0 else 0)
        
    def _get_indices_per_group(self):
        indices_per_group = defaultdict(list)
        for idx in range(len(self.labels)):
            group = self.groups[idx]
            indices_per_group[group].append(idx)
        return indices_per_group
    
    def _generate_batch_indices(self):
        # Calculate base samples per group
        samples_per_group = self.batch_size // len(self.unique_groups)
        extra_samples = self.batch_size % len(self.unique_groups)
        
        batch_indices = []
        
        # Get samples from each group
        for group in self.unique_groups:
            group_indices = self.indices_per_group[group]
            if not group_indices:
                continue
                
            # Get number of samples for this group
            n_samples = samples_per_group + (1 if extra_samples > 0 else 0)
            extra_samples = max(0, extra_samples - 1)
            
            # Randomly sample indices
            if len(group_indices) < n_samples:
                # If not enough samples, use all with replacement
                sampled = np.random.choice(group_indices, size=n_samples, replace=True)
            else:
                # Otherwise sample without replacement
                sampled = np.random.choice(group_indices, size=n_samples, replace=False)
            
            batch_indices.extend(sampled)
        
        # Shuffle the batch indices
        np.random.shuffle(batch_indices)
        return batch_indices
    
    def __iter__(self):
        all_indices = []
        for _ in range(self.num_batches):
            batch_indices = self._generate_batch_indices()
            all_indices.extend(batch_indices)
        return iter(all_indices)
    
    def __len__(self):
        return self.num_batches 