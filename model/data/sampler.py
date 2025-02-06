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
        
        # Calculate weights for balanced sampling
        self.weights = self._calculate_weights()
        
        # Calculate samples per group
        self.samples_per_group = max(2, self.batch_size // len(self.unique_groups))
        self.remainder = self.batch_size - (self.samples_per_group * len(self.unique_groups))
    
    def _calculate_weights(self):
        try:
            weights = torch.ones(len(self.labels))
            for group in self.unique_groups:
                group_mask = self.groups == group
                if not np.any(group_mask):
                    continue
                    
                group_labels = self.labels[torch.from_numpy(group_mask)]
                
                if len(group_labels) == 0:
                    continue
                
                # Calculate inverse frequency for each class
                class_weights = []
                for c in range(self.num_classes):
                    pos_count = group_labels[:, c].sum().item()
                    if pos_count > 0:
                        w = len(group_labels) / (2 * pos_count)
                        class_weights.append(w)
                    else:
                        class_weights.append(1.0)
                
                # Apply weights to samples in this group
                group_weights = torch.tensor(class_weights).mean()
                weights[torch.from_numpy(group_mask)] = group_weights
            
            return weights
            
        except Exception as e:
            print(f"Warning: Error calculating weights: {str(e)}")
            return torch.ones(len(self.labels))
    
    def __iter__(self):
        try:
            # Convert weights to probabilities safely
            weights = self.weights.clamp(min=1e-5)
            probs = weights / weights.sum()
            
            # Generate indices ensuring each batch has samples from each group
            indices = []
            remaining_indices = set(range(len(self.labels)))
            
            while remaining_indices:
                batch_indices = []
                
                # First, ensure minimum samples from each group
                for group in self.unique_groups:
                    group_mask = self.groups == group
                    group_indices = np.where(group_mask)[0]
                    available_indices = list(set(group_indices) & remaining_indices)
                    
                    if available_indices:
                        # Get probabilities for available indices
                        group_probs = probs[available_indices].numpy()
                        group_probs /= group_probs.sum()
                        
                        # Sample indices
                        try:
                            selected = np.random.choice(
                                available_indices,
                                size=min(self.samples_per_group, len(available_indices)),
                                p=group_probs,
                                replace=False
                            )
                            batch_indices.extend(selected)
                            remaining_indices -= set(selected)
                        except ValueError:
                            continue
                
                # Fill remaining slots if any
                if self.remainder > 0 and remaining_indices:
                    available_probs = probs[list(remaining_indices)].numpy()
                    available_probs /= available_probs.sum()
                    
                    try:
                        remaining = np.random.choice(
                            list(remaining_indices),
                            size=min(self.remainder, len(remaining_indices)),
                            p=available_probs,
                            replace=False
                        )
                        batch_indices.extend(remaining)
                        remaining_indices -= set(remaining)
                    except ValueError:
                        pass
                
                if batch_indices:
                    random.shuffle(batch_indices)
                    indices.extend(batch_indices)
            
            return iter(indices)
            
        except Exception as e:
            print(f"Warning: Error in sampling: {str(e)}")
            # Fallback to random sampling
            indices = list(range(len(self.labels)))
            random.shuffle(indices)
            return iter(indices)
    
    def __len__(self):
        return len(self.labels) 