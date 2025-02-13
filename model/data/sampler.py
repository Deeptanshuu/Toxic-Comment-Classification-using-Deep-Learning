from torch.utils.data import Sampler
import numpy as np

class MultilabelStratifiedSampler(Sampler):
    def __init__(self, labels, groups, batch_size):
        super().__init__(None)
        self.labels = labels
        self.groups = groups
        self.batch_size = batch_size
        self.num_samples = len(labels)
        
        # Calculate number of batches
        self.num_batches = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.num_batches += 1
        
        # Get group sizes for proportional sampling
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        self.group_weights = group_counts / group_counts.sum()
        self.unique_groups = unique_groups
    
    def __iter__(self):
        # Generate indices for one epoch
        indices = []
        samples_per_batch = {
            group: int(self.batch_size * weight)
            for group, weight in zip(self.unique_groups, self.group_weights)
        }
        
        # Ensure we have at least one sample per group
        remaining = self.batch_size - sum(samples_per_batch.values())
        if remaining > 0:
            # Distribute remaining samples proportionally
            for group in samples_per_batch:
                samples_per_batch[group] += 1
                remaining -= 1
                if remaining == 0:
                    break
        
        # Generate indices for all batches
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Sample from each group
            for group in self.unique_groups:
                # Get indices for this group
                group_indices = np.where(self.groups == group)[0]
                n_samples = samples_per_batch[group]
                
                # Sample with replacement if needed
                sampled = np.random.choice(
                    group_indices, 
                    size=n_samples,
                    replace=len(group_indices) < n_samples
                )
                batch_indices.extend(sampled)
            
            # Shuffle batch indices
            np.random.shuffle(batch_indices)
            indices.extend(batch_indices)
        
        return iter(indices)
    
    def __len__(self):
        return self.num_batches * self.batch_size 