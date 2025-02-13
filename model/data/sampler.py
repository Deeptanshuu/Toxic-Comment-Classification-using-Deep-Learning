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
        
        # Get indices for each group
        self.group_indices = {}
        for group in np.unique(groups):
            self.group_indices[group] = np.where(groups == group)[0]
            
        # Calculate samples per group per batch
        total_samples = sum(len(indices) for indices in self.group_indices.values())
        self.group_ratios = {
            group: len(indices) / total_samples
            for group, indices in self.group_indices.items()
        }
    
    def __iter__(self):
        # Generate indices for all complete batches
        all_indices = []
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Calculate samples needed from each group for this batch
            remaining = self.batch_size
            for group, ratio in self.group_ratios.items():
                if remaining <= 0:
                    break
                    
                # Calculate number of samples for this group
                n_samples = max(1, min(
                    int(self.batch_size * ratio),
                    remaining,
                    len(self.group_indices[group])
                ))
                
                # Sample indices for this group
                sampled = np.random.choice(
                    self.group_indices[group],
                    size=n_samples,
                    replace=False
                )
                batch_indices.extend(sampled)
                remaining -= n_samples
            
            # If we still need more samples, take them randomly
            if remaining > 0:
                all_available = np.concatenate(list(self.group_indices.values()))
                extra = np.random.choice(
                    all_available,
                    size=remaining,
                    replace=False
                )
                batch_indices.extend(extra)
            
            # Shuffle the batch indices
            np.random.shuffle(batch_indices)
            all_indices.extend(batch_indices)
        
        return iter(all_indices)
    
    def __len__(self):
        return self.num_batches * self.batch_size 