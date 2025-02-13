from torch.utils.data import Sampler
import numpy as np

class MultilabelStratifiedSampler(Sampler):
    def __init__(self, labels, groups, batch_size):
        super().__init__(None)
        self.labels = labels
        self.groups = groups
        self.batch_size = batch_size
        self.num_samples = len(labels)
        
        # Calculate number of complete batches
        self.num_batches = self.num_samples // self.batch_size
        if self.num_batches == 0:
            raise ValueError(f"Batch size {batch_size} is larger than dataset size {self.num_samples}")
        
        # Store valid indices
        self.indices = np.arange(self.num_samples)
    
    def __iter__(self):
        # Shuffle indices for this epoch
        indices = self.indices.copy()
        np.random.shuffle(indices)
        
        # Return only complete batches
        valid_indices = indices[:self.num_batches * self.batch_size]
        
        # Reshape into batches and shuffle batch order
        batches = valid_indices.reshape(-1, self.batch_size)
        np.random.shuffle(batches)
        
        # Flatten and return iterator
        return iter(batches.flatten())
    
    def __len__(self):
        # Return total number of samples that will be yielded
        return self.num_batches * self.batch_size 