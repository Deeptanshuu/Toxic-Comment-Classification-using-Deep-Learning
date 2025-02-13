from torch.utils.data import Sampler
import numpy as np

class MultilabelStratifiedSampler(Sampler):
    def __init__(self, labels, groups, batch_size):
        super().__init__(None)
        self.num_samples = len(labels)
        self.batch_size = batch_size
        self.indices = np.arange(self.num_samples)
        
        # Calculate number of batches
        self.num_batches = self.num_samples // self.batch_size
        
    def __iter__(self):
        # Shuffle all indices
        indices = self.indices.copy()
        np.random.shuffle(indices)
        
        # Only return complete batches
        n = self.num_batches * self.batch_size
        return iter(indices[:n])
    
    def __len__(self):
        return self.num_batches * self.batch_size 