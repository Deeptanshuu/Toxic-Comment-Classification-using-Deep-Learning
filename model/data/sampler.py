from torch.utils.data import Sampler
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MultilabelStratifiedSampler(Sampler):
    def __init__(self, labels, groups, batch_size):
        super().__init__(None)
        self.labels = labels
        self.groups = groups
        self.batch_size = batch_size
        self.num_samples = len(labels)
        
        if self.num_samples == 0:
            raise ValueError("Empty dataset")
            
        logger.info(f"Dataset size: {self.num_samples}")
        logger.info(f"Batch size: {self.batch_size}")
        
        # Store valid indices
        self.indices = np.arange(self.num_samples)
        
        # Calculate number of complete batches
        self.num_batches = len(self.indices) // self.batch_size
        if self.num_batches == 0:
            raise ValueError(f"Batch size {batch_size} is larger than dataset size {self.num_samples}")
            
        # Calculate total samples that will be used
        self.total_samples = self.num_batches * self.batch_size
        
        logger.info(f"Will use {self.total_samples} samples in {self.num_batches} batches")
        if self.total_samples < self.num_samples:
            logger.warning(f"Dropping {self.num_samples - self.total_samples} samples to maintain complete batches")
    
    def __iter__(self):
        try:
            # Create a copy of indices and shuffle
            indices = self.indices.copy()
            np.random.shuffle(indices)
            
            # Take only the indices we need for complete batches
            valid_indices = indices[:self.total_samples]
            
            # Reshape into batches and shuffle batch order
            batches = valid_indices.reshape(self.num_batches, self.batch_size)
            np.random.shuffle(batches)
            
            # Flatten and verify final indices
            final_indices = batches.flatten()
            
            # Double check indices are within bounds
            if len(final_indices) != self.total_samples:
                raise ValueError(f"Generated {len(final_indices)} indices but expected {self.total_samples}")
            if np.any(final_indices >= self.num_samples):
                raise ValueError(f"Invalid indices generated: max index {final_indices.max()} >= dataset size {self.num_samples}")
            
            logger.debug(f"Returning {len(final_indices)} indices in {self.num_batches} batches")
            return iter(final_indices)
            
        except Exception as e:
            logger.error(f"Error in sampler iteration: {str(e)}")
            raise
    
    def __len__(self):
        return self.total_samples 