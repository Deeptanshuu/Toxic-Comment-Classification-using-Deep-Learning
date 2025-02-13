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
        
        # Calculate number of complete batches
        self.num_batches = self.num_samples // self.batch_size
        if self.num_batches == 0:
            raise ValueError(f"Batch size {batch_size} is larger than dataset size {self.num_samples}")
        
        # Log dataset statistics
        logger.info(f"Initializing sampler with {self.num_samples} samples")
        logger.info(f"Creating {self.num_batches} batches of size {self.batch_size}")
        
        # Create valid indices array
        self.indices = np.arange(self.num_samples)
        
        # Verify indices are within bounds
        if len(self.indices) > self.num_samples:
            logger.warning(f"Truncating indices from {len(self.indices)} to {self.num_samples}")
            self.indices = self.indices[:self.num_samples]
    
    def __iter__(self):
        try:
            # Create a copy of indices and shuffle
            indices = self.indices.copy()
            np.random.shuffle(indices)
            
            # Calculate total samples to use (only complete batches)
            total_samples = self.num_batches * self.batch_size
            
            # Ensure we don't exceed dataset size
            if total_samples > len(indices):
                logger.warning(f"Reducing samples from {total_samples} to {len(indices)}")
                total_samples = (len(indices) // self.batch_size) * self.batch_size
            
            # Take only the indices we need for complete batches
            valid_indices = indices[:total_samples]
            
            # Reshape into batches and shuffle batch order
            batches = valid_indices.reshape(-1, self.batch_size)
            np.random.shuffle(batches)
            
            # Flatten and verify final indices
            final_indices = batches.flatten()
            
            # Verify all indices are within bounds
            if np.any(final_indices >= self.num_samples):
                raise ValueError(f"Invalid indices generated: max index {final_indices.max()} >= dataset size {self.num_samples}")
            
            logger.info(f"Generated {len(final_indices)} indices in {len(batches)} batches")
            return iter(final_indices)
            
        except Exception as e:
            logger.error(f"Error in sampler iteration: {str(e)}")
            raise
    
    def __len__(self):
        # Return total number of samples that will be yielded
        return self.num_batches * self.batch_size 