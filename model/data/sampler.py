from torch.utils.data import Sampler
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class MultilabelStratifiedSampler(Sampler):
    def __init__(self, labels, groups, batch_size):
        super().__init__(None)
        self.labels = np.array(labels)
        self.groups = np.array(groups)
        self.batch_size = batch_size
        self.num_samples = len(labels)
        
        if self.num_samples == 0:
            raise ValueError("Empty dataset")
            
        logger.info(f"Dataset size: {self.num_samples}")
        logger.info(f"Batch size: {self.batch_size}")
        
        # Create indices per group
        self.group_indices = defaultdict(list)
        for idx, group in enumerate(self.groups):
            self.group_indices[group].append(idx)
        
        # Log group distribution
        for group, indices in self.group_indices.items():
            logger.info(f"Group {group}: {len(indices)} samples")
        
        # Calculate minimum samples per group to ensure representation
        min_group_size = min(len(indices) for indices in self.group_indices.values())
        samples_per_group = (min_group_size // batch_size) * batch_size
        logger.info(f"Using {samples_per_group} samples per group to ensure balanced batches")
        
        # Calculate total batches and samples
        num_groups = len(self.group_indices)
        self.num_batches = (samples_per_group * num_groups) // batch_size
        self.total_samples = self.num_batches * batch_size
        
        logger.info(f"Will use {self.total_samples} total samples in {self.num_batches} batches")
        if self.total_samples < self.num_samples:
            logger.warning(f"Note: Using {self.total_samples}/{self.num_samples} samples to maintain balanced batches")
    
    def __iter__(self):
        try:
            # Shuffle indices within each group
            all_indices = []
            for group in sorted(self.group_indices.keys()):
                group_idx = np.array(self.group_indices[group])
                np.random.shuffle(group_idx)
                samples_for_group = self.total_samples // len(self.group_indices)
                all_indices.extend(group_idx[:samples_for_group])
            
            # Verify we have the right number of indices
            if len(all_indices) != self.total_samples:
                logger.error(f"Generated {len(all_indices)} indices but expected {self.total_samples}")
                # Adjust if needed by truncating or padding
                if len(all_indices) > self.total_samples:
                    all_indices = all_indices[:self.total_samples]
                else:
                    # Pad with random indices if we're short
                    padding_needed = self.total_samples - len(all_indices)
                    padding_indices = np.random.choice(all_indices, size=padding_needed, replace=True)
                    all_indices.extend(padding_indices)
            
            # Convert to numpy array and shuffle
            indices = np.array(all_indices)
            np.random.shuffle(indices)
            
            # Reshape into batches and shuffle batch order
            batches = indices.reshape(self.num_batches, self.batch_size)
            np.random.shuffle(batches)
            
            # Flatten and verify final indices
            final_indices = batches.flatten()
            
            # Final validation
            if len(final_indices) != self.total_samples:
                raise ValueError(f"Generated {len(final_indices)} indices but expected {self.total_samples}")
            if np.any(final_indices >= self.num_samples):
                raise ValueError(f"Invalid indices generated: max index {final_indices.max()} >= dataset size {self.num_samples}")
            
            # Log distribution of groups in final indices
            group_counts = defaultdict(int)
            for idx in final_indices:
                group_counts[self.groups[idx]] += 1
            for group, count in sorted(group_counts.items()):
                logger.debug(f"Group {group} has {count} samples in final indices")
            
            return iter(final_indices.tolist())
            
        except Exception as e:
            logger.error(f"Error in sampler iteration: {str(e)}")
            raise
    
    def __len__(self):
        return self.total_samples 