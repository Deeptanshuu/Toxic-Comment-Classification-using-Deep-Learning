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
        
        # Validate input arrays
        if len(self.labels) != len(self.groups):
            raise ValueError(
                f"Length mismatch: labels ({len(self.labels)}) != groups ({len(self.groups)})"
            )
        
        # Use cached size if provided, otherwise use minimum of labels and groups length
        self.num_samples = min(
            cached_size if cached_size is not None else len(labels),
            len(self.groups)
        )
        
        if self.num_samples == 0:
            raise ValueError("Empty dataset")
        
        # Log dataset statistics
        logger.info(f"Dataset size: {self.num_samples}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Labels shape: {self.labels.shape}")
        logger.info(f"Groups shape: {self.groups.shape}")
        
        # Create indices per group with validation
        self.group_indices = defaultdict(list)
        
        # Calculate maximum valid index
        max_valid_idx = min(len(self.groups), self.num_samples) - 1
        valid_indices = np.arange(max_valid_idx + 1)
        
        # Convert groups to numeric IDs if they're strings
        unique_groups = np.unique(self.groups[:self.num_samples])
        if len(unique_groups) == 0:
            raise ValueError("No valid groups found in the dataset")
        
        # Create mapping if groups are strings
        if unique_groups.dtype.kind in ['U', 'S', 'O']:  # Unicode, String, or Object dtype
            self.group_to_id = {group: idx for idx, group in enumerate(unique_groups)}
            self.groups = np.array([self.group_to_id[g] for g in self.groups])
            logger.info("Converted string groups to numeric IDs")
            unique_groups = np.unique(self.groups[:self.num_samples])
        
        min_group, max_group = unique_groups.min(), unique_groups.max()
        logger.info(f"Group range: {min_group} to {max_group}")
        
        # Safely assign indices to groups
        for idx in valid_indices:
            try:
                group = self.groups[idx]
                if isinstance(group, (int, np.integer)):  # Only process numeric groups
                    self.group_indices[int(group)].append(idx)
            except IndexError as e:
                raise ValueError(
                    f"Index {idx} is out of bounds for groups array of size {len(self.groups)}"
                ) from e
        
        # Validate group indices
        valid_groups = []
        for group in self.group_indices:
            if not self.group_indices[group]:
                logger.warning(f"Group {group} has no valid indices - will be excluded from sampling")
                continue
            valid_groups.append(group)
        
        if not valid_groups:
            raise ValueError("No valid groups found with indices")
        
        # Keep only valid groups
        self.group_indices = {g: self.group_indices[g] for g in valid_groups}
        
        # Verify all indices are within valid range
        all_indices = set()
        for group, indices in self.group_indices.items():
            all_indices.update(indices)
            max_idx = max(indices)
            if max_idx > max_valid_idx:
                raise ValueError(
                    f"Invalid index {max_idx} found in group {group}. "
                    f"Maximum valid index is {max_valid_idx}"
                )
        
        # Verify no missing indices
        expected_indices = set(range(self.num_samples))
        missing_indices = expected_indices - all_indices
        if missing_indices:
            raise ValueError(
                f"Missing indices in groups: {sorted(missing_indices)}. "
                "This may indicate a gap in the dataset."
            )
        
        # Calculate group probabilities with validation
        group_counts = np.array([len(indices) for indices in self.group_indices.values()])
        self.group_counts = group_counts
        self.group_probs = group_counts / group_counts.sum()  # Normalize to sum to 1
        
        # Store valid groups for sampling
        self.valid_groups = np.array(list(self.group_indices.keys()))
        
        # Log detailed group distribution
        logger.info("\nGroup distribution:")
        for group, count in sorted(zip(self.valid_groups, self.group_counts)):
            prob = count / self.num_samples
            indices = self.group_indices[group]
            min_idx = min(indices)
            max_idx = max(indices)
            orig_group = next((k for k, v in self.group_to_id.items() if v == group), group) if hasattr(self, 'group_to_id') else group
            logger.info(
                f"Group {orig_group}: {count} samples ({prob:.2%} of total) "
                f"[index range: {min_idx}-{max_idx}]"
            )
        
        # Calculate number of batches
        self.num_batches = max(1, self.num_samples // self.batch_size)
        self.total_samples = self.num_batches * self.batch_size
        
        logger.info(f"\nWill generate {self.total_samples} samples in {self.num_batches} batches")
        
        if self.batch_size > self.num_samples:
            raise ValueError(
                f"Batch size ({self.batch_size}) cannot be larger than "
                f"dataset size ({self.num_samples})"
            )
    
    def __iter__(self):
        try:
            logger.info("Starting batch index generation...")
            batch_indices = []
            
            # Generate batches using controlled iteration
            for batch_num in range(self.num_batches):
                batch = []
                logger.debug(f"Generating batch {batch_num + 1}/{self.num_batches}")
                
                # Generate fixed-size batch
                for _ in range(self.batch_size):
                    # Select group based on probabilities
                    selected_group = np.random.choice(
                        self.valid_groups,
                        p=self.group_probs
                    )
                    
                    # Get indices for selected group
                    group_indices = self.group_indices[selected_group]
                    
                    # Randomly sample from selected group
                    selected_idx = np.random.choice(group_indices)
                    batch.append(selected_idx)
                
                # Shuffle the batch
                np.random.shuffle(batch)
                batch_indices.extend(batch)
                
                # Log progress
                if (batch_num + 1) % 10 == 0:
                    logger.debug(
                        f"Generated {len(batch_indices)}/{self.total_samples} indices "
                        f"({(len(batch_indices)/self.total_samples)*100:.1f}%)"
                    )
            
            # Validate final indices
            if len(batch_indices) != self.total_samples:
                raise ValueError(f"Generated {len(batch_indices)} indices but expected {self.total_samples}")
            
            # Log final distribution
            final_counts = defaultdict(int)
            for idx in batch_indices:
                group = self.groups[idx]
                final_counts[group] += 1
            
            logger.debug("\nFinal group distribution:")
            for group, count in sorted(final_counts.items()):
                actual_prob = count / len(batch_indices)
                target_prob = self.group_probs[np.where(self.valid_groups == group)[0][0]]
                logger.debug(
                    f"Group {group}: {count} samples "
                    f"(actual: {actual_prob:.2%} vs target: {target_prob:.2%})"
                )
            
            return iter(batch_indices)
            
        except Exception as e:
            logger.error(f"Error in sampler iteration: {str(e)}")
            raise
    
    def __len__(self):
        return self.total_samples 