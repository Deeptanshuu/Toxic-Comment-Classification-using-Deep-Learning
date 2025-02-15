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
        
        # Verify all groups have at least one sample
        empty_groups = [g for g, indices in self.group_indices.items() if not indices]
        if empty_groups:
            logger.warning(f"Found empty groups: {empty_groups}")
        
        # Verify all indices are within valid range
        all_indices = set()
        for group, indices in self.group_indices.items():
            if not indices:  # Skip empty groups
                continue
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
        
        # Validate total samples match
        total_group_samples = sum(len(indices) for indices in self.group_indices.values())
        if total_group_samples != self.num_samples:
            raise ValueError(
                f"Total samples in groups ({total_group_samples}) "
                f"does not match dataset size ({self.num_samples})"
            )
        
        # Calculate group probabilities with validation
        group_counts = np.array([len(indices) for indices in self.group_indices.values()])
        if not group_counts.any():
            raise ValueError("No valid samples found in any group")
        
        self.group_counts = group_counts
        self.group_probs = group_counts / group_counts.sum()  # Normalize to sum to 1
        
        # Log detailed group distribution
        logger.info("\nGroup distribution:")
        for group, count in sorted(zip(self.group_indices.keys(), self.group_counts)):
            prob = count / self.num_samples
            indices = self.group_indices[group]
            min_idx = min(indices) if indices else -1
            max_idx = max(indices) if indices else -1
            orig_group = next((k for k, v in self.group_to_id.items() if v == group), group) if hasattr(self, 'group_to_id') else group
            logger.info(
                f"Group {orig_group}: {count} samples ({prob:.2%} of total) "
                f"[index range: {min_idx}-{max_idx}]"
            )
        
        # Calculate number of batches to maintain approximately same dataset size
        self.num_batches = max(1, self.num_samples // self.batch_size)
        self.total_samples = self.num_batches * self.batch_size
        
        logger.info(f"\nWill generate {self.total_samples} samples in {self.num_batches} batches")
        
        # Validate batch size
        if self.batch_size > self.num_samples:
            raise ValueError(
                f"Batch size ({self.batch_size}) cannot be larger than "
                f"dataset size ({self.num_samples})"
            )
    
    def __iter__(self):
        try:
            logger.info("Starting batch index generation...")
            batch_indices = []
            max_attempts = 3 * self.total_samples  # Safety limit
            attempts = 0
            
            # Generate batches using dynamic ratio-based sampling
            for batch_num in range(self.num_batches):
                batch = []
                batch_attempts = 0
                max_batch_attempts = 3 * self.batch_size
                
                print(f"Generating batch {batch_num + 1}/{self.num_batches}")
                
                while len(batch) < self.batch_size:
                    batch_attempts += 1
                    if batch_attempts > max_batch_attempts:
                        logger.error(f"Failed to generate batch after {max_batch_attempts} attempts")
                        raise RuntimeError("Failed to generate balanced batch")
                    
                    # Select group based on probabilities
                    selected_group = np.random.choice(
                        len(self.group_indices),
                        p=self.group_probs
                    )
                    
                    # Get indices for selected group
                    group_indices = self.group_indices[selected_group]
                    if not group_indices:  # Skip empty groups
                        continue
                    
                    # Randomly sample from selected group
                    selected_idx = np.random.choice(group_indices)
                    
                    # Validate index
                    if selected_idx >= self.num_samples:
                        logger.warning(f"Invalid index {selected_idx} generated, skipping")
                        continue
                    
                    batch.append(selected_idx)
                
                # Shuffle the batch
                np.random.shuffle(batch)
                batch_indices.extend(batch)
                
                # Print progress
                print(f"Generated {len(batch_indices)} of {self.total_samples} indices")
                if len(batch_indices) % 1000 == 0:
                    print(f"Progress: {len(batch_indices)/self.total_samples:.1%}")
            
            # Final validation
            if len(batch_indices) != self.total_samples:
                raise ValueError(f"Generated {len(batch_indices)} indices but expected {self.total_samples}")
            if np.any(np.array(batch_indices) >= self.num_samples):
                raise ValueError(f"Invalid indices generated: max index {max(batch_indices)} >= dataset size {self.num_samples}")
            
            # Log distribution of groups in final indices
            group_counts = defaultdict(int)
            for idx in batch_indices:
                group_counts[self.groups[idx]] += 1
            
            for group, count in sorted(group_counts.items()):
                actual_prob = count / len(batch_indices)
                target_prob = self.group_probs[group]
                logger.debug(
                    f"Group {group}: {count} samples ({actual_prob:.2%} vs target {target_prob:.2%})"
                )
            
            return iter(batch_indices)
            
        except Exception as e:
            logger.error(f"Error in sampler iteration: {str(e)}")
            raise
    
    def __len__(self):
        return self.total_samples 