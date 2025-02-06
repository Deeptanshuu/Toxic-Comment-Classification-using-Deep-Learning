import random
from collections import defaultdict
from torch.utils.data import Sampler

class MultilabelStratifiedSampler(Sampler):
    def __init__(self, labels, batch_size, min_samples_per_lang=4):
        super().__init__(None)
        self.labels = labels
        self.batch_size = batch_size
        self.min_samples_per_lang = min_samples_per_lang
        self.num_languages = 7  # Total number of languages
        
        # Calculate minimum batch size needed to satisfy min_samples_per_lang
        min_batch_size = self.min_samples_per_lang * self.num_languages
        if self.batch_size < min_batch_size:
            raise ValueError(f"Batch size {batch_size} is too small to satisfy minimum {min_samples_per_lang} samples per language. Need at least {min_batch_size}.")
            
    def __iter__(self):
        indices = list(range(len(self.labels)))
        lang_indices = defaultdict(list)
        
        # Group indices by language
        for idx in indices:
            lang = self.labels[idx]['language']
            lang_indices[lang].append(idx)
            
        batches = []
        remaining_indices = indices.copy()
        
        while remaining_indices:
            batch = []
            # First ensure minimum samples per language
            for lang in lang_indices:
                available = [idx for idx in lang_indices[lang] if idx in remaining_indices]
                if available:
                    # Take min_samples_per_lang or all available if less
                    selected = random.sample(available, min(self.min_samples_per_lang, len(available)))
                    batch.extend(selected)
                    for idx in selected:
                        remaining_indices.remove(idx)
                        
            # Fill remaining batch slots randomly
            remaining_slots = self.batch_size - len(batch)
            if remaining_slots > 0 and remaining_indices:
                additional = random.sample(remaining_indices, min(remaining_slots, len(remaining_indices)))
                batch.extend(additional)
                for idx in additional:
                    remaining_indices.remove(idx)
                    
            if batch:  # Only yield non-empty batches
                random.shuffle(batch)  # Shuffle within batch
                batches.append(batch)
                
        random.shuffle(batches)  # Shuffle batch order
        for batch in batches:
            for idx in batch:
                yield idx
                
    def __len__(self):
        return len(self.labels) 