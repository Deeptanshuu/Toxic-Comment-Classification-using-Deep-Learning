from torch.utils.data import Sampler
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MultilabelStratifiedSampler(Sampler):
    """Language-stratified sampler that yields every index exactly once per epoch.

    Each epoch the indices of every language group are shuffled, then batches
    are filled by giving each group its proportional share of the batch
    (largest remainder method), consuming each group's stream front to back.
    One epoch is therefore a full pass over the data with the per-batch
    language mix within one sample of the global mix; the last batch may be
    short so len(sampler) always equals the number of indices yielded.

    When ``lengths`` is provided, batches additionally group samples of
    similar length: streams are cut into megabatches of
    ``megabatch_factor * batch_size`` (proportional per language), each
    language's members are sorted by length within the megabatch, batches are
    formed from aligned length slices, and the order of the resulting full
    batches is shuffled. Language stratification is kept exact; length
    homogeneity is only as good as the agreement between the per-language
    length distributions.

    Seeding follows the torch DistributedSampler convention: the ordering is
    a pure function of ``seed + epoch``. Call ``set_epoch(epoch)`` at the
    start of each epoch to get a different, reproducible shuffle.
    """

    def __init__(self, labels, groups, batch_size, cached_size=None, *,
                 seed=0, lengths=None, megabatch_factor=20):
        super().__init__(None)
        self.labels = np.array(labels)
        self.groups = np.array(groups)
        self.batch_size = batch_size
        self.num_samples = len(self.labels)
        self.seed = seed
        self.epoch = 0

        # Simple validation
        if len(self.labels) != len(self.groups):
            raise ValueError("Length mismatch between labels and groups")

        self.lengths = None
        if lengths is not None:
            self.lengths = np.asarray(lengths)
            if len(self.lengths) != self.num_samples:
                raise ValueError("Length mismatch between lengths and labels")

        self.megabatch_factor = int(megabatch_factor)
        if self.megabatch_factor < 1:
            raise ValueError("megabatch_factor must be >= 1")

        # Create indices per group
        self.group_indices = {}
        unique_groups = np.unique(self.groups)

        for group in unique_groups:
            indices = np.where(self.groups == group)[0]
            if len(indices) > 0:
                self.group_indices[group] = indices

        # Global group proportions
        group_sizes = np.array([len(indices) for indices in self.group_indices.values()])
        self.group_probs = group_sizes / group_sizes.sum()
        self.valid_groups = list(self.group_indices.keys())

        # Every sample is yielded once per epoch; the last batch may be short
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.total_samples = self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _proportional_take(self, remaining, size, rng):
        # Largest remainder method: each group gets floor or ceil of its
        # proportional share of the remaining samples
        ideal = remaining * (size / remaining.sum())
        take = np.floor(ideal).astype(np.int64)
        shortfall = size - int(take.sum())
        if shortfall > 0:
            frac = np.where(remaining > take, ideal - take, -1.0)
            order = np.lexsort((rng.random(len(frac)), -frac))
            take[order[:shortfall]] += 1
        return take

    def _quota_chunks(self, streams, chunk_size, rng, shuffle_within=True):
        # Cut per-group streams into chunks of chunk_size, each chunk taking a
        # proportional share of every group. All chunks are full except the
        # last one, which holds the leftover tail.
        pointers = np.zeros(len(streams), dtype=np.int64)
        remaining = np.array([len(s) for s in streams], dtype=np.int64)
        chunks = []
        while remaining.sum() > 0:
            size = min(chunk_size, int(remaining.sum()))
            take = self._proportional_take(remaining, size, rng)
            chunk = np.concatenate([
                streams[gi][pointers[gi]:pointers[gi] + t]
                for gi, t in enumerate(take) if t > 0
            ])
            pointers += take
            remaining -= take
            if shuffle_within:
                rng.shuffle(chunk)
            chunks.append(chunk)
        return chunks

    def _length_grouped_batches(self, streams, rng):
        mega_size = self.megabatch_factor * self.batch_size
        batches = []
        for mega in self._quota_chunks(streams, mega_size, rng, shuffle_within=False):
            mega_groups = self.groups[mega]
            # Sort each language's members by length; stream order was
            # shuffled, so the stable sort breaks length ties randomly
            sub_streams = []
            for group in self.valid_groups:
                sub = mega[mega_groups == group]
                if len(sub) > 0:
                    sub = sub[np.argsort(self.lengths[sub], kind="stable")]
                    sub_streams.append(sub)
            # Proportional quotas advance every language through its own
            # length-sorted stream in step, so each batch is an aligned
            # length slice with the language mix preserved
            batches.extend(self._quota_chunks(sub_streams, self.batch_size, rng))
        # Shuffle batch order so training time is not correlated with length.
        # Any short batch must stay last: the DataLoader re-chops the flat
        # index stream every batch_size samples.
        full = [b for b in batches if len(b) == self.batch_size]
        tail = [b for b in batches if len(b) != self.batch_size]
        order = rng.permutation(len(full))
        return [full[i] for i in order] + tail

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        streams = [rng.permutation(self.group_indices[g]) for g in self.valid_groups]

        if self.lengths is None:
            batches = self._quota_chunks(streams, self.batch_size, rng)
        else:
            batches = self._length_grouped_batches(streams, rng)

        if not batches:
            return iter([])
        return iter(np.concatenate(batches).tolist())

    def __len__(self):
        return self.num_samples
