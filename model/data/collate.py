import torch

class DynamicPadCollator:
    """Pads variable-length samples at collate time.

    Expects each sample to be a dict with unpadded 1-D tensors 'input_ids'
    and 'attention_mask', a float 'labels' tensor of shape (6,) and a scalar
    long 'lang' tensor, as returned by ToxicDataset when dynamic padding is
    enabled. input_ids is padded with the tokenizer pad token and
    attention_mask with 0 up to the batch max length rounded up to a
    multiple of pad_to_multiple_of (tensor-core alignment). If all sequences
    already share one length (static padding), they are stacked unchanged.

    Stores only plain ints, so it pickles cleanly for DataLoader workers.
    """

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = max(1, int(pad_to_multiple_of))

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = torch.stack([f["labels"] for f in features]).float()
        lang = torch.stack([f["lang"] for f in features]).long()

        seq_lens = [ids.size(0) for ids in input_ids]
        max_len = max(seq_lens)

        if min(seq_lens) == max_len:
            # Already uniform (e.g. statically padded): stack as-is
            batch_input_ids = torch.stack(input_ids)
            batch_attention_mask = torch.stack(attention_mask)
        else:
            multiple = self.pad_to_multiple_of
            target_len = ((max_len + multiple - 1) // multiple) * multiple
            batch_input_ids = torch.full(
                (len(features), target_len), self.pad_token_id,
                dtype=input_ids[0].dtype
            )
            batch_attention_mask = torch.zeros(
                (len(features), target_len), dtype=attention_mask[0].dtype
            )
            for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
                batch_input_ids[i, :ids.size(0)] = ids
                batch_attention_mask[i, :mask.size(0)] = mask

        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': labels,
            'lang': lang,
        }
