import torch
from transformers import AutoTokenizer

from config import CFG
from dist_utils import dev


PROMPT_TEXT = "Question: Which bird species is in this image?\nAnswer:"
SEPARATOR_TEXT = "\nDescription:"


class TextCodec:
    """Everything on the text side of the model.

    Owns:
        - the tokenizer and its eos / pad ids
        - tokenized class names (+ separator) and tokenized descriptions
        - the fixed question prompt
        - a trie over valid class names (for constrained decoding)
        - the builder that turns labels into padded training targets
    """

    # =========================================================================
    # BUILD
    # =========================================================================

    def __init__(self, class_names, train_classes, train_descs):

        # ---------- Tokenizer and special ids ----------
        self.tok = tok = AutoTokenizer.from_pretrained(CFG["llm"])

        self.eos = tok.eos_token_id
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else self.eos

        # ---------- Class names: " name" tokens + "\nDescription:" separator ----------
        separator = tok(SEPARATOR_TEXT, add_special_tokens=False).input_ids

        self.name_key = [
            tok(" " + name, add_special_tokens=False).input_ids + separator
            for name in class_names
        ]

        # Longest name + a little slack for generation
        self.max_new = max(len(tokens) for tokens in self.name_key) + 2

        # ---------- Descriptions: truncated, then terminated with EOS ----------
        self.desc_tok = []

        for desc in train_descs:
            if desc:
                tokens = tok(" " + desc, add_special_tokens=False).input_ids
                tokens = tokens[:CFG["desc_max_tokens"]] + [self.eos]
            else:
                tokens = [self.eos]

            self.desc_tok.append(tokens)

        # ---------- Fixed question prompt ----------
        self.prompt = tok(
            PROMPT_TEXT,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        # ---------- Trie over training class names (constrained decoding) ----------
        # trie   : nested dicts, each key is a token id
        # lookup : full token sequence -> class id
        self.trie = {}
        self.lookup = {}

        for class_id in train_classes:
            node = self.trie

            for token in self.name_key[class_id]:
                node = node.setdefault(token, {})

            self.lookup[tuple(self.name_key[class_id])] = class_id

    # =========================================================================
    # CONSTRAINED DECODING
    # =========================================================================

    def allowed_fn(self, batch_id, input_ids):
        """Used as `prefix_allowed_tokens_fn` in generate().

        Walks the trie along the tokens generated so far and returns the
        tokens that are allowed next, so the output is always a valid class name.
        """

        node = self.trie

        for token in input_ids.tolist():
            node = node.get(token)

            # Fell off the trie -> force the sequence to end
            if node is None:
                return [self.eos]

        # At a leaf (nothing left to continue) -> end
        return list(node.keys()) or [self.eos]

    # =========================================================================
    # TRAINING TARGETS
    # =========================================================================

    def make_ans(self, y, descs=None):
        """Build padded training targets: name tokens (+ description tokens).

        Args:
            y     : class ids for the batch (tensor, length B)
            descs : optional list of token lists, one description per sample

        Returns (all on the GPU):
            ans       : token ids, padded               (B, L)
            ans_mask  : 1 for real tokens, 0 for pad    (B, L)
            name_mask : True on name tokens, flattened  (B * L,)
            desc_mask : True on description tokens,     (B * L,)
                        flattened
        """

        # ---------- 1) Concatenate name + description for every sample ----------
        sequences, name_lens, desc_lens = [], [], []

        for j, class_id in enumerate(y.tolist()):
            name = self.name_key[class_id]
            desc = descs[j] if descs is not None else []

            sequences.append(name + desc)
            name_lens.append(len(name))
            desc_lens.append(len(desc))

        # ---------- 2) Allocate padded tensors ----------
        batch_size = len(sequences)
        max_len    = max(len(seq) for seq in sequences)

        ans       = torch.full((batch_size, max_len), self.pad, dtype=torch.long)
        ans_mask  = torch.zeros(batch_size, max_len, dtype=torch.long)
        name_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        desc_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        # ---------- 3) Fill them in ----------
        for i, seq in enumerate(sequences):
            name_end = name_lens[i]
            desc_end = name_end + desc_lens[i]

            ans[i, :len(seq)]       = torch.tensor(seq)
            ans_mask[i, :len(seq)]  = 1
            name_mask[i, :name_end] = 1
            desc_mask[i, name_end:desc_end] = 1

        # ---------- 4) Move to GPU; masks flattened to match flattened logits ----------
        return (
            ans.to(dev),
            ans_mask.to(dev),
            name_mask.to(dev).bool().reshape(-1),
            desc_mask.to(dev).bool().reshape(-1),
        )
