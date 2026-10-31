import torch
from transformers import AutoTokenizer
from config import CFG
from dist_utils import dev

class TextCodec:
    """Tokenizer + prompt + trie for constrained decoding + training-target builder."""
    def __init__(self, class_names, train_classes, train_descs):
        self.tok = tok = AutoTokenizer.from_pretrained(CFG["llm"])
        self.eos = tok.eos_token_id
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else self.eos
        sep = tok("\nDescription:", add_special_tokens=False).input_ids
        self.name_key = [tok(" " + n, add_special_tokens=False).input_ids + sep for n in class_names]
        self.max_new = max(len(a) for a in self.name_key) + 2
        self.desc_tok = [(tok(" " + d, add_special_tokens=False).input_ids[:CFG["desc_max_tokens"]] + [self.eos]) if d else [self.eos]
                         for d in train_descs]
        self.prompt = tok("Question: Which bird species is in this image?\nAnswer:",
                          return_tensors="pt", add_special_tokens=False).input_ids
        self.trie, self.lookup = {}, {}
        
        for c in train_classes:                    # output is always a valid class name
            node = self.trie
            for t in self.name_key[c]: node = node.setdefault(t, {})
            self.lookup[tuple(self.name_key[c])] = c

    def allowed_fn(self, batch_id, input_ids):
        node = self.trie
        for t in input_ids.tolist():
            node = node.get(t)
            if node is None: return [self.eos]
        return list(node.keys()) or [self.eos]

    def make_ans(self, y, descs=None):
        """target = name tokens (+ description tokens). Returns ids, attn mask, name mask, desc mask."""
        seqs, nl, dl = [], [], []
        for j, c in enumerate(y.tolist()):
            n = self.name_key[c]; d = descs[j] if descs is not None else []
            seqs.append(n + d); nl.append(len(n)); dl.append(len(d))
        L = max(map(len, seqs)); Bn = len(seqs)
        a = torch.full((Bn, L), self.pad, dtype=torch.long)
        m, nm, dm = (torch.zeros(Bn, L, dtype=torch.long) for _ in range(3))
        for i, s in enumerate(seqs):
            a[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1; nm[i, :nl[i]] = 1; dm[i, nl[i]:nl[i] + dl[i]] = 1
        return a.to(dev), m.to(dev), nm.to(dev).bool().reshape(-1), dm.to(dev).bool().reshape(-1)
