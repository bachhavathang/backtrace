"""Shared test doubles. No network."""
import zlib

import numpy as np
import torch


class FakeModel:
    """Embeds by hashing words into 512 dims: deterministic, instant, offline.

    crc32, not hash(): Python salts str hashes per process, which made word
    collisions — and so these tests — random.
    """
    def encode(self, texts, convert_to_tensor=True, normalize_embeddings=True, **_):
        single = isinstance(texts, str)
        rows = []
        for t in ([texts] if single else texts):
            v = np.zeros(512, dtype=np.float32)
            for w in t.lower().split():
                v[zlib.crc32(w.encode()) % 512] += 1
            rows.append(v / (np.linalg.norm(v) or 1))
        out = torch.from_numpy(np.stack(rows))
        return out[0] if single else out
