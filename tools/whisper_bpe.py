# NpuEmbeddings -- the Whisper tokenizer table, built once at pack time.
#
# Whisper's tokenizer is GPT-2 byte-level BPE: vocab.json (string -> id) plus
# merges.txt (rank-ordered pairs). Both are small, both ship with the
# checkpoint, and both are needed to turn decoder output back into text -- which
# is the whole point of an STT container, so a Whisper .npue is not usable
# without them.
#
# This module is the SINGLE definition of the table's binary layout and of the
# byte-level alphabet. The C++ reader (runtime/src/tokenizers/whisper.cpp) and
# the Python oracle (tools/verify_whisper.py) both consume what this produces;
# neither re-derives it. A second copy of the layout is how the three pooling
# implementations and the two npue layout descriptors ended up disagreeing in
# this project, so the layout is documented here in full:
#
#   offset  size  field
#   0       4     magic "WBP1"
#   4       4     version = 1
#   8       4     n_vocab          number of vocab entries
#   12      4     n_merges         number of merge rules, in rank order
#   16      4     pool_bytes       length of the UTF-8 string pool
#   20      4     reserved, zero
#   24      ...   string pool      n_vocab NUL-terminated strings, in entry order
#   then    8*n   entry offsets    u32[n_vocab], byte offset into the pool
#   then    4*n   entry lengths    u32[n_vocab]
#   then    4*n   ids              u32[n_vocab], the token id of each entry
#   then    8*m   merges           u32[2*n_merges], entry indices, rank order
#
# Every integer is little-endian. Entry order is vocab.json's iteration order,
# NOT id order: the ids are carried explicitly because nothing guarantees the
# file is id-sorted, and assuming it would make a reordered vocab.json silently
# produce a different tokenizer.
#
# SPECIAL TOKENS ARE NOT A SEPARATE SECTION. <|startoftranscript|>,
# <|notimestamps|>, <|transcribe|>, <|translate|>, <|endoftext|> and the
# <|lang|> family are ordinary vocab entries; the runtime looks them up by name
# and REFUSES to start a transcription if one is missing, rather than inventing
# an id.
#
# Env: stdlib only (json), plus numpy for the oracle at the bottom.

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"WBP1"
VERSION = 1
HEADER_FORMAT = "<4sIIIII"     # magic, version, n_vocab, n_merges, pool_bytes, reserved
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 24


def bytes_to_unicode():
    """GPT-2's reversible byte <-> printable-codepoint map.

    The 256 byte values are mapped so that all 256 have a distinct printable
    representation, which is what lets BPE operate on text without ever seeing
    a byte. Bytes 33..126 and 161..172 and 174..255 map to themselves; the rest
    are shifted into 256.. and above. This is the standard construction, and it
    is reproduced rather than imported because the tokenizer is not optional to
    this container: a wrong table yields confidently wrong text.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def unicode_to_bytes():
    inv = {v: k for k, v in bytes_to_unicode().items()}
    # chr(256+n) is not a single byte; invert through ord().
    return {ord(v): k for k, v in inv.items()}


class VocabMerges:
    """vocab.json + merges.txt, with the byte-level alphabet attached."""

    def __init__(self, tokens, ids, merges):
        self.tokens = tokens          # entry order
        self.ids = ids                # parallel to tokens
        self.merges = merges          # [(entry_a, entry_b)], rank order
        self.index = {t: i for i, t in enumerate(tokens)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {ord(v): k for k, v in self.byte_encoder.items()}
        self.ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def load(cls, model_dir):
        """vocab.json + added_tokens.json + merges.txt.

        TWO FILES, NOT ONE, and this is the first thing that bites: Whisper's
        vocab.json is GPT-2's 50258 entries and nothing else. Every token that
        makes Whisper Whisper -- <|startoftranscript|>, <|notimestamps|>,
        <|transcribe|>, <|translate|>, <|endoftext|> and the 1600-odd timestamp
        and language tokens -- lives in added_tokens.json, with ids from 50258
        to 51865. Reading vocab.json alone yields a container that transcribes
        nothing and cannot even name its own control tokens.

        The two id ranges are disjoint in every shipped checkpoint, and that is
        CHECKED rather than assumed: a silent overlap would mean one token string
        had two ids, and the decode direction would pick whichever came last.
        """
        d = Path(model_dir)
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        added = {}
        added_path = d / "added_tokens.json"
        if added_path.exists():
            raw = json.loads(added_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                added = {k: int(v) for k, v in raw.items()}
            else:                       # the list-of-objects form
                added = {e["content"]: int(e["id"]) for e in raw}

        pairs = []
        seen_ids = {}
        for tok, tid in list(vocab.items()) + list(added.items()):
            tid = int(tid)
            if tid in seen_ids and seen_ids[tid] != tok:
                raise SystemExit(
                    f"{d}: id {tid} is claimed by both {seen_ids[tid]!r} and "
                    f"{tok!r}; the tokenizer cannot be represented")
            seen_ids[tid] = tok
            pairs.append((tid, tok))
        pairs.sort()
        ids = [t for t, _ in pairs]
        tokens = [s for _, s in pairs]
        expected = list(range(len(pairs)))
        if ids != expected:
            raise SystemExit(
                f"{d}: token ids are not 0..{len(pairs) - 1} with no gaps "
                f"(first gap at index {next(i for i, t in enumerate(ids) if t != i)}). "
                f"The table format stores ids explicitly, but the runtime's "
                f"logit indexing assumes a dense range, so this is refused here "
                f"rather than mis-decoded later.")

        merges = []
        for line in (d / "merges.txt").read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            a, b = line.split(" ")
            merges.append((a, b))
        return cls(tokens, ids, merges)

    def id_of(self, token):
        return self.ids[self.index[token]]

    def token_of(self, token_id):
        for i, tid in enumerate(self.ids):
            if tid == token_id:
                return self.tokens[i]
        raise KeyError(token_id)

    def to_bytes(self, token):
        """A token string (byte-level alphabet) back to raw bytes."""
        return bytes(self.byte_decoder[ord(ch)] for ch in token)

    def from_bytes(self, raw):
        return "".join(self.byte_encoder[b] for b in raw)


def build_table(vm):
    """The binary blob described in this module's header comment."""
    pool = bytearray()
    offs, lens = [], []
    for t in vm.tokens:
        offs.append(len(pool))
        b = t.encode("utf-8")
        pool += b + b"\0"
        lens.append(len(b))
    index = vm.index
    pairs = []
    for a, b in vm.merges:
        if a not in index or b not in index:
            # A merge naming an absent token can never fire; dropping it is
            # exact rather than lossy, and a missing ENTRY is a real problem
            # worth failing on.
            if a not in index and b not in index:
                raise SystemExit(
                    f"merges.txt names tokens absent from vocab.json: {a!r} {b!r}")
            continue
        pairs.append((index[a], index[b]))

    out = bytearray()
    out += struct.pack(HEADER_FORMAT, MAGIC, VERSION, len(vm.tokens),
                       len(pairs), len(pool), 0)
    out += pool
    # Two SEPARATE arrays, not interleaved (off,len) pairs: the C++ reader
    # memcpys each one straight out of the mapping, and a reader that has to
    # stride by two u32s to get the lengths is a reader that will be written
    # wrong once.
    for o in offs:
        out += struct.pack("<I", o)
    for l in lens:
        out += struct.pack("<I", l)
    for tid in vm.ids:
        out += struct.pack("<I", tid)
    for a, b in pairs:
        out += struct.pack("<II", a, b)
    return bytes(out)


class TableReader:
    """Reference reader, so the C++ side has something to be checked against."""

    def __init__(self, blob):
        (magic, version, n_vocab, n_merges, pool_bytes, _res) = struct.unpack_from(
            HEADER_FORMAT, blob, 0)
        if magic != MAGIC:
            raise ValueError(f"not a whisper tokenizer table: {magic!r}")
        if version != VERSION:
            raise ValueError(f"table version {version}, expected {VERSION}")
        self.n_vocab, self.n_merges = n_vocab, n_merges
        pos = HEADER_SIZE
        self.pool = blob[pos:pos + pool_bytes]
        pos += pool_bytes
        self.offs = np.frombuffer(blob, dtype="<u4", count=n_vocab, offset=pos)
        pos += 4 * n_vocab
        self.lens = np.frombuffer(blob, dtype="<u4", count=n_vocab, offset=pos)
        pos += 4 * n_vocab
        self.ids = np.frombuffer(blob, dtype="<u4", count=n_vocab, offset=pos)
        pos += 4 * n_vocab
        m = np.frombuffer(blob, dtype="<u4", count=2 * n_merges, offset=pos)
        self.merges = m.reshape(n_merges, 2)
        self.by_id = {int(t): i for i, t in enumerate(self.ids)}

    def token(self, i):
        o, l = int(self.offs[i]), int(self.lens[i])
        return self.pool[o:o + l].decode("utf-8")
