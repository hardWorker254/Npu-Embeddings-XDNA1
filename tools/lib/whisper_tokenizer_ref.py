# NpuEmbeddings -- pure-Python reference for the C++ Whisper tokenizer.
# SPDX-License-Identifier: Apache-2.0
#
# THIS FILE IS THE EXECUTABLE SPEC for runtime/src/tokenizers/whisper.cpp, in
# the same relationship tools/lib/xlmr_tokenizer_ref.py has to the XLM-R tokenizer.
# It consumes the BLOB out of a packed .npue, not vocab.json/merges.txt, so it
# exercises the exact bytes the C++ reader will see. A divergence between this
# and the C++ is a wrong transcript, not a wrong number: text either matches or
# it does not, so the gate is byte-exact token ids and byte-exact text.
#
# WHY decode IS THE DIRECTION THAT MATTERS: an STT container receives ids from
# the decoder and owes the caller a string. encode exists because the CLI has a
# `tokenize` subcommand and because a transcript oracle needs to score a
# hypothesis, but a bug in encode cannot corrupt a transcription.
#
# PIPELINE (GPT-2 byte-level BPE, as Whisper inherits it):
#   1. UTF-8 bytes -> the 256-symbol printable alphabet (bytes_to_unicode).
#   2. GPT-2 pretokenisation, in this order:
#        's 't 're 've 'm 'll 'd
#        ?\p{L}+          an optional single leading space, then letters
#        ?\p{N}+          an optional single leading space, then digits
#        ?[^\s\p{L}\p{N}]+ an optional single leading space, then the rest
#        \s+(?!\S)        a run of whitespace that is not followed by a
#                         non-space, i.e. all but the last space when a word
#                         follows -- that last space belongs to the word
#   3. BPE within each pretoken: start from the single characters, repeatedly
#      apply the lowest-RANK adjacent pair that is in merges.txt.
#   4. decode: id -> token string -> raw bytes -> UTF-8.
#
# The pretokeniser needs \p{L} and \p{N}, which Python's `re` does not have and
# the `regex` module does. stdlib `re` is not an option here, and silently
# substituting [^\W\d_] for \p{L} would be the kind of approximation that turns
# a non-Latin script into a different token sequence.
#
# Env: numpy + regex. Run tools/verify/verify_whisper_tokenizer.py to hold this against
# HuggingFace's own tokenizer.

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_bpe import TableReader, VocabMerges, bytes_to_unicode  # noqa: E402

# The pre-tokeniser, settled EMPIRICALLY against tokenizers 0.22.2's own
# ByteLevel pre-tokenizer over an adversarial corpus (tools/verify/
# verify_whisper_tokenizer.py holds it there byte-exactly). It is OpenAI's GPT-2
# pattern plus
# two branches HuggingFace added:
#
#   [\r\n]*  after the punctuation run, so a full stop before a newline takes
#            the newline with it
#   |\s+     a final catch-all, so a space that \s+(?!\S) declined (because a
#            word follows and claimed it) is still a pretoken of its own
#
# Two plausible alternatives were measured and REJECTED, which is why they are
# named: tokenizers-rs's published variant (\p{N}{1,3} and a non-letter prefix
# class) disagrees on 9 of 16 cases -- it cuts "1234567890" into four pieces and
# separates the space from "42" -- and the bare OpenAI pattern disagrees on
# newlines. Both change token ids, so both are wrong here.
#
# ORDER OF OPERATIONS, which is the subtle one: the pattern runs on the ORIGINAL
# text, and only each resulting piece is byte-encoded afterwards. Byte-encoding
# first and then splitting is wrong, because the byte-level alphabet is not the
# original alphabet -- 'Ģ' (U+0122) is a letter and '²' (U+00B2) is not, so
# "Привет" comes out as five pieces instead of one and every id after the first
# diverges. This is why the C++ must not be tempted to fold the two steps
# together either.
GPT2_PATTERN = (r"'s|'t|'re|'ve|'m|'ll|'d"
                r"| ?\p{L}+"
                r"| ?\p{N}+"
                r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s+(?!\S)"
                r"|\s+")


def _compile():
    try:
        import regex
    except ImportError:
        raise SystemExit(
            "this reference needs the `regex` module (pip install regex): "
            "Python's re has no \\p{L}. The C++ side uses std::regex, which "
            "does, so the two must be tested against the same pattern.")
    return regex.compile(GPT2_PATTERN)


class WhisperTokenizerRef:
    """encode(text) -> ids, decode(ids) -> text, over a packed table blob."""

    def __init__(self, blob):
        self.t = TableReader(blob)
        self.tokens = [self.t.token(i) for i in range(self.t.n_vocab)]
        self.ids = [int(x) for x in self.t.ids]
        self.index = {s: i for i, s in enumerate(self.tokens)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {ord(v): k for k, v in self.byte_encoder.items()}
        # merges as a rank -> (entry_a, entry_b) table, plus the same thing
        # keyed by token STRING for the inner loop.
        self.ranks = {}
        for rank, (a, b) in enumerate(self.t.merges):
            self.ranks[(self.tokens[int(a)], self.tokens[int(b)])] = rank
        self.by_id = {}
        for i, tid in enumerate(self.ids):
            self.by_id[tid] = i
        self._re = _compile()

    @classmethod
    def from_npz(cls, path, tensor="tokenizer.whisper_table"):
        """Read the blob straight out of a .npue (raw U8 bytes, no tiling)."""
        from npue import Reader
        with Reader(path) as r:
            raw = r.raw(tensor)
            return cls(np.ascontiguousarray(raw, dtype=np.uint8).tobytes())

    @classmethod
    def from_model_dir(cls, model_dir):
        return cls(_table_from_dir(model_dir))

    # -- encode ------------------------------------------------------------

    def pretokenize(self, text):
        """Split the ORIGINAL text. Byte-encoding happens per piece, after."""
        return self._re.findall(text)

    def bpe(self, piece):
        """One pretoken of the ORIGINAL text -> entry indices.

        The piece is byte-encoded here, not before: see the note on
        GPT2_PATTERN. A pretoken that happens to be a single vocab entry is
        checked in its byte-level form, because that is the form the ids are
        keyed by.
        """
        bpe_form = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
        if bpe_form in self.index:
            return [self.index[bpe_form]]
        # Every single byte is in the vocab, so this cannot fail; if it does,
        # the table is corrupt and a wrong token id would be worse than a throw.
        syms = [self.index[self.byte_encoder[b]] for b in piece.encode("utf-8")]
        while len(syms) > 1:
            best_i, best_rank = -1, None
            for i in range(len(syms) - 1):
                r = self.ranks.get((self.tokens[syms[i]], self.tokens[syms[i + 1]]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_i, best_rank = i, r
            if best_i < 0:
                break
            merged = self.tokens[syms[best_i]] + self.tokens[syms[best_i + 1]]
            syms[best_i:best_i + 2] = [self.index[merged]]
        return syms

    def encode(self, text):
        out = []
        for piece in self.pretokenize(text):
            out.extend(self.bpe(piece))
        return [self.ids[i] for i in out]

    # -- decode ------------------------------------------------------------

    def token_bytes(self, entry):
        return bytes(self.byte_decoder[ord(c)] for c in self.tokens[entry])

    def decode(self, ids):
        raw = bytearray()
        for tid in ids:
            raw += self.token_bytes(self.by_id[int(tid)])
        return raw.decode("utf-8", errors="replace")

    def id_of(self, token):
        return self.ids[self.index[token]]


def _table_from_dir(model_dir):
    from whisper_bpe import build_table
    return build_table(VocabMerges.load(model_dir))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Whisper byte-level BPE reference: print token ids and the round trip.")
    ap.add_argument("model_dir", help="a Whisper checkpoint directory")
    ap.add_argument("text", nargs="*", help="text to encode; prints ids and the round trip")
    a = ap.parse_args()
    tk = WhisperTokenizerRef.from_model_dir(a.model_dir)
    print(f"vocab {tk.t.n_vocab}  merges {tk.t.n_merges}")
    for line in (a.text or ["hello world"]):
        ids = tk.encode(line)
        back = tk.decode(ids)
        print(f"{line!r}\n  ids  {ids}\n  back {back!r}  {'OK' if back == line else 'MISMATCH'}")
