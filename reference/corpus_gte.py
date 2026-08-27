# NpuEmbeddings -- 0.5.0 (tasks/0136): golden corpus for the
# gte-multilingual-base check. Reuses reference/corpus.py's four sentences
# exactly like corpus_nomic.py / corpus_gemma.py do, so a future cross-model
# comparison has a shared baseline. The tokenizer-stress reasons carry over
# with a twist: gte's tokenizer is XLM-R SentencePiece Unigram (tasks/0127),
# not WordPiece, so sentence 1's accents are NOT stripped (no lowercasing, no
# NFD here -- the Precompiled charsmap decides) and sentence 2's Han runs
# through Viterbi rather than per-codepoint splitting. Same texts, different
# failure surfaces -- which is the point of keeping them identical.
#
# gte has NO task prompts (the container carries no prompts table), so unlike
# corpus_nomic.py there is no prefix to apply anywhere downstream.
#
# SEQ_LEN matches the project's bucket convention (64). Verified at
# generation time: make_goldens_gte.py refuses if any sentence fills all 64
# slots.

from corpus import SENTENCES  # noqa: F401 -- re-exported, same four sentences

SEQ_LEN = 64
