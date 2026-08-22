# NpuEmbeddings -- M13 (tasks/0069): golden corpus for the nomic-embed-text-v1.5
# check. Reuses reference/corpus.py's four sentences (same tokenizer-stress
# reasons: ASCII, accented Latin, CJK, and a long OOV-heavy sentence) exactly
# like corpus_gemma.py does, so a future cross-model comparison has a shared
# baseline. nomic REQUIRES a task prefix (see encoder_nomic.PROMPTS) prepended
# before tokenization -- the prefix is not part of this file, it is applied by
# make_goldens_nomic.py/check_reference_nomic.py, same split as corpus_gemma.py
# vs encoder_gemma.PROMPTS.
#
# SEQ_LEN matches the rest of this project's bucket convention (64). nomic's
# tokenizer is the same WordPiece vocab.txt as MiniLM/bge (tasks/0068:
# byte-identical sha256 across all five models), so the corpus's untprefixed
# token counts are already known to fit comfortably in 64; the 2-4 extra
# prefix tokens (tasks/0068 TASK.md) leave headroom to spare.

from corpus import SENTENCES  # noqa: F401 -- re-exported, same four sentences

SEQ_LEN = 64
