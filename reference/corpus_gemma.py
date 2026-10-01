# NpuEmbeddings -- C1 spike: golden corpus for the EmbeddingGemma-300M check.
#
# Reuses reference/corpus.py's four sentences (same tokenizer-stress reasons:
# ASCII, accented Latin, CJK, and a long OOV-heavy sentence) so a future
# cross-model comparison has a shared baseline. EmbeddingGemma additionally
# requires a TASK PREFIX (see encoder_gemma.PROMPTS) -- unlike WordPiece BERT
# models, the prefix is part of the model's trained contract, not an add-on.
#
# SEQ_LEN matches MiniLM's bucket (64): tokenized with the "document" prefix
# (BOS + prefix + sentence + EOS), the corpus is 14/25/24/30 tokens -- 64
# leaves comfortable padding room without changing the bucket convention.
# (Checked empirically, tasks/0055 TASK.md.)

from corpus import SENTENCES  # noqa: F401 -- re-exported, same four sentences

SEQ_LEN = 64
