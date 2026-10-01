# NpuEmbeddings -- the fixed golden corpus.
#
# Four sentences, chosen so the goldens exercise the specific things
# docs/04-model/README.md warns about, rather than being four generic sentences
# that all tokenize down the happy path:
#
#  0. Plain ASCII, short. The baseline; also the shortest, so it produces the
#     most padding and therefore the most masked positions.
#  1. Accented Latin + a hyphen + digits. `strip_accents: null` inherits from
#     `do_lower_case: true`, so accents ARE stripped -- if a future tokenizer
#     gets that backwards, this sentence is where it shows.
#  2. CJK, which the BasicTokenizer splits per codepoint, mixed with ASCII.
#     Note Hiragana/Katakana/Hangul are NOT in the CJK ranges; this uses Han.
#  3. Long, with punctuation and an out-of-vocab word that must decompose into
#     `##` continuation pieces. Longest, so it exercises a nearly-full row.
#
# The batch is deliberately ragged: the whole point is that the attention mask,
# the mean-pool denominator and the padded GEMM rows all get tested.
#
# Frozen. Changing this list invalidates every stored golden -- add a new corpus
# instead.

SENTENCES = [
    "A man is eating food.",
    "Le café coûte 5 euros — c'est bien trop cher pour un espresso.",
    "The 北京 office opens at 9am, 上海 at 10am.",
    "Antidisestablishmentarianism, transubstantiation, and other "
    "unnecessarily long words; do they tokenize correctly?",
]

# A real bucket from docs/04-model ("seq_len -> bucket to {64, 128, 256}").
# Long enough that sentence 3 is not truncated, short enough that a full tap
# dump stays a manageable size.
SEQ_LEN = 64
