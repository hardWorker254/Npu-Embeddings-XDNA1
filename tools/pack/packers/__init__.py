"""Whisper packing: an openai/whisper-* checkpoint into an arch=4 .npue.

Split out of tools/pack/pack_npue.py rather than appended to it. That file is already
1600 lines and carries four other architectures; a fifth, with a conv frontend,
a positional-embedding table, a second (decoder) stack and a tokenizer table,
belongs in its own module. pack_npue.py keeps the dispatch and calls in here.
"""
