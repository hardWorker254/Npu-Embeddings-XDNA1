"""Exporter package.

tools/export/export_gemm_rtp.py and tools/export/export_eltwise.py are the historic entry
points and remain working shims; the code lives here, split so that the parts
shared by both (targets file, cache identity, output layout, validation) have
one home instead of one copy per script.
"""
