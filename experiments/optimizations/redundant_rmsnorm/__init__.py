# Redundant RMSNorm Optimization
#
# This optimization eliminates grid.sync() calls by having all thread blocks
# compute RMSNorm redundantly instead of only block 0.
#
# Files:
#   kernel.cu    - Optimized CUDA kernel
#   benchmark.py - Python bindings and benchmarking
