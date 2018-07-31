from __future__ import division, unicode_literals, print_function, with_statement
from run_gemm import run_gemm
from run_rnn import run_rnn
from run_conv import run_conv

if __name__ == "__main__":
    runs = 3
    gemm_results = run_gemm(runs=runs)
    rnn_results = run_rnn(runs=runs)
    conv_results = run_conv(runs=runs)
    print(".---------------------------------------------------.")
    print("| Benchmark results (geometric mean over all tests) |")
    print("'---------------------------------------------------'")
    print("\n".join(gemm_results+rnn_results+conv_results))
