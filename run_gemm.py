from __future__ import division, unicode_literals, print_function, with_statement
from scipy.stats import gmean
import numpy as np
import subprocess
import re


def extract_timings(bench_out):
    results = []
    res_line = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s*$")
    pad_line = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s*$")
    for line in bench_out.splitlines():
        print(line)
        match = res_line.match(line)
        if match:
            m, n, k, a_t, b_t, precision, time = match.groups()
            results.append(int(time))
        else:
            match = pad_line.match(line)
            if match:
                m, n, k, a_t, b_t, precision, time, pad_kernels = match.groups()
                results.append(int(time))
    return results


def run_gemm(runs=3):
    results = []
    for usecase in ["train", "inference"]:
        for precision in ["float", "half", "int8"]:
            if usecase == "train" and precision == "int8":
                continue
            run_results = []
            for i in range(runs):
                print("RUNNING: gemm_bench {} {}. This is iteration {} of {}".format(usecase, precision, i+1, runs))
                prc = subprocess.Popen(["../DeepBench/code/bin/gemm_bench", usecase, precision], stdout=subprocess.PIPE)
                out = prc.communicate()[0]
                run_results.append(extract_timings(out))
            results.append("gemm_bench {} {}: {}".format(usecase, precision, gmean(np.array(run_results).min(axis=0))))
    return results


if __name__ == "__main__":
    runs = 3
    gemm_results = run_gemm(runs=runs)
    print(".---------------------------------------------------.")
    print("| Benchmark results (geometric mean over all tests) |")
    print("'---------------------------------------------------'")
    print("\n".join(gemm_results))
