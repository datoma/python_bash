from __future__ import division, unicode_literals, print_function, with_statement
from scipy.stats import gmean
import numpy as np
import subprocess
import re


def extract_timings(bench_out, inference=False):
    results = []
    if inference:
        res_line = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)" +
                              r"\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\w+)\s*$")
        pad_line = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)" +
                              r"\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\w+)\s*$")
        for line in bench_out.splitlines():
            print(line)
            match = res_line.match(line)
            if match:
                w, h, c, n, k, f_w, f_h, pad_w, pad_h, stride_w, stride_h, precision, \
                fwd_time, fwd_algo = match.groups()
                results.append(int(fwd_time))
            else:
                match = pad_line.match(line)
                if match:
                    w, h, c, n, k, f_w, f_h, pad_w, pad_h, stride_w, stride_h, precision, \
                    fwd_time, pad_kernels, fwd_algo = match.groups()
                    results.append(int(fwd_time))
    else:
        res_line = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)" +
                              r"\s+(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s*$")
        pad_line = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)" +
                              r"\s+(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s*$")
        for line in bench_out.splitlines():
            print(line)
            match = res_line.match(line)
            if match:
                w, h, c, n, k, f_w, f_h, pad_w, pad_h, stride_w, stride_h, precision, \
                fwd_time, bwd_in_time, bwd_par_time, time, fwd_algo = match.groups()
                results.append(int(time))
            else:
                match = pad_line.match(line)
                if match:
                    w, h, c, n, k, f_w, f_h, pad_w, pad_h, stride_w, stride_h, precision, \
                    fwd_time, bwd_in_time, bwd_par_time, time, pad_kernels, fwd_algo = match.groups()
                    results.append(int(time))
    return results


def run_conv(runs=3):
    results = []
    for usecase in ["train", "inference"]:
        for precision in ["float", "half", "int8"]:
            if usecase == "train" and precision == "int8":
                continue
            run_results = []
            for i in range(runs):
                print("RUNNING: conv_bench {} {}. This is iteration {} of {}".format(usecase, precision, i + 1, runs))
                prc = subprocess.Popen(["../DeepBench/code/bin/conv_bench", usecase, precision], stdout=subprocess.PIPE)
                out = prc.communicate()[0]
                run_results.append(extract_timings(out, inference=(usecase == "inference")))
            results.append("conv_bench {} {}: {}".format(usecase, precision,
                                                         str(gmean(np.array(run_results).min(axis=0)))))
    return results


if __name__ == "__main__":
    runs = 3
    conv_results = run_conv(runs=runs)
    print(".---------------------------------------------------.")
    print("| Benchmark results (geometric mean over all tests) |")
    print("'---------------------------------------------------'")
    print("\n".join(conv_results))
