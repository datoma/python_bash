from __future__ import division, unicode_literals, print_function, with_statement
from scipy.stats import gmean
import numpy as np
import subprocess
import re


def extract_timings(bench_out, inference=False):
    results = []
    if inference:
        res_line = re.compile(r"^\s*(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s*$")
        for line in bench_out.splitlines():
            print(line)
            match = res_line.match(line)
            if match:
                test_type, hidden, n, timesteps, precision, fwd_time = match.groups()
                results.append(int(fwd_time))
    else:
        res_line = re.compile(r"^\s*(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s*$")
        for line in bench_out.splitlines():
            print(line)
            match = res_line.match(line)
            if match:
                test_type, hidden, n, timesteps, precision, fwd_time, bwd_time = match.groups()
                results.append(int(fwd_time)+int(bwd_time))
    return results


def run_rnn(runs=3):
    results = []
    for usecase in ["train", "inference"]:
        for precision in ["float", "half"]:
            run_results = []
            for i in range(runs):
                print("RUNNING: rnn_bench {} {}. This is iteration {} of {}".format(usecase, precision, i+1, runs))
                prc = subprocess.Popen(["../DeepBench/code/bin/rnn_bench", usecase, precision], stdout=subprocess.PIPE)
                out = prc.communicate()[0]
                run_results.append(extract_timings(out, inference=(usecase == "inference")))
            results.append("rnn_bench {} {}: {}".format(usecase, precision, gmean(np.array(run_results).min(axis=0))))
    return results


if __name__ == "__main__":
    print("\n".join(run_rnn()))
