# DeepBenchCollector
cloned from https://github.com/pinae/DeepBenchCollector
Run Baidu's DeepBench and collect the results.

This Python script runs the GEMM, RNN and Conv-Benchmark from
`../DeepBench/` and calculates the geometric Mean of the results.

Run all benchmarks with:

```
python benchmark.py
```

You may also run one or all of the three with these commands:

```
python run_gemm.py
python run_rnn.py
python run_conv.py
```
