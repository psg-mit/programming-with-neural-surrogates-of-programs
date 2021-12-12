# Programming with Neural Surrogates of Programs

## Citation

```
@inproceedings{renda2021programming, 
  title={Programming with Neural Surrogates of Programs}, 
  author={Renda, Alex and Ding, Yi and Carbin, Michael}, 
  booktitle={ACM SIGPLAN International Symposium on New Ideas, New Paradigms, and Reflections on Programming and Software (Onward!)}, 
  doi = {10.1145/3486607.3486748}, 
  year={2021}, 
}
```

## Dependencies

The codebase has been validated on Debian 10 (Buster) with the following dependencies/versions:

- python=3.7
  - datasets=1.16.1
  - matplotlib=3.5.0
  - numpy=1.21.2
  - onnxruntime=1.10.0
  - pandas=1.3.4
  - pytorch=1.10.0
  - tokenizers=0.10.3
  - tqdm=4.62.3
  - transformers=4.13.0
- numactl
- cmake
- git
- ninja-build
- build-essential

You should also have cuda installed and pytorch configured to use cuda to train efficiently (though it is not required).

## How to use

### Hyperparameter Search / Surrogate Compilation

To run the initial hyperparameter search, using `[JOBS]` parallel jobs:
```python train.py --jobs [JOBS] hyperparameter-search```

To print the results of the hyperparameter search (corresponding to Table 4 in the paper), run:

```python results.py table-4```

To print the results of surrogate compilation (corresponding to Section 3in the paper), first build llvm-mca:

```./build_llvm.sh```

Them print the results:

```python results.py section-3```

To generate the telemetry figures in Appendix B, run:

```python results.py surrogate-compilation-telemetry```

which will create plots in the `figures` directory.

### Surrogate Adaptation

To run the surrpgate adaptation experiments, using `[JOBS]` parallel jobs:
```python train.py --jobs [JOBS] adaptation```

To generate Figure 2 in the paper:

```python results.py figure-2```

To generatee the telemetry figures in Appendix B, run:

```python results.py surrogate-adaptation-telemetry```
