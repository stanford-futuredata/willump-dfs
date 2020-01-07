# willump-dfs
This repository enables replication of the Purchase benchmark results in Figures 6 and 8 
of the [Willump](https://arxiv.org/pdf/1906.01974.pdf) paper.  Willump's Purchase benchmark is adapted from
the predict-next-purchase Featuretools benchmark, located [here](https://github.com/Featuretools/predict-next-purchase).

This benchmark requires Python 3 and was tested with Python 3.6.8.

First, install the [Featuretools library](https://www.featuretools.com/) and add the willump-dfs root folder to your PYTHONPATH.

Then download the Purchase dataset from [here]( https://willump-datasets.s3.us-east-2.amazonaws.com/data_huge.zip)
and unzip into the tests/test_resources/predict_next_purchase_resources/data_huge folder.  This dataset
has been processed from the [Instacart dataset](https://www.instacart.com/datasets/grocery-shopping-2017) by
scripts in the original [predict-next-purchase benchmark](https://github.com/Featuretools/predict-next-purchase).

To replicate results in Figure 6, run:

    python3 tests/benchmark_scripts/purchase_train.py -d huge
    python3 tests/benchmark_scripts/purchase_batch.py -d huge
    python3 tests/benchmark_scripts/purchase_batch.py -d huge -c
    
The first command trains a model (note that this takes around three hours), the second executes it natively, the third optimizes it with Willump's cascades
optimization.  The throughput reported by the third command should be significantly higher
than that reported by the second.

To replica results in Figure 8, run:

    python3 tests/benchmark_scripts/purchase_train.py -d huge -k 100
    python3 tests/benchmark_scripts/purchase_batch.py -d huge -k 100
    python3 tests/benchmark_scripts/purchase_batch.py -d huge -k 100 -c
    
The first command trains a model (note that this takes around three hours), the second executes it natively, the third optimizes it with Willump's
top-k approximation optimization. The throughput reported by the third command should be
significantly higher than that reported by the second.