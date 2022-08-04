# Revisiting Time-Aware Item Based Neighbourhood Methods.

## Installing required components
Requirements for the experiments are specified in the `requirements.txt` file, and can be installed using pip.

```
pip install -r requirements.txt
```

The Time-aware ItemKNN models are not yet included in the latest release of [recpack](https://gitlab.com/recpack-maintainers/recpack).
To install these models, you will need to manually install the [feature branch](https://gitlab.com/recpack-maintainers/recpack/-/tree/feature/TARSItemKNN)

## Running an experiment
To run an experiment, use the `run.py` file. This file uses click to specify command line parameters.

```
python run.py --dataset <dataset> [--scenario <scenario> --dataset-path <path/to/datasets/folder> -a <algorithm-1> -a <algorithm-2>]
```

This will run the experiment for the specified dataset, scenario and algorithms. 
If no explicit algorithms are specified, the experiment will run all. 
If no scenario is defined, the `TimedLastItemPrediction` scenario will be used.

Supported values for `<dataset>`, `<scenario>` and `<algorithm>` are given below.

### Supported datasets

* adressa
* recsys2015
* cosmeticsshop

## Supported scenarios

* TimedLastItemPrediction
* Timed

### Supported algorithms

You can find all supported algorithms in the `algorithm_config.py` file in this repository.

### Reproducing the experiments in the paper
In order to reproduce the experiments in the paper, for each dataset run `python3 run.py --dataset <dataset> --scenario TimedLastItemPrediction` and `python3 run.py --dataset <dataset> --scenario Timed`

Note that results for the baseline algorithms (GRU4Rec, Sequential Rules and EASE) are gotten from a different paper: "Are We Forgetting Something? Correctly Evaluate a Recommender System With an Optimal Training Window".
Their experiments can be found in [their own repository](https://github.com/verachtertr/short-intent) 

## Rendering results
Results are stored in the `results/<dataset>` folders. There is a json file with optimisation and evaluation results.
We provide a notebook to generate the tables.
Follow the steps in the `Results.ipynb` notebook to generate the result tables.

