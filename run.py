import click
import datetime

from recpack.algorithms import TARSItemKNN
from recpack.datasets import AdressaOneWeek, CosmeticsShop, RecsysChallenge2015
from recpack.pipelines import PipelineBuilder
from recpack.scenarios import TimedLastItemPrediction
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem


DATASET_PATH = "/home/robinverachtert/datasets/"


def recsys_dataset(dataset_path):
    ds = RecsysChallenge2015(path=dataset_path, filename="yoochoose-clicks.dat", preprocess_default=False)
    ds.add_filter(MinUsersPerItem(50, item_ix=ds.ITEM_IX, user_ix=ds.USER_IX))
    ds.add_filter(MinItemsPerUser(3, item_ix=ds.ITEM_IX, user_ix=ds.USER_IX))

    return ds


def get_datasets_info(dataset_path, dataset):
    datasets = {
        "adressa": {
            "dataset": AdressaOneWeek(path=dataset_path),
            "t": int(datetime.datetime(2017, 1, 7, 12).strftime("%s")),
            "t_val": int(datetime.datetime(2017, 1, 6, 12).strftime("%s")),
            "delta_out": 12 * 3600,
        },
        "cosmeticsshop": {
            "dataset": CosmeticsShop(path=dataset_path, filename="cosmeticsshop_views.csv"),
            "t": int(datetime.datetime(2020, 2, 15, 0).strftime("%s")),
            "t_val": int(datetime.datetime(2020, 2, 1, 0).strftime("%s")),
            "delta_out": 14 * 24 * 3600,
        },
        "recsys2015": {
            "dataset": recsys_dataset(dataset_path),
            "t": int(datetime.datetime(2014, 9, 15, 0).strftime("%s")),
            "t_val": int(datetime.datetime(2014, 9, 1, 0).strftime("%s")),
            "delta_out": 14 * 24 * 3600,
        },
    }
    return datasets[dataset]


HOUR = 3600  # seconds
DAY = 24 * HOUR


ALGORITHM_CONFIG = {
    # Literature algorithms
    "TARSItemKNNLee_W3": {
        "grid": {"similarity": ["cosine", "pearson"]},
    },
    "TARSItemKNNLee_W5": {
        "grid": {"similarity": ["cosine", "pearson"]},
    },
    "TARSItemKNNLiu": {
        "grid": {
            "fit_decay": [0]
            + [1 / x for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY]],
            "predict_decay": [0]
            + [1 / x for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY]],
        },
    },
    "TARSItemKNNDing": {
        "grid": {
            "predict_decay": [0]
            + [1 / x for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY]],
            "similarity": ["cosine", "conditional_probability"],
        },
    },
    "TARSItemKNNLiu2012": {
        "grid": {
            "decay": [2, 5, 10, 50, 100, 200, 500, 1000],
        },
    },
    "TARSItemKNNVaz": {
        "grid": {
            "fit_decay": [
                1 / x
                for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY, 30 * DAY]
            ],
            "predict_decay": [
                1 / x
                for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY, 30 * DAY]
            ],
        }
    },
    # Extensions
    "TARSItemKNNLee": {
        "grid": {"similarity": ["cosine", "pearson"], "W": [2, 3, 4, 5, 8, 10, 16]},
    },
    "TARSItemKNNexponential": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": TARSItemKNN.SUPPORTED_SIMILARITIES,
            "fit_decay": [
                1 / x
                for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY, 30 * DAY]
            ],
            "predict_decay": [
                1 / x
                for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY, 30 * DAY]
            ],
        },
        "params": {"decay_function": "exponential"},
    },
    "TARSItemKNNlog": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": TARSItemKNN.SUPPORTED_SIMILARITIES,
            "fit_decay": [2, 4, 8, 16, 32],
            "predict_decay": [2, 4, 8, 16, 32],
        },
        "params": {"decay_function": "log"},
    },
    "TARSItemKNNlinear": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": TARSItemKNN.SUPPORTED_SIMILARITIES,
            "fit_decay": [0.1, 0.3, 0.5, 0.7, 0.9, 1],
            "predict_decay": [0.1, 0.3, 0.5, 0.7, 0.9, 1],
        },
        "params": {"decay_function": "linear"},
    },
    "TARSItemKNNconcave": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": TARSItemKNN.SUPPORTED_SIMILARITIES,
            "fit_decay": [0.01, 0.1, 0.3, 0.9],
            "predict_decay": [0.01, 0.1, 0.3, 0.9],
        },
        "params": {"decay_function": "concave"},
    },
    "TARSItemKNNconvex": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": TARSItemKNN.SUPPORTED_SIMILARITIES,
            "fit_decay": [0.01, 0.1, 0.3, 0.9],
            "predict_decay": [0.01, 0.1, 0.3, 0.9],
        },
        "params": {"decay_function": "convex"},
    },
    "TARSItemKNNlinear_steeper": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": TARSItemKNN.SUPPORTED_SIMILARITIES,
            "fit_decay": [1, 5, 10, 50, 100, 1000],
            "predict_decay": [1, 5, 10, 50, 100, 1000],
        },
        "params": {"decay_function": "linear_steeper"},
    },
}


@click.command()
@click.option("--dataset", help="Dataset to use for running the experiment.")
@click.option("--dataset-path", help="path to the dataset files", default=DATASET_PATH)
@click.option(
    "--algorithm",
    "-a",
    help="The algorithm to run, specify multiple times to run multiple. Defaults to running all",
    multiple=True,
    default=list(ALGORITHM_CONFIG.keys()),
)
@click.option("--results-path", help="path to put results, defaults to results", default="results")
@click.option(
    "--experiment-name",
    "-en",
    help="name of the experiment, will define the folder written inside results-path. Defaults to {dataset}",
    default=None,
)
def run(dataset, dataset_path, algorithm, results_path, experiment_name):
    print(f"running {', '.join(algorithm)} on {dataset}")

    experiment_name = experiment_name if experiment_name else dataset

    for a in algorithm:
        if a not in ALGORITHM_CONFIG:
            raise ValueError(f"{a} not supported in experiment, please use one of the preconfigured algorithms.")

    print(">> Loading dataset")
    im = get_datasets_info(dataset_path, dataset)["dataset"].load()
    print("<< Loaded dataset")
    print(f"dataset shape = {im.shape}")

    t = get_datasets_info(dataset_path, dataset)["t"]
    t_val = get_datasets_info(dataset_path, dataset)["t_val"]
    delta_out = get_datasets_info(dataset_path, dataset)["delta_out"]

    print(">> Splitting dataset")
    scenario = TimedLastItemPrediction(t=t, t_validation=t_val, validation=True, delta_out=delta_out)
    scenario.split(im)
    print("<< Split dataset")

    builder = PipelineBuilder(experiment_name, base_path=results_path)
    builder.set_data_from_scenario(scenario)
    builder.set_optimisation_metric("NDCGK", 10)
    builder.add_metric("NDCGK", [10, 20, 50])
    builder.add_metric("CoverageK", [10, 20])
    builder.add_metric("CalibratedRecallK", K=[10, 20, 50])
    builder.add_metric("ReciprocalRankK", K=[10, 20, 50])
    builder.add_metric("PrecisionK", K=[10, 20, 50])

    for a in algorithm:
        conf = ALGORITHM_CONFIG[a]
        builder.add_algorithm(conf.get("algorithm", a), params=conf.get("params", {}), grid=conf.get("grid", {}))

    pipe = builder.build()
    print(">> Running pipeline")
    pipe.run()
    print("<< Finished running pipeline")

    pipe.save_metrics()


if __name__ == "__main__":
    run()
