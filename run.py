import click
import datetime

from recpack.datasets import AdressaOneWeek, CosmeticsShop, RecsysChallenge2015, Netflix
from recpack.pipelines import PipelineBuilder, OptimisationInfo, GridSearchInfo, HyperoptInfo
from recpack.scenarios import Timed, TimedLastItemPrediction
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem

from amazon_dataset import AmazonGamesDataset, AmazonToysAndGamesDataset

from algorithm_config import ALGORITHM_CONFIG

DATASET_PATH = "/home/robinverachtert/datasets/"


def recsys_dataset(dataset_path):
    ds = RecsysChallenge2015(path=dataset_path, filename="yoochoose-clicks.dat", use_default_filters=False)
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
            "dataset": CosmeticsShop(path=dataset_path, filename="archive_cosmeticsshop.zip"),
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
        "netflix": {
            "dataset": Netflix(dataset_path),
            "t": int(datetime.datetime(2005, 10, 1, 0).strftime("%s")),
            "t_val": int(datetime.datetime(2005, 9, 1, 0).strftime("%s")),
            "delta_out": 31 * 24 * 3600,
        },
        "amazon_games": {
            "dataset": AmazonGamesDataset(dataset_path),
            "t": int(datetime.datetime(2018, 4, 1, 0).strftime("%s")),
            "t_val": int(datetime.datetime(2017, 10, 1, 0).strftime("%s")),
            "delta_out": 6 * 31 * 24 * 3600,
        },
        "amazon_toys_and_games": {
            "dataset": AmazonToysAndGamesDataset(dataset_path),
            "t": int(datetime.datetime(2018, 4, 1, 0).strftime("%s")),
            "t_val": int(datetime.datetime(2017, 10, 1, 0).strftime("%s")),
            "delta_out": 6 * 31 * 24 * 3600,
        },
    }
    return datasets[dataset]


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
@click.option("--scenario", help="the scenario to evaluate", default="TimedLastItemPrediction")
@click.option("--results-path", help="path to put results, defaults to results", default="results")
@click.option(
    "--experiment-name",
    "-en",
    help="name of the experiment, will define the folder written inside results-path. Defaults to {dataset}",
    default=None,
)
def run(dataset, dataset_path, algorithm, scenario, results_path, experiment_name):
    print(f"running {', '.join(algorithm)} on {dataset}")

    experiment_name = experiment_name if experiment_name else dataset

    for a in algorithm:
        if a not in ALGORITHM_CONFIG:
            raise ValueError(f"{a} not supported in experiment, please use one of the preconfigured algorithms.")

    if scenario not in ["Timed", "TimedLastItemPrediction"]:
        raise ValueError(f"{scenario} is not supported. Use one of {['Timed', 'TimedLastItemPrediction']}")

    print(">> Loading dataset")
    im = get_datasets_info(dataset_path, dataset)["dataset"].load()
    print("<< Loaded dataset")
    print(f"dataset shape = {im.shape}")

    t = get_datasets_info(dataset_path, dataset)["t"]
    t_val = get_datasets_info(dataset_path, dataset)["t_val"]
    delta_out = get_datasets_info(dataset_path, dataset)["delta_out"]

    print(">> Splitting dataset")
    if scenario == "TimedLastItemPrediction":
        scenario = TimedLastItemPrediction(t=t, t_validation=t_val, validation=True, delta_out=delta_out)
        scenario.split(im)
    elif scenario == "Timed":
        scenario = Timed(t=t, t_validation=t_val, validation=True, delta_out=delta_out)
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
        builder.add_algorithm(
            conf.get("algorithm", a),
            params=conf.get("params", {}),
            optimisation_info=conf.get("optimisation_info", None),
        )

    pipe = builder.build()
    print(">> Running pipeline")
    pipe.run()
    print("<< Finished running pipeline")

    pipe.save_metrics()


if __name__ == "__main__":
    run()
