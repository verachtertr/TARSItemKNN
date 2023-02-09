# %%
from run import get_datasets_info
from recpack.algorithms import (
    TARSItemKNNDing,
    TARSItemKNNLee_W5,
    TARSItemKNNLiu,
    TARSItemKNNLiu2012,
    TARSItemKNNVaz,
    GRU4RecNegSampling,
    EASE,
    ItemKNN,
    SequentialRules,
    TARSItemKNN,
    TARSItemKNNHermann,
    TARSItemKNNXia,
    TorchMLAlgorithm,
)

# %%
dataset_path = "/home/robinverachtert/datasets/"


# %%
from recpack.scenarios import Timed


# %%
import time

timings = []

# %%

for dataset in ["cosmeticsshop", "adressa", "recsys2015", "amazon_games", "amazon_toys_and_games"]:
    ds_info = get_datasets_info(dataset_path, dataset)
    im = ds_info["dataset"].load()
    scenario = Timed(t=ds_info["t"], t_validation=ds_info["t_val"], validation=True, delta_out=ds_info["delta_out"])
    scenario.split(im)

    for algorithm in [
        TARSItemKNNDing(),
        TARSItemKNNLee_W5(),
        TARSItemKNNLiu(),
        TARSItemKNNLiu2012(),
        TARSItemKNNVaz(),
        GRU4RecNegSampling(validation_sample_size=10000, batch_size=512, max_epochs=8, predict_topK=100),
        EASE(),
        ItemKNN(),
        SequentialRules(),
        TARSItemKNN(),
        TARSItemKNNHermann(),
        TARSItemKNNXia(),
    ]:
        if dataset in ["recsys2015", "amazon_toys_and_games"] and issubclass(type(algorithm), EASE):
            continue  # These datasets have too many items.
        start = time.time()
        if issubclass(type(algorithm), TorchMLAlgorithm):
            algorithm.fit(scenario.validation_training_data, scenario.validation_data)
        else:
            algorithm.fit(scenario.full_training_data)
        fit = time.time()
        algorithm.predict(scenario.test_data_in)
        end = time.time()
        time_dict = {
            "algorithm": algorithm.name,
            "dataset": dataset,
            "timing": end - start,
            "training_time": fit - start,
            "prediction_time": end - fit,
        }
        timings.append(time_dict)
        print(f"timing: {time_dict}")

# %%
import pandas as pd

df = pd.DataFrame.from_records(timings)
df.to_csv("algorithm_timings.csv", header=True, index=False)

# %%
