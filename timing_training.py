# %%
from run import get_datasets_info
from recpack.algorithms import (
    TARSItemKNNDing,
    TARSItemKNNLee_W5,
    TARSItemKNNLiu,
    TARSItemKNNLiu2012,
    TARSItemKNNVaz,
    GRU4Rec,
    EASE,
    ItemKNN,
    SequentialRules,
    TARSItemKNN,
    TARSItemKNNHermann,
    TARSItemKNNXia,
)

# %%
dataset_path = "/Users/robinverachtert/workspace/datasets/"


# %%
from recpack.scenarios import Timed


# %%
import time

timings = []

# %%

for dataset in ["adressa", "cosmeticsshop", "recsys2015", "amazon_games", "amazon_toys_and_games"]:
    ds_info = get_datasets_info(dataset_path, dataset)
    im = ds_info["dataset"].load()
    scenario = Timed(t=ds_info["t"], t_validation=ds_info["t_val"], validation=True, delta_out=ds_info["delta_out"])
    scenario.split(im)

    for algorithm in [
        TARSItemKNNDing,
        TARSItemKNNLee_W5,
        TARSItemKNNLiu,
        TARSItemKNNLiu2012,
        TARSItemKNNVaz,
        GRU4Rec,
        EASE,
        ItemKNN,
        SequentialRules,
        TARSItemKNN,
        TARSItemKNNHermann,
        TARSItemKNNXia,
    ]:
        if dataset in ["recsys2015", "amazon_toys_and_games"] and algorithm == EASE:
            continue  # These datasets have too many items.
        start = time.time()
        algo = algorithm()  # Just using default parameters, as we just need the timing.
        algo.fit(scenario.full_training_data)
        fit = time.time()
        algo.predict(scenario.test_data_in)
        end = time.time()
        time_dict = {
            "algorithm": algo.name,
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
