from hyperopt import hp
from recpack.algorithms import TARSItemKNNCoocDistance
from recpack.pipelines import GridSearchInfo, HyperoptInfo

HOUR = 3600  # seconds
DAY = 24 * HOUR

SIMILARITY_FUNCTIONS = ["cosine", "conditional_probability"]

ALGORITHM_CONFIG = {
    # Literature algorithms
    "TARSItemKNNLee_W3": {
        "optimisation_info": GridSearchInfo({"similarity": ["cosine", "pearson"]}),
    },
    "TARSItemKNNLee_W5": {
        "optimisation_info": GridSearchInfo({"similarity": ["cosine", "pearson"]}),
    },
    "TARSItemKNNLiu": {
        "optimisation_info": GridSearchInfo(
            {
                "fit_decay": [0]
                + [1 / x for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY]],
                "predict_decay": [0]
                + [1 / x for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY]],
            }
        ),
    },
    "TARSItemKNNDing": {
        "optimisation_info": GridSearchInfo(
            {
                "predict_decay": [0]
                + [1 / x for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY]],
                "similarity": ["cosine", "conditional_probability"],
            }
        ),
    },
    "TARSItemKNNLiu2012": {
        "optimisation_info": GridSearchInfo(
            {
                "decay": [2, 5, 10, 50, 100, 200, 500, 1000],
            }
        ),
    },
    "TARSItemKNNVaz": {
        "optimisation_info": GridSearchInfo(
            {
                "fit_decay": [
                    1 / x
                    for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY, 30 * DAY]
                ],
                "predict_decay": [
                    1 / x
                    for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY, 30 * DAY]
                ],
            }
        )
    },
    "TARSItemKNNHermann": {
        "optimisation_info": HyperoptInfo(
            space={"decay_interval": hp.uniformint("decay_interval", 1, 30 * DAY)}, timeout=DAY, max_evals=50
        )
    },
    "TARSItemKNNXia_concave": {
        "algorithm": "TARSItemKNNXia",
        "optimisation_info": HyperoptInfo(
            {
                "fit_decay": hp.uniform("fit_decay", 0, 1),
                "decay_interval": hp.uniformint("decay_interval", 1, 30 * DAY),
            },
            timeout=DAY,
            max_evals=50,
        ),
        "params": {"decay_function": "concave"},
    },
    "TARSItemKNNXia_linear": {
        "algorithm": "TARSItemKNNXia",
        "optimisation_info": HyperoptInfo(
            {
                "fit_decay": hp.uniform("fit_decay", 0, 10),
                "decay_interval": hp.uniformint("decay_interval", 1, 30 * DAY),
            },
            timeout=DAY,
            max_evals=50,
        ),
        "params": {"decay_function": "linear"},
    },
    "TARSItemKNNXia_convex": {
        "algorithm": "TARSItemKNNXia",
        "optimisation_info": HyperoptInfo(
            {
                "fit_decay": hp.uniform("fit_decay", 0, 1),
                "decay_interval": hp.uniformint("decay_interval", 1, 30 * DAY),
            },
            timeout=DAY,
            max_evals=50,
        ),
        "params": {"decay_function": "convex"},
    },
    # Extensions
    "TARSItemKNNexponential": {
        "algorithm": "TARSItemKNN",
        "optimisation_info": GridSearchInfo(
            {
                "similarity": SIMILARITY_FUNCTIONS,
                "fit_decay": [
                    1 / x
                    for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY, 30 * DAY]
                ]
                + [0],
                "predict_decay": [
                    1 / x
                    for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY, 30 * DAY]
                ]
                + [0],
            }
        ),
        "params": {"decay_function": "exponential"},
    },
    "TARSItemKNNlog": {
        "algorithm": "TARSItemKNN",
        "optimisation_info": GridSearchInfo(
            {
                "similarity": SIMILARITY_FUNCTIONS,
                "fit_decay": [2, 4, 8, 16, 32],
                "predict_decay": [2, 4, 8, 16, 32],
                "decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY],
            }
        ),
        "params": {"decay_function": "log"},
    },
    "TARSItemKNNlinear": {
        "algorithm": "TARSItemKNN",
        "optimisation_info": GridSearchInfo(
            {
                "similarity": SIMILARITY_FUNCTIONS,
                "fit_decay": [0.1, 0.3, 0.5, 0.7, 0.9, 1, 5, 10, 50, 100, 1000],
                "predict_decay": [0.1, 0.3, 0.5, 0.7, 0.9, 1, 5, 10, 50, 100, 1000],
            }
        ),
        "params": {"decay_function": "linear"},
    },
    "TARSItemKNNconcave": {
        "algorithm": "TARSItemKNN",
        "optimisation_info": GridSearchInfo(
            {
                "similarity": SIMILARITY_FUNCTIONS,
                "fit_decay": [0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
                "predict_decay": [0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
            }
        ),
        "params": {"decay_function": "concave"},
    },
    "TARSItemKNNinverse": {
        "algorithm": "TARSItemKNN",
        "optimisation_info": GridSearchInfo(
            {
                "similarity": SIMILARITY_FUNCTIONS,
                "decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY],
            }
        ),
        "params": {"decay_function": "inverse"},
    },
    "TARSItemKNNCoocDistanceexponential": {
        "algorithm": "TARSItemKNNCoocDistance",
        "optimisation_info": HyperoptInfo(
            {
                "similarity": hp.choice("similarity", TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES),
                "fit_decay": hp.uniform("fit_decay", 0, 1),
                "predict_decay": hp.uniform("predict_decay", 0, 1),
                "event_age_weight": hp.uniform("event_age_weight", 0, 1),
            },
            timeout=DAY,
            max_evals=50,
        ),
        "params": {"decay_function": "exponential"},
    },
    "TARSItemKNNCoocDistancelog": {
        "algorithm": "TARSItemKNNCoocDistance",
        "optimisation_info": HyperoptInfo(
            {
                "similarity": hp.choice("similarity", TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES),
                "fit_decay": hp.uniformint("fit_decay", 2, 64),
                "predict_decay": hp.uniformint("predict_decay", 2, 64),
                "decay_interval": hp.uniformint("decay_interval", 1, 30 * DAY),
                "event_age_weight": hp.uniform("event_age_weight", 0, 1),
            },
            timeout=DAY,
            max_evals=50,
        ),
        "params": {"decay_function": "log"},
    },
    "TARSItemKNNCoocDistancelinear": {
        "algorithm": "TARSItemKNNCoocDistance",
        "optimisation_info": HyperoptInfo(
            {
                "similarity": hp.choice("similarity", TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES),
                "fit_decay": hp.uniform("fit_decay", 0, 10),
                "predict_decay": hp.uniform("predict_decay", 0, 10),
                "decay_interval": hp.uniformint("decay_interval", 1, 30 * DAY),
                "event_age_weight": hp.uniform("event_age_weight", 0, 1),
            },
            timeout=DAY,
            max_evals=50,
        ),
        "params": {"decay_function": "linear"},
    },
    "TARSItemKNNCoocDistanceconcave": {
        "algorithm": "TARSItemKNNCoocDistance",
        "optimisation_info": HyperoptInfo(
            {
                "similarity": hp.choice("similarity", TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES),
                "fit_decay": hp.uniform("fit_decay", 0, 1),
                "predict_decay": hp.uniform("predict_decay", 0, 1),
                "decay_interval": hp.uniformint("decay_interval", 1, 30 * DAY),
                "event_age_weight": hp.uniform("event_age_weight", 0, 1),
            },
            timeout=DAY,
            max_evals=50,
        ),
        "params": {"decay_function": "concave"},
    },
}
