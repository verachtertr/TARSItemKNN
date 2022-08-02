from recpack.algorithms import TARSItemKNN, TARSItemKNNCoocDistance

HOUR = 3600  # seconds
DAY = 24 * HOUR

SIMILARITY_FUNCTIONS = ["cosine", "conditional_probability"]

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
    # "TARSItemKNNHermann": {"grid": {"decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY]}},
    # "TARSItemKNNXia_concave": {"algorithm": "TARSItemKNNXia", "grid": {"fit_decay": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]}},
    # "TARSItemKNNXia_convex": {"algorithm": "TARSItemKNNXia", "grid": {"fit_decay": [0.1, 0.3, 0.5, 0.7, 0.9]}},
    # "TARSItemKNNXia_linear": {
    #     "algorithm": "TARSItemKNNXia",
    #     "grid": {"fit_decay": [0, 0.5, 1], "decay_interval": [DAY]},
    # },
    # Extensions
    # "TARSItemKNNLee": {
    #     "grid": {"similarity": ["cosine", "pearson"], "W": [2, 3, 4, 5, 8, 10, 16]},
    # },
    "TARSItemKNNexponential": {
        "algorithm": "TARSItemKNN",
        "grid": {
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
        },
        "params": {"decay_function": "exponential"},
    },
    "TARSItemKNNlog": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": SIMILARITY_FUNCTIONS,
            "fit_decay": [2, 4, 8, 16, 32],
            "predict_decay": [2, 4, 8, 16, 32],
            "decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY],
        },
        "params": {"decay_function": "log"},
    },
    "TARSItemKNNlinear": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": SIMILARITY_FUNCTIONS,
            "fit_decay": [0.1, 0.3, 0.5, 0.7, 0.9, 1],
            "predict_decay": [0.1, 0.3, 0.5, 0.7, 0.9, 1],
        },
        "params": {"decay_function": "linear"},
    },
    "TARSItemKNNconcave": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": SIMILARITY_FUNCTIONS,
            "fit_decay": [0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
            "predict_decay": [0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
        },
        "params": {"decay_function": "concave"},
    },
    "TARSItemKNNlinear_steeper": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": SIMILARITY_FUNCTIONS,
            "fit_decay": [1, 5, 10, 50, 100, 1000],
            "predict_decay": [1, 5, 10, 50, 100, 1000],
        },
        "params": {"decay_function": "linear_steeper"},
    },
    "TARSItemKNNinverse": {
        "algorithm": "TARSItemKNN",
        "grid": {
            "similarity": SIMILARITY_FUNCTIONS,
            "decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY],
        },
        "params": {"decay_function": "inverse"},
    },
    # "TARSItemKNNCoocDistanceexponential": {
    #    "algorithm": "TARSItemKNNCoocDistance",
    #    "grid": {
    #        "similarity": TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES,
    #        "fit_decay": [
    #            1 / x
    #            for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY, 30 * DAY]
    #        ]
    #        + [0],
    #        "predict_decay": [
    #            1 / x
    #            for x in [1 * HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, 1 * DAY, 7 * DAY, 14 * DAY, 30 * DAY]
    #        ]
    #        + [0],
    #        "event_age_weight": [0, 0.25, 0.5, 0.75, 1],
    #    },
    #    "params": {"decay_function": "exponential"},
    # },
    # "TARSItemKNNCoocDistancelog": {
    #     "algorithm": "TARSItemKNNCoocDistance",
    #     "grid": {
    #         "similarity": TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES,
    #         "fit_decay": [2, 4, 8, 16, 32],
    #         "predict_decay": [2, 4, 8, 16, 32],
    #         "decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY],
    #         "event_age_weight": [0, 0.25, 0.5, 0.75, 1],
    #     },
    #     "params": {"decay_function": "log"},
    # },
    # "TARSItemKNNCoocDistancelinear": {
    #    "algorithm": "TARSItemKNNCoocDistance",
    #    "grid": {
    #        "similarity": TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES,
    #        "fit_decay": [0.1, 0.3, 0.5, 0.7, 0.9, 1],
    #        "predict_decay": [0.1, 0.3, 0.5, 0.7, 0.9, 1],
    #        "decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY],
    #        "event_age_weight": [0, 0.25, 0.5, 0.75, 1],
    #    },
    #    "params": {"decay_function": "linear"},
    # },
    # "TARSItemKNNCoocDistanceconcave": {
    #     "algorithm": "TARSItemKNNCoocDistance",
    #     "grid": {
    #         "similarity": TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES,
    #         "fit_decay": [0.01, 0.1, 0.3, 0.9],
    #         "predict_decay": [0.01, 0.1, 0.3, 0.9],
    #         "decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY],
    #         "event_age_weight": [0, 0.25, 0.5, 0.75, 1],
    #     },
    #     "params": {"decay_function": "concave"},
    # },
    # "TARSItemKNNCoocDistanceconvex": {
    #     "algorithm": "TARSItemKNNCoocDistance",
    #     "grid": {
    #         "similarity": TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES,
    #         "fit_decay": [0.01, 0.1, 0.3, 0.9],
    #         "predict_decay": [0.01, 0.1, 0.3, 0.9],
    #         "decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY],
    #         "event_age_weight": [0, 0.25, 0.5, 0.75, 1],
    #     },
    #     "params": {"decay_function": "convex"},
    # },
    # "TARSItemKNNCoocDistancelinear_steeper": {
    #     "algorithm": "TARSItemKNNCoocDistance",
    #     "grid": {
    #         "similarity": TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES,
    #         "fit_decay": [1, 5, 10, 50, 100, 1000],
    #         "predict_decay": [1, 5, 10, 50, 100, 1000],
    #         "decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY],
    #         "event_age_weight": [0, 0.25, 0.5, 0.75, 1],
    #     },
    #     "params": {"decay_function": "linear_steeper"},
    # },
    # "TARSItemKNNCoocDistanceinverse": {
    #    "algorithm": "TARSItemKNNCoocDistance",
    #    "grid": {
    #        "similarity": TARSItemKNNCoocDistance.SUPPORTED_SIMILARITIES,
    #        "decay_interval": [1, HOUR, DAY, 7 * DAY, 30 * DAY],
    #        "event_age_weight": [0, 0.25, 0.5, 0.75, 1],
    #    },
    #    "params": {"decay_function": "inverse"},
    # },
}
