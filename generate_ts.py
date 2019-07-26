from argparse import ArgumentParser

import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
rinterpreter = robjects.r
rinterpreter.source("timeseries.R")

AVAILABLE_IMPUTATION_METHODS = [
    str(n) for n in rinterpreter("AVAILABLE_IMPUTATION_METHODS").names
]
generate_dataset_ = rinterpreter("generate_dataset")


def generate_dataset(
        num_ts,
        num_obs,
        probs=None,
        methods=None,
        num_iters=10,
        seed=42,
):
    if probs is None:
        probs = [0.2]
    if methods is None:
        methods = AVAILABLE_IMPUTATION_METHODS
    df = generate_dataset_(
        num_ts,
        num_obs,
        probs=probs,
        methods=methods,
        num_iters=num_iters,
        seed=seed,
    )
    if not isinstance(df, pd.DataFrame):
        df = pandas2ri.ri2py(df)
    return df


def get_args():
    parser = ArgumentParser(
        description="Generate dataset for timeseries w/ missing data"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Location to save down dataframe",
    )
    parser.add_argument(
        "--num_ts",
        type=int,
        help="Number of different original (complete) timeseries",
    )
    parser.add_argument(
        "--num_obs",
        type=int,
        help="Number of observations per time series",
    )
    parser.add_argument(
        "--probs",
        type=float,
        nargs="+",
        help=
        "Missing values with given probabilities (generates new missing mask per prob)",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        help="Number of times to generate each missing prob mask",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        help="List of imputation methods to try (see timeseries.R for details)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed",
        default=42,
    )
    return parser.parse_args()


def main():
    args = get_args()
    df = generate_dataset(
        args.num_ts,
        args.num_obs,
        probs=args.probs,
        methods=args.methods,
        num_iters=args.num_iters,
        seed=args.seed,
    )
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
