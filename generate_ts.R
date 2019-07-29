# Script version in R of generate_ts.py
source("timeseries.R")
library("argparse")

parser <- ArgumentParser(description='Generate dataset')
parser$add_argument("--num_ts", type="integer", help="Number of timeseries")
parser$add_argument("--num_obs", type="integer", help="Number of obs")
parser$add_argument("--num_iters", type="integer", default=10, help="Number of iterations")
parser$add_argument("--methods", type="character", nargs="+", default=names(AVAILABLE_IMPUTATION_METHODS), help="Imputation methods")
parser$add_argument("--seed", type="integer", default=42, help="Seed")
parser$add_argument("--output", type="character", help="Output path")

args <- parser$parse_args()
data <- generate_dataset(
    args$num_ts,
    num_obs=args$num_obs,
    methods=args$methods,
    num_iters=args$num_iters,
    seed=args$seed
)
write.csv(data, args$output)
