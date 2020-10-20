# Script version in R of generate_ts.py
source("timeseries.R")
library("argparse")
library("data.table")

parser <- ArgumentParser(description='Generate dataset')
parser$add_argument("--existing_ts", type="character", help="Existing timeseries as csv table")
parser$add_argument("--num_ts", type="integer", help="Number of timeseries")
parser$add_argument("--num_obs", type="integer", help="Number of obs")
parser$add_argument("--simulate_ts", action="store_true", help="Replace original ts with auto.arima simulation", default=FALSE)
parser$add_argument("--num_iters", type="integer", default=10, help="Number of iterations")
parser$add_argument("--prob_bounds", type="double", nargs="+", default=c(0.2, 0.5))
parser$add_argument("--methods", type="character", nargs="+", default=names(AVAILABLE_IMPUTATION_METHODS), help="Imputation methods")
parser$add_argument("--seed", type="integer", default=42, help="Seed")
parser$add_argument("--output", type="character", help="Output path")

args <- parser$parse_args()

if (!is.null(args$num_ts)) {
    input_ <- args$num_ts
} else {
    input_ <- read.csv(args$existing_ts, header=TRUE, sep=",")
}

data <- generate_dataset(
    input_,
    num_obs=args$num_obs,
    simualte_ts=args$simulate_ts,
    prob_bounds=args$prob_bounds,
    methods=args$methods,
    num_iters=args$num_iters,
    seed=args$seed
)
# much faster version of write.csv from data.table package
print("Saving out file")
fwrite(data, file=args$output, row.names=FALSE)
