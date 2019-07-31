# Script version in R of generate_ts.py
source("forecast_impact.R")
library("argparse")

parser <- ArgumentParser(description='Evaluate impact on forecasting MSE of imputation/reimputation')
parser$add_argument("--old_data", type="character", help="Original csv dataset")
parser$add_argument("--new_data", type="character", help="New csv dataset")
parser$add_argument("--n_first", type="integer", help="Take first n to fit forecasting model", default=100)
parser$add_argument("--n_next", type="integer", help="Steps into future for prediction", default=10)
parser$add_argument("--output", type="character", help="Output path")

args <- parser$parse_args()
old_data <- read.csv(args$old_data)
new_data <- read.csv(args$new_data)

results <- compare_mse_impact(
    old_data,
    new_data,
    args$n_first
)
write.csv(results, args$output)
