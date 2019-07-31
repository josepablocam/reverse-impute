#!/usr/bin/env bash
METHODS="mean median \
  mode random \
  zeros ones \
  forward backward \
  ma_simple ma_linear \
  ma_exponential linear_interpolation \
  spline_interpolation stine_interpolation \
  kalman_arima"

# Generate synthetic data
Rscript generate_ts.R \
  --num_ts 1000 \
  --num_obs 100 \
  --num_iters 10 \
  --prob_bounds 0.2 0.5 \
  --methods ${METHODS} \
  --seed 42 \
  --output generated.csv

# Download real world stock price data
python download_stock_prices.py --output sp500_prices.csv

# Add synthetic missing values
Rscript generate_ts.R \
  --existing_ts sp500_prices.csv \
  --num_iters 10 \
  --prob_bounds 0.2 0.5 \
  --methods ${METHODS} \
  --seed 42 \
  --output sp500_prices_with_missing.csv

# Train model on synthetic data
python train.py \
  --input generated.csv \
  --output train-results/ \
  --valid 0.2 \
  --test 0.1 \
  --hidden 50 \
  --num_iters 20 \
  --batch_size 100 \
  --valid_every_n_batches 10 \
  --seed 42

# Compute synthetic data results with different methods
python evaluate.py \
    --input train-results/dataset.pkl \
    --model train-results/model.pth \
    --baselines tsoutliers tsclean manual \
    --output eval-synthetic-results/generated-results.csv

# Compute results on sp500 prices
python evaluate.py \
    --csv sp500_prices_with_missing.csv \
    --valid 0.5 \
    --test 0.5 \
    --model exp1/model.pth \
    --baselines tsoutliers tsclean manual \
    --output eval-sp500-results/sp500-results.csv

# Compute MSE impact for future forecasts
python repair_ts.py \
  --dataset eval-sp500-results/eval-dataset.pkl \
  --model train-results/model.pth \
  --hidden_size 50 \
  --output eval-sp500-results/predicted-imputed-sp500.csv

Rscript forecast_impact.R \
  --orig_data sp500_prices_with_missing.csv \
  --new_data eval-sp500-results/predicted-imputed-sp500.csv \
  --n_first 100 \
  --output eval-sp500-results/forecast-impact.csv \
