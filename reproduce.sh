#!/usr/bin/env bash
METHODS="mean median \
  mode random \
  zeros ones \
  forward backward \
  ma_simple ma_linear \
  ma_exponential linear_interpolation \
  spline_interpolation stine_interpolation \
  kalman_arima"

DATA_DIR="/Data/reverse-impute"
if [ ! -d ${DATA_DIR} ]
then
  echo "Creating ${DATA_DIR}"
  mkdir -p ${DATA_DIR}
fi

# length of time series
TS_LENGTH=100
EVAL_SAMPLE=5000
HIDDEN_SIZE=100
NUM_LAYERS=3

# Generate synthetic data
if [[ $# -eq 1 && $1 == "--generate" ]]
then
    Rscript generate_ts.R \
      --num_ts 1000 \
      --num_obs ${TS_LENGTH} \
      --num_iters 10 \
      --prob_bounds 0.0 0.3 \
      --methods ${METHODS} \
      --seed 42 \
      --output "${DATA_DIR}/generated.csv"

    # Download real world stock price data
    python download_stock_prices.py --output "${DATA_DIR}/sp500_prices.csv" --key "x-TJ2nRnm7qvede5xHx6"

    # Add synthetic missing values
    Rscript generate_ts.R \
      --existing_ts "${DATA_DIR}/sp500_prices.csv" \
      --num_iters 10 \
      --prob_bounds 0.0 0.3 \
      --methods ${METHODS} \
      --seed 42 \
      --output "${DATA_DIR}/sp500_prices_with_missing.csv"
fi

# Launch tensorboard to monitor training
TRAIN_LOG="${DATA_DIR}/train/logs/"
if [ -d ${TRAIN_LOG} ]
then
  rm -rf ${TRAIN_LOG}
fi
tensorboard --logdir ${TRAIN_LOG} --host 0.0.0.0 --port 8888 > /dev/null &

# Train model on synthetic data
mkdir "${DATA_DIR}/train/"
python training.py \
  --input "${DATA_DIR}/generated.csv" \
  --output "${DATA_DIR}/train/" \
  --valid 0.1 \
  --test 0.2 \
  --hidden_size ${HIDDEN_SIZE} \
  --num_layers ${NUM_LAYERS} \
  --num_iters 5 \
  --batch_size 100 \
  --valid_every_n_batches 100 \
  --seed 42 \
  --log ${TRAIN_LOG}


# Compute synthetic data results with different methods
mkdir "${DATA_DIR}/eval-synthetic/"
python evaluate.py \
    --dataset "${DATA_DIR}/train/dataset.pkl" \
    --model "${DATA_DIR}/train/model.pth" \
    --baselines tsoutliers tsclean manual \
    --hidden_size ${HIDDEN_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --sample ${EVAL_SAMPLE} \
    --seed 42 \
    --output "${DATA_DIR}/eval-synthetic/no-noise-results.csv"

python evaluate.py \
    --dataset "${DATA_DIR}/train/dataset.pkl" \
    --model "${DATA_DIR}/train/model.pth" \
    --baselines tsoutliers tsclean manual \
    --hidden_size ${HIDDEN_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --with_noise \
    --sample ${EVAL_SAMPLE} \
    --seed 42 \
    --output "${DATA_DIR}/eval-synthetic/with-noise-results.csv"

# Compute results on sp500 prices
mkdir "${DATA_DIR}/eval-sp500/"
python evaluate.py \
    --csv "${DATA_DIR}/sp500_prices_with_missing.csv" \
    --valid 0.25 \
    --test 0.75 \
    --model "${DATA_DIR}/train/model.pth" \
    --hidden_size ${HIDDEN_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --baselines tsoutliers tsclean manual \
    --ts_length ${TS_LENGTH} \
    --sample ${EVAL_SAMPLE} \
    --seed 42 \
    --output "${DATA_DIR}/eval-sp500/no-noise-results.csv"


python evaluate.py \
    --csv "${DATA_DIR}/sp500_prices_with_missing.csv" \
    --valid 0.25 \
    --test 0.75 \
    --model "${DATA_DIR}/train/model.pth" \
    --hidden_size ${HIDDEN_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --baselines tsoutliers tsclean manual \
    --ts_length ${TS_LENGTH} \
    --with_noise \
    --seed 42 \
    --sample ${EVAL_SAMPLE} \
    --output "${DATA_DIR}/eval-sp500/with-noise-results.csv"



# train on real data
mkdir "${DATA_DIR}/train-sp500/"
python training.py \
  --input "${DATA_DIR}/sp500_prices_with_missing.csv" \
  --output "${DATA_DIR}/train-sp500/" \
  --valid 0.2 \
  --test 0.2 \
  --hidden_size ${HIDDEN_SIZE} \
  --num_layers ${NUM_LAYERS} \
  --num_iters 5 \
  --batch_size 100 \
  --ts_length ${TS_LENGTH} \
  --sample ${EVAL_SAMPLE} \
  --valid_every_n_batches 100 \
  --seed 42 \
  --log "${DATA_DIR}/train-sp500/logs/"



# # TODO: this later part hasn't really been worked on
# # Compute MSE impact for future forecasts
# python repair_ts.py \
#   --dataset eval-sp500-results/eval-dataset.pkl \
#   --model train-results/model.pth \
#   --hidden_size 50 \
#   --output eval-sp500-results/predicted-imputed-sp500.csv
#
# Rscript forecast_impact.R \
#   --orig_data sp500_prices_with_missing.csv \
#   --new_data eval-sp500-results/predicted-imputed-sp500.csv \
#   --n_first 100 \
#   --output eval-sp500-results/forecast-impact.csv \
