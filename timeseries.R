require("imputeTS")
require("progress")
require("scales")

AVAILABLE_IMPUTATION_METHODS <- list(
  mean=function(x) na_mean(x, option="mean"),
  median=function(x) na_mean(x, option="median"),
  mode=function(x) na_mean(x, option="mode"),
  random=function(x) na_random(x),
  zeros=function(x) na_replace(x, fill=0),
  ones=function(x) na_replace(x, fill=1),
  forward=function(x) na_locf(x, option="locf"),
  backward=function(x) na_locf(x, option="nocb"),
  ma_simple=function(x) na_ma(x, weighting="simple"),
  ma_linear=function(x) na_ma(x, weighting="linear"),
  ma_exponential=function(x) na_ma(x, weighting="exponential"),
  linear_interpolation=function(x) na_interpolation(x, option="linear"),
  spline_interpolation=function(x) na_interpolation(x, option="spline"),
  stine_interpolation=function(x) na_interpolation(x, option="stine"),
  kalman_arima=function(x) scaled_na_kalman(x, model="auto.arima")
)


gen_ts_config <- function(max_ar=5, max_ma=5, max_diff=3) {
  num_ar <- sample(1:max_ar, 1)
  num_ma <- sample(1:max_ma, 1)
  num_d <- sample(1:max_diff, 1)

  frac_non_ar <- runif(1, 0.0, 1.0)
  ar <- runif(num_ar, 0.0, 1.0)
  # dumb way to make sure stationary
  ar <- (ar / sum(ar))  * (1 - frac_non_ar)
  ma <- runif(num_ma, 0.0, 1.0)

  list(order=c(num_ar, num_d, num_ma), ar=ar, ma=ma)
}


gen_ts <- function(config, n) {
  ext_n <- n + config$order[3] - 1
  vec <- arima.sim(model=config, n=ext_n)
  vec <- as.numeric(vec)
  vec[seq(config$order[3], ext_n)]
}


gen_missing_mask <- function(n, prob) {
    runif(n) < prob
}


add_missing <- function(vec, prob) {
    mask <- gen_missing_mask(length(vec), prob)
    copy_vec <- vec
    copy_vec[mask] <- NA
    list(orig=vec, with_missing=copy_vec, mask=mask, time=seq_along(vec))
}

impute_missing <- function(vec, method_name) {
    chosen_method <- AVAILABLE_IMPUTATION_METHODS[[method_name]]
    chosen_method(vec)
}

generate_dataset <- function(num_ts, num_obs=200, probs=c(0.2), methods=c("mean"), num_iters=10, seed=NULL) {
    if (!is.null(seed)) {
      set.seed(seed)
    }

    all_generated <- list()
    list_ix <- 1

    total_iters <- num_ts * length(probs) * num_iters * length(methods)
    pb_bar <- progress_bar$new(total=total_iters)


    for (ts_id in seq(num_ts)) {
        ts_config <- gen_ts_config()
        ts_vec <- gen_ts(ts_config, num_obs)

        for (prob in probs) {

          for (iter_ in seq(num_iters)) {

              modified <- add_missing(ts_vec, prob)
              for (method in methods) {

                copied <- modified
                copied$filled <- impute_missing(copied$with_missing, method)

                generated_df <- as.data.frame(copied)
                generated_df$prob <- prob
                generated_df$method <- method
                generated_df$ts_id <- ts_id
                generated_df$iter <- iter_
                generated_df$unique_id <- list_ix

                all_generated[[list_ix]] <- generated_df
                list_ix <- list_ix + 1
                pb_bar$tick()
              }
          }
        }
    }
    do.call(rbind, all_generated)
}
