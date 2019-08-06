require("imputeTS")
require("progress")
require("scales")
require("forecast")
require("tsoutliers")
require("data.table")


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
  kalman_arima=function(x) na_kalman(x, model="auto.arima")
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

generate_dataset <- function(input_, num_obs=200, prob_bounds=c(0.1, 0.5), methods=c("mean"), num_iters=10, seed=NULL) {
    if (!is.null(seed)) {
      set.seed(seed)
    }

    all_generated <- list()
    list_ix <- 1

    if (is.numeric(input_)) {
        generating_ts <- TRUE
        num_ts <- input_
    } else {
        stopifnot(is.data.frame(input_))
        generating_ts <- FALSE
        existing_ts_df <- input_
        ts_ids <- unique(existing_ts_df$ts_id)
        num_ts <- length(ts_ids)
    }

    total_iters <- num_ts * num_iters * length(methods)
    pb_bar <- progress_bar$new(total=total_iters)


    for (ts_ix in seq(num_ts)) {
        if (generating_ts) {
            ts_config <- gen_ts_config()
            ts_vec <- gen_ts(ts_config, num_obs)
            ts_id <- ts_ix
        } else {
            ts_id <- ts_ids[ts_ix]
            ts_data <- existing_ts_df[existing_ts_df$ts_id == ts_id, ]
            ts_vec <- ts_data$orig
        }
        for (iter_ in seq(num_iters)) {
            prob <- runif(1, min=prob_bounds[1], max=prob_bounds[2])
            modified <- add_missing(ts_vec, prob)
            for (method in methods) {

              copied <- modified
              copied$filled <- tryCatch(
                {impute_missing(copied$with_missing, method)},
                error = function(e) NULL
              )
              if (is.null(copied$filled)) {
                  print(paste("Skipping", "ts_id=", ts_id, ",iter_=", iter_, ",method=", method))
                  print("Fails to impute due to too few non-missing values")
                  pb_bar$tick()
                  next
              }
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
    # much faster than do.call(rbind)
    generated_df <- rbindlist(all_generated)
    as.data.frame(generated_df)
}


get_mse <- function(v1, v2) {
    mean((v1 - v2) ^ 2)
}


minimize_mse <- function(vec, probs, step_size, methods=NULL) {
  if (is.null(methods)) {
    methods <- names(AVAILABLE_IMPUTATION_METHODS)
  }

  thresh <- max(probs)
  curr_min_mse <- NULL
  curr_min_mse_method <- NULL
  curr_min_thresh <- thresh
  curr_predicted <- rep(FALSE, length(vec))

  history_mse <- c()
  history_thresh <- c()
  num_iters <- 0

  while (thresh > 0.0) {
      copy_vec <- vec
      copy_vec[probs >= thresh] <- NA
      iter_mse <- c()
      for(method in methods) {
          filled_vec <- tryCatch(
            {impute_missing(copy_vec, method)},
            error = function(e) NULL
          )
          if (!is.null(filled_vec)) {
              method_mse <- get_mse(vec, filled_vec)
              iter_mse <- c(iter_mse, method_mse)
          }
      }
      if (length(iter_mse) == 0) {
          # all methods failed to impute, too many missing values
          break
      }
      min_iter_mse <- min(iter_mse)
      history_mse <- c(history_mse, min_iter_mse)
      history_thresh <- c(history_thresh, thresh)
      if (is.null(curr_min_mse) || (min_iter_mse <= curr_min_mse)) {
        curr_min_mse <- min_iter_mse
        curr_min_mse_method <- methods[which(min_iter_mse == iter_mse)[1]]
        curr_min_thresh <- thresh
        curr_predicted <- is.na(copy_vec)
      }
      thresh <- thresh - step_size
      num_iters <- num_iters + 1
  }
  list(mse=curr_min_mse, method=curr_min_mse_method, threshold=curr_min_thresh, predicted=curr_predicted, num_iters=num_iters, history_mse=history_mse, history_thresh=history_thresh)
}

forecast_tsclean <- function(x) {
    cleaned <- tsclean(x, replace.missing=FALSE, lambda="auto")
    changed <- which(cleaned != x)
    copy_x <- x
    copy_x[changed] <- NA
    copy_x
}

tsoutliers_tsoutliers <- function(x) {
    results <- tsoutliers(x)
    copy_x <- x
    copy_x[results$index] <- NA
    copy_x
}
