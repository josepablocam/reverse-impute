library("imputeTS")
library("forecast")
source("timeseries.R")

compare_mse_impact <- function(orig_data, new_data, n_first) {
      results <- list()
      unique_ids <- intersect(unique(orig_data$unique_id), unique(new_data$unique_id))
      iter_ct <- 0

      for(unique_id in unique_ids) {

          prev_data <- orig_data[orig_data$unique_id == unique_id]
          new_data <- new_data[new_data$unique_id == unique_id]
          # remove predicted imputed in new_data
          new_data$filled[which(new_data$predicted_mask)] <- NA
          # fill in again but with kalman filter
          new_data$filled <- na_kalman(new_data$filled, model="auto.arima")

          prev_prediction <- forecast(prev_data$filled[1:n_first])
          new_prediction <- forecast(new_data$filled[1:n_first])


          observed <- prev_data$orig[(n_first + 1):]
          prev_mse <- get_mse(observed, prev_prediction)
          new_mse <- get_mse(observed, new_prediction)

          iter_result <- list(
              unique_id=unique_id,
              prev_method=prev_data$impute_method[1],
              prob=prev_data$prob[1],
              prev_mse=prev_mse,
              new_mse=new_mse,
          )
          iter_ct <- iter_ct + 1
          results[[iter_ct]] <- as.data.frame(iter_result)
      }

      do.call(rbind, results)
}
