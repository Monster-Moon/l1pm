\name{l1_p}
\alias{l1_p}
\title{The l1 Penalization method}
\usage{
l1_p(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, hidden_dim3 = NULL,
     learning_rate, max_deep_iter, lambda_obj, penalty)
}
\description{
Estimates non crossing quantile regression with a neural network.
}
\examples{
set.seed(1)
n = 1000L
input_dim = 1L

### train data
x_data = matrix(runif( n* input_dim, -1, 1), n, input_dim)
sincx = sin(pi * x_data) / (pi * x_data)
Z = matrix(sincx, nrow = n, ncol = input_dim)
ep = rnorm(n, mean = 0, sd = 0.1 * exp(1 - x_data)) ## example 1
y_data = Z + ep
plot(x_data, y_data)

### valid data
x_valid_data = matrix(runif( n * input_dim, -1, 1), n, input_dim)
sincx_valid = sin(pi * x_valid_data) / (pi * x_valid_data)
y_valid = matrix(sincx_valid, nrow = n, ncol = input_dim) +
            rnorm(n, mean = 0, sd = 0.1 * exp(1 - x_valid_data))

### test data
x_test_data = matrix(runif( n * input_dim, -1, 1), n, input_dim)
sincx_test = sin(pi * x_test_data) / (pi * x_test_data)
y_test = matrix(sincx_test, nrow = n, ncol = input_dim) +
          rnorm(n, mean = 0, sd = 0.1 * exp(1 - x_test_data))

### Model fitting
tau_vec = seq(0.1, 0.9, 0.1)
fit_result = l1_p(X = x_data,
                  y = y_data,
                  test_X = x_test_data,
                  tau = tau_vec,
                  hidden_dim1 = 4,
                  hidden_dim2 = 4,
                  learning_rate = 0.005,
                  max_deep_iter = 5000,
                  lambda_obj = 5)

predicted_train = fit_result$y_predict
predicted_valid = fit_result$y_valid_predict
predicted_test = fit_result$y_test_predict
}
