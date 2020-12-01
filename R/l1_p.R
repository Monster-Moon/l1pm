
#' l1_p
#' @description Estimates non crossing quantile regression with a neural network.
#' @param X train predictor data
#' @param y train response data
#' @param test_X test predictor data
#' @param valid_X validation predictor data
#' @param tau target quantiles
#' @param hidden_dim1 the number of nodes in the first hidden layer
#' @param hidden_dim2 the number of nodes in the second hidden layer
#' @param learning_rate learning rate in the optimization process
#' @param max_deep_iter the number of iterations
#' @param lambda_obj the value of tuning parameter in the l1 penalization method
#' @param penalty the value of tuning parameter for ridge penalty on weights
#' @return y_predicted, y_test_predicted, y_valid_predited : predicted quantile based on train, test, and validation data, respectively
# @examples
# set.seed(1)
# n = 1000L
# input_dim = 1L
#
# ### train data
# x_data = matrix(runif( n* input_dim, -1, 1), n, input_dim)
# sincx = sin(pi * x_data) / (pi * x_data)
# Z = matrix(sincx, nrow = n, ncol = input_dim)
# ep = rnorm(n, mean = 0, sd = 0.1 * exp(1 - x_data)) ## example 1
# y_data = Z + ep
# plot(x_data, y_data)
#
# ### valid data
# x_valid_data = matrix(runif( n * input_dim, -1, 1), n, input_dim)
# sincx_valid = sin(pi * x_valid_data) / (pi * x_valid_data)
# y_valid = matrix(sincx_valid, nrow = n, ncol = input_dim) +
#   rnorm(n, mean = 0, sd = 0.1 * exp(1 - x_valid_data))
#
# ### test data
# x_test_data = matrix(runif( n * input_dim, -1, 1), n, input_dim)
# sincx_test = sin(pi * x_test_data) / (pi * x_test_data)
# y_test = matrix(sincx_test, nrow = n, ncol = input_dim) +
#   rnorm(n, mean = 0, sd = 0.1 * exp(1 - x_test_data))
#
# ### Model fitting
# tau_vec = seq(0.1, 0.9, 0.1)
# fit_result = l1_p(X = x_data,
#                   y = y_data,
#                   test_X = x_test_data,
#                   tau = tau_vec,
#                   hidden_dim1 = 4,
#                   hidden_dim2 = 4,
#                   learning_rate = 0.005,
#                   max_deep_iter = 5000,
#                   lambda_obj = 5)
#
# predicted_train = fit_result$y_predict
# predicted_valid = fit_result$y_valid_predict
# predicted_test = fit_result$y_test_predict

l1_p = function(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, learning_rate, max_deep_iter, lambda_obj, penalty)
{
  input_dim = ncol(X)
  n = nrow(X)
  r = length(tau)
  p = hidden_dim2 + 1
  tau_mat = matrix(rep(tau, each = n), ncol = 1)

  input_x = tf$placeholder(tf$float32, shape(NULL, input_dim))
  output_y = tf$placeholder(tf$float32, shape(NULL, 1))
  output_y_tiled = tf$tile(output_y, shape(r, 1))
  tau_tf = tf$placeholder(tf$float32, shape(n * r, 1))

  ### layer 1
  hidden_theta_1 = tf$Variable(tf$random_normal(shape(input_dim, hidden_dim1)))
  hidden_bias_1 = tf$Variable(tf$random_normal(shape(hidden_dim1)))
  hidden_layer_1 = tf$nn$sigmoid(tf$matmul(input_x, hidden_theta_1) + hidden_bias_1)

  ### layer 2
  hidden_theta_2 = tf$Variable(tf$random_normal(shape(hidden_dim1, hidden_dim2)))
  hidden_bias_2 = tf$Variable(tf$random_normal(shape(hidden_dim2)))
  feature_vec = tf$nn$sigmoid(tf$matmul(hidden_layer_1, hidden_theta_2) + hidden_bias_2) ##

  ### output layer
  delta_coef_mat = tf$Variable(tf$random_normal(shape(hidden_dim2, r)))
  delta_0_mat = tf$Variable(tf$random_normal(shape(1, r)))

  delta_mat = tf$concat(list(delta_0_mat, delta_coef_mat), axis = 0L)
  beta_mat = tf$transpose(tf$cumsum(tf$transpose(delta_mat)))

  delta_vec = delta_mat[2:p, 2:r]
  delta_0_vec = delta_mat[1, 2:r ,drop = F]
  delta_minus_vec = tf$maximum(0, -delta_vec)
  delta_minus_vec_sum = tf$reduce_sum(delta_minus_vec, 0L)
  delta_0_vec_clipped = tf$clip_by_value(delta_0_vec,
                                                     clip_value_min = tf$reshape(delta_minus_vec_sum, shape(nrow(delta_0_vec), ncol(delta_0_vec))),
                                                     clip_value_max = matrix(Inf, nrow(delta_0_vec), ncol(delta_0_vec)))

  #### optimization
  delta_constraint = delta_0_vec_clipped - delta_minus_vec_sum
  delta_clipped = tf$clip_by_value(delta_constraint, clip_value_min = 10e-20, clip_value_max = Inf)

  predicted_y_modified = tf$matmul(feature_vec, beta_mat[2:p, ]) +
    tf$cumsum(tf$concat(list(beta_mat[1, 1, drop = F], delta_0_vec_clipped), axis = 1L), axis = 1L)
  predicted_y = tf$matmul(feature_vec, beta_mat[2:p, ]) + beta_mat[1, ]
  predicted_y_tiled = tf$reshape(tf$transpose(predicted_y), shape(n * r, 1))

  diff_y = output_y_tiled - predicted_y_tiled
  quantile_loss = tf$reduce_mean(diff_y * (tau_tf - (tf$sign(-diff_y) + 1)/2 ))

  objective_fun = quantile_loss +
    penalty * (tf$reduce_mean(hidden_theta_1^2) + tf$reduce_mean(hidden_theta_2^2) +
                 tf$reduce_mean(delta_coef_mat^2)) +
    lambda_obj * tf$reduce_mean(tf$abs(delta_0_vec - delta_0_vec_clipped))

  train_opt = tf$train$RMSPropOptimizer(learning_rate = learning_rate)$minimize(objective_fun)

  sess = tf$Session()
  sess$run(tf$global_variables_initializer())

  tmp_vec = numeric(max_deep_iter)
  for(step in 1:max_deep_iter)
  {
    sess$run(train_opt,
             feed_dict = dict(input_x = X,
                              output_y = y,
                              tau_tf = tau_mat))
  }

  y_predict = sess$run(predicted_y_modified, feed_dict = dict(input_x = X))
  y_test_predict = sess$run(predicted_y_modified, feed_dict = dict(input_x = test_X))
  y_valid_predict = sess$run(predicted_y_modified, feed_dict = dict(input_x = valid_X))

  sess$close()
  barrier_result = list(y_predict = y_predict, y_valid_predict = y_valid_predict, y_test_predict = y_test_predict)
  return(barrier_result)
}

