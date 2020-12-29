
#' nnqp
#' @description Estimates non crossing quantile regression with a neural network using projected gradient method
#' @param X Train predictor data
#' @param y Train response data
#' @param test_X Test predictor data
#' @param valid_X Validation predictor data
#' @param tau Target quantiles
#' @param hidden_dim1 The number of nodes in the first hidden layer
#' @param hidden_dim2 The number of nodes in the second hidden layer
#' @param learning_rate Learning rate in the optimization process
#' @param max_deep_iter The number of iterations
#' @param penalty The value of tuning parameter for ridge penalty on weights
#' @import tensorflow
#' @import quadprog solve.QP
#' @return y_predicted, y_test_predicted, y_valid_predited

nnqp = function(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, learning_rate, max_deep_iter, penalty = 0.05)
{
  input_dim = ncol(X)
  n = nrow(X)
  r = length(tau)
  tau_mat = matrix(rep(tau, each = n), ncol = 1)
  p = hidden_dim2 + 1

  input_x = tf$placeholder(tf$float32, shape(NULL, input_dim))
  output_y = tf$placeholder(tf$float32, shape(NULL, 1))
  output_y_tiled = tf$tile(output_y, shape(r, 1))
  tau_tf = tf$placeholder(tf$float32, shape(n * r, 1))

  ### layer 1
  hidden_theta_1 = tf$Variable(tf$random_normal(shape(input_dim, hidden_dim1)))
  hidden_bias_1 = tf$Variable(tf$random_normal(shape(hidden_dim1)))
  hidden_layer_1 = tf$nn$sigmoid(tf$matmul(input_x, hidden_theta_1) + hidden_bias_1) ##

  ### layer 2
  hidden_theta_2 = tf$Variable(tf$random_normal(shape(hidden_dim1, hidden_dim2)))
  hidden_bias_2 = tf$Variable(tf$random_normal(shape(hidden_dim2)))
  feature_vec = tf$nn$sigmoid(tf$matmul(hidden_layer_1, hidden_theta_2) + hidden_bias_2) ##

  ### output layer
  delta_coef_mat = tf$Variable(tf$random_normal(shape(hidden_dim2, r)))
  delta_0_mat = tf$Variable(tf$random_normal(shape(1, r)))

  delta_mat = tf$concat(list(delta_0_mat, delta_coef_mat), axis = 0L)
  beta_mat = tf$transpose(tf$cumsum(tf$transpose(delta_mat)))

  delta_vec = delta_mat[2:p, 2:ncol(delta_mat)]
  delta_0_vec = delta_mat[1, 2:ncol(delta_mat) ,drop = FALSE]
  delta_minus_vec = tf$maximum(0, -delta_vec)
  delta_minus_vec_sum = tf$reduce_sum(delta_minus_vec, 0L)
  delta_constraint = delta_0_vec - delta_minus_vec_sum

  predicted_y = tf$matmul(feature_vec, beta_mat[2:p, ]) + beta_mat[1, ]
  predicted_y_tiled = tf$reshape(tf$transpose(predicted_y), shape(n * r, 1))

  diff_y = output_y_tiled - predicted_y_tiled
  quantile_loss = tf$reduce_mean(diff_y * (tau_tf - (tf$sign(-diff_y) + 1)/2 )) +
    penalty * (tf$reduce_sum(hidden_theta_1^2) + tf$reduce_sum(hidden_theta_2^2) +
                 tf$reduce_sum(delta_coef_mat^2))
  train_opt = tf$train$RMSPropOptimizer(learning_rate = learning_rate)$minimize(quantile_loss) ## optimizer

  #### tensorflow session ####
  sess = tf$Session()
  sess$run(tf$global_variables_initializer())

  curr_Dmat = diag(1, 2 * r * p , 2 * r * p)
  curr_tau1_plus_Amat = matrix(0, nrow = r-1, ncol = p)
  curr_plus_Amat = do.call('rbind', lapply(1:(r-1), function(i) c(rep(0, (i-1)*p), c(1, rep(0, p-1)), rep(0, (r -i -1) * p))))
  curr_tau1_minus_Amat = matrix(0, nrow = r-1, ncol = p)
  curr_minus_Amat = do.call('rbind', lapply(1:(r-1), function(i) c(rep(0, (i-1) * p), c(rep(-1, p)), rep(0, (r-i -1) * p))))

  curr_Amat = cbind(curr_tau1_plus_Amat, curr_plus_Amat, curr_tau1_minus_Amat, curr_minus_Amat)
  curr_Amat = rbind(curr_Amat, curr_Dmat)
  for(step in 1:max_deep_iter)
  {
    sess$run(train_opt,
             feed_dict = dict(input_x = X,
                              output_y = y,
                              tau_tf = tau_mat))

    if(any(sess$run(delta_constraint) < 0))
    {
      curr_delta_mat = sess$run(delta_mat)
      curr_delta_vec = as.numeric(curr_delta_mat)
      curr_delta_plus_vec = pmax(0, curr_delta_vec)
      curr_delta_minus_vec = pmax(0, -curr_delta_vec)

      curr_dvec = c(curr_delta_plus_vec, curr_delta_minus_vec)
      solve_qp = solve.QP(Dmat = curr_Dmat, dvec = curr_dvec, Amat = t(curr_Amat))
      qp_solution = solve_qp$solution
      qp_delta_plus_vec = qp_solution[1:(r * p)]
      qp_delta_minus_vec = qp_solution[-(1:(r*p))]
      qp_delta_vec = qp_delta_plus_vec - qp_delta_minus_vec
      qp_delta_mat = matrix(qp_delta_vec, ncol = r)

      sess$run(delta_0_mat$assign(qp_delta_mat[1, , drop = F]))
      sess$run(delta_coef_mat$assign(qp_delta_mat[-1,]))
    }
  }
  y_predict = sess$run(predicted_y, feed_dict = dict(input_x = X))
  y_test_predict = sess$run(predicted_y, feed_dict = dict(input_x = test_X))
  y_valid_predict = sess$run(predicted_y, feed_dict = dict(input_x = valid_X))

  sess$close()
  qp_result = list(y_predict = y_predict, y_valid_predict = y_valid_predict, y_test_predict = y_test_predict)
  return(qp_result)
}

