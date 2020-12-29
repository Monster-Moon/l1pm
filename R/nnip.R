
#' nnip
#' @description Estimates non crossing quantile regression with a neural network using interior point method
#' @param X train predictor data
#' @param y train response data
#' @param test_X test predictor data
#' @param valid_X validation predictor data
#' @param tau target quantiles
#' @param hidden_dim1 the number of nodes in the first hidden layer
#' @param hidden_dim2 the number of nodes in the second hidden layer
#' @param learning_rate learning rate in the optimization process
#' @param max_deep_iter the number of iterations
#' @return y_predicted, y_test_predicted, y_valid_predited

nnip = function(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, learning_rate, max_deep_iter)
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
  hidden_layer_1 = tf$nn$sigmoid(tf$matmul(input_x, hidden_theta_1) + hidden_bias_1) ## 
  
  ### layer 2
  hidden_theta_2 = tf$Variable(tf$random_normal(shape(hidden_dim1, hidden_dim2)))
  hidden_bias_2 = tf$Variable(tf$random_normal(shape(hidden_dim2)))
  feature_vec = tf$nn$sigmoid(tf$matmul(hidden_layer_1, hidden_theta_2) + hidden_bias_2) ##
  
  ### output layer
  delta_mat = tf$Variable(tf$random_normal(shape(p, r)))
  delta_plus_mat = tf$maximum(0, delta_mat)
  delta_minus_mat = tf$maximum(0, -delta_mat)
  
  beta_mat = tf$transpose(tf$cumsum(tf$transpose(delta_mat)))
  delta_coef_mat = delta_mat[2:nrow(delta_mat), 2:ncol(delta_mat)]
  delta_0_vec = delta_mat[1, 2:ncol(delta_mat), drop = F]
  
  delta_minus_mat_constraint = delta_minus_mat[2:nrow(delta_mat), 2:ncol(delta_mat)]
  delta_minus_mat_constraint_sum = tf$reduce_sum(delta_minus_mat_constraint, 0L)
  
  delta_constraint = delta_0_vec - delta_minus_mat_constraint_sum
  delta_clipped = tf$clip_by_value(delta_constraint, clip_value_min = 10e-20, clip_value_max = Inf)
  
  #### initial point step
  initial_s_maximum = tf$reduce_max(-delta_0_vec + delta_minus_mat_constraint_sum)
  initial_s = tf$Variable(
    tf$random_uniform(shape(1L), 
                      minval = tf$ceil(initial_s_maximum),
                      maxval = tf$ceil(initial_s_maximum) + 0.05))
  
  initial_obj_fun = initial_s - tf$reduce_sum(tf$log(initial_s + delta_0_vec - delta_minus_mat_constraint_sum))
  initial_train_opt = tf$train$RMSPropOptimizer(learning_rate = 0.01)$minimize(initial_obj_fun)
  
  #### optimization
  M = 10^10
  penalty_term = -tf$reduce_sum(tf$log(delta_clipped)) * 1/M
  
  predicted_y = tf$matmul(feature_vec, beta_mat[2:p, ]) + beta_mat[1, ]
  predicted_y_tiled = tf$reshape(tf$transpose(predicted_y), shape(n * r, 1))
  
  diff_y = output_y_tiled - predicted_y_tiled
  quantile_loss = tf$reduce_mean(diff_y * (tau_tf - (tf$sign(-diff_y) + 1)/2 ))
  
  objective_fun = quantile_loss + penalty_term
  delta_mat_gradient = tf$gradients(objective_fun, delta_mat)
  
  train_opt = tf$train$GradientDescentOptimizer(learning_rate = learning_rate)$minimize(objective_fun)
  
  sess = tf$Session()
  sess$run(tf$global_variables_initializer())
  max_iter_initial = 5000
  for(initial_step in 1:max_iter_initial)
  {
    sess$run(initial_train_opt)
    if(sess$run(initial_s) < 0) break
    if(initial_step == max_iter_initial) cat('initilize failed \n')
  }
  
  for(step in 1:max_deep_iter)
  {
    curr_delta_mat = sess$run(delta_mat)
    sess$run(train_opt, feed_dict = dict(input_x = X, output_y = y, tau_tf = tau_mat))
    next_delta_constraint = sess$run(delta_constraint)
    learning_rate_mat = matrix(learning_rate, nrow = nrow(delta_mat), ncol = ncol(delta_mat))
    while(any(next_delta_constraint < 0))
    {
      learning_rate_mat[, 1 + which(next_delta_constraint < 0)] = 0
      sess$run(delta_mat$assign(curr_delta_mat))
      if(all(learning_rate_mat[,-1] == 0))
      {
        break
      }
      curr_delta_mat_gradient = sess$run(delta_mat_gradient, feed_dict = dict(input_x = X, output_y = y, tau_tf = tau_mat))[[1]]
      next_delta_mat = curr_delta_mat - learning_rate_mat * curr_delta_mat_gradient
      sess$run(delta_mat$assign(next_delta_mat))
      next_delta_constraint = sess$run(delta_constraint)
    }
    
    if(all(learning_rate_mat[,-1] == 0))
    {
      break
    }
  }
  y_predict = sess$run(predicted_y, feed_dict = dict(input_x = X))
  y_test_predict = sess$run(predicted_y, feed_dict = dict(input_x = test_X))
  y_valid_predict = sess$run(predicted_y, feed_dict = dict(input_x = valid_X))
  
  sess$close()
  barrier_origin_result = list(y_predict = y_predict, y_valid_predict = y_valid_predict, y_test_predict = y_test_predict)
  return(barrier_origin_result)
}