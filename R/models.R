

if(!require(tensorflow)) install.packages("tensorflow")
if(!require(quadprog)) install.packages("quadprog")
require(tensorflow)
require(quadprog)

# if(tensorflow::tf_version() != '1.14') tensorflow::install_tensorflow(version = '1.14')


#### Projection method (PG)
nnqp = function(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, hidden_dim3 = NULL, learning_rate, max_deep_iter, penalty = 0.05)
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
  delta_0_vec = delta_mat[1, 2:ncol(delta_mat) ,drop = F]
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

    if(step %% 1000 == 0)
    {
      loss_val = sess$run(quantile_loss,
                          feed_dict = dict(input_x = X,
                                           output_y = y,
                                           tau_tf = tau_mat))

      const_val = sess$run(delta_constraint)
      cat(step, "step's loss :", loss_val , "\n")
      cat(step, "step's constraint :", const_val, "\n")
    }
  }
  y_predict = sess$run(predicted_y, feed_dict = dict(input_x = X))
  y_test_predict = sess$run(predicted_y, feed_dict = dict(input_x = test_X))
  y_valid_predict = sess$run(predicted_y, feed_dict = dict(input_x = valid_X))

  sess$close()
  qp_result = list(y_predict = y_predict, y_valid_predict = y_valid_predict, y_test_predict = y_test_predict)
  return(qp_result)
}

#### Interior point (IP)
nnip = function(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, hidden_dim3 = NULL, learning_rate, max_deep_iter, max_initial_iter)
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

  # delta_minus_mat_constraint = tf$maximum(0, -delta_coef_mat)
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

    if(step %% 1000 == 0)
    {
      cat(step, 'th constraint', sess$run(delta_constraint), '\n')
      cat(step, 'th loss', sess$run(objective_fun, feed_dict = dict(input_x = X, output_y = y, tau_tf = tau_mat)), '\n')
    }
  }
  y_predict = sess$run(predicted_y, feed_dict = dict(input_x = X))
  y_test_predict = sess$run(predicted_y, feed_dict = dict(input_x = test_X))
  y_valid_predict = sess$run(predicted_y, feed_dict = dict(input_x = valid_X))

  sess$close()
  barrier_origin_result = list(y_predict = y_predict, y_valid_predict = y_valid_predict, y_test_predict = y_test_predict)
  return(barrier_origin_result)
}

#### Rearrange method (RA)
nnra = function(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, hidden_dim3 = NULL, learning_rate, max_deep_iter, penalty)
{
  input_dim = ncol(X)
  n = nrow(X)
  r = length(tau)
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
  # hidden_layer_2 = tf$nn$sigmoid(tf$matmul(hidden_layer_1, hidden_theta_2) + hidden_bias_2) ##
  feature_vec = tf$nn$sigmoid(tf$matmul(hidden_layer_1, hidden_theta_2) + hidden_bias_2) ##

  ### layer 3
  # hidden_theta_3 = tf$Variable(tf$random_normal(shape(hidden_dim2, hidden_dim3)))
  # hidden_bias_3 = tf$Variable(tf$random_normal(shape(hidden_dim3)))
  # feature_vec = tf$nn$sigmoid(tf$matmul(hidden_layer_2, hidden_theta_3) + hidden_bias_3) ##

  ### output layer
  hidden_theta_4 = tf$Variable(tf$random_normal(shape(hidden_dim2, r)))
  # hidden_theta_4 = tf$Variable(tf$random_normal(shape(hidden_dim3, r)))
  hidden_bias_4 = tf$Variable(tf$random_normal(shape(r)))

  predicted_y = tf$matmul(feature_vec, hidden_theta_4) + hidden_bias_4
  predicted_y_tiled = tf$reshape(tf$transpose(predicted_y), shape(n * r, 1))


  diff_y = output_y_tiled - predicted_y_tiled
  quantile_loss = tf$reduce_mean(diff_y * (tau_tf - (tf$sign(-diff_y) + 1)/2 )) +
    penalty * (tf$reduce_sum(hidden_theta_1^2) + tf$reduce_sum(hidden_theta_2^2) +
                 tf$reduce_sum(hidden_bias_1^2) + tf$reduce_sum(hidden_bias_2^2) +
                 tf$reduce_sum(hidden_theta_4^2) + tf$reduce_sum(hidden_bias_4^2))
  train_opt = tf$train$RMSPropOptimizer(learning_rate = learning_rate)$minimize(quantile_loss) ## optimizer

  #### tensorflow session ####
  sess = tf$Session()
  sess$run(tf$global_variables_initializer())
  for(step in 1:max_deep_iter)
  {
    sess$run(train_opt,
             feed_dict = dict(input_x = X,
                              output_y = y,
                              tau_tf = tau_mat))
    if(step %% 1000 == 0)
    {
      loss_val = sess$run(quantile_loss,
                          feed_dict = dict(input_x = X,
                                           output_y = y,
                                           tau_tf = tau_mat))
      cat(step, "step's loss :", loss_val , "\n")
    }
  }

  y_predict = sess$run(predicted_y, feed_dict = dict(input_x = X))
  y_test_predict = sess$run(predicted_y, feed_dict = dict(input_x = test_X))
  y_valid_predict = sess$run(predicted_y, feed_dict = dict(input_x = valid_X))

  y_predict = apply(y_predict, 1, sort) %>% t()
  y_test_predict = apply(y_test_predict, 1, sort) %>% t()
  y_valid_predict = apply(y_valid_predict, 1, sort) %>% t()

  sess$close()
  barrier_result = list(y_predict = y_predict, y_valid_predict = y_valid_predict, y_test_predict = y_test_predict)
  return(barrier_result)
}

#### L_1 penalization method (l_1)
l1_p = function(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, hidden_dim3 = NULL, learning_rate, max_deep_iter, lambda_obj, penalty)
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
  delta_0_vec_clipped = tf$clip_by_value(delta_0_vec, clip_value_min = tf$reshape(delta_minus_vec_sum, shape(nrow(delta_0_vec), ncol(delta_0_vec))),
                                         clip_value_max = matrix(Inf, nrow(delta_0_vec), ncol(delta_0_vec)))

  #### optimization
  # t_val = tf$placeholder(tf$float32, shape(1L, 1L))
  delta_constraint = delta_0_vec_clipped - delta_minus_vec_sum
  delta_clipped = tf$clip_by_value(delta_constraint, clip_value_min = 10e-20, clip_value_max = Inf)
  # penalty_term = -tf$reduce_sum(tf$log(delta_clipped)) * 1/t_val

  predicted_y_modified = tf$matmul(feature_vec, beta_mat[2:p, ]) +
    tf$cumsum(tf$concat(list(beta_mat[1, 1, drop = F], delta_0_vec_clipped), axis = 1L), axis = 1L)
  predicted_y = tf$matmul(feature_vec, beta_mat[2:p, ]) + beta_mat[1, ]
  predicted_y_tiled = tf$reshape(tf$transpose(predicted_y), shape(n * r, 1))

  diff_y = output_y_tiled - predicted_y_tiled
  quantile_loss = tf$reduce_mean(diff_y * (tau_tf - (tf$sign(-diff_y) + 1)/2 ))

  objective_fun = quantile_loss +
    # penalty_term +
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
    # t_val = matrix(M, 1, 1)))
  }

  y_predict = sess$run(predicted_y_modified, feed_dict = dict(input_x = X))
  y_test_predict = sess$run(predicted_y_modified, feed_dict = dict(input_x = test_X))
  y_valid_predict = sess$run(predicted_y_modified, feed_dict = dict(input_x = valid_X))

  sess$close()
  barrier_result = list(y_predict = y_predict, y_valid_predict = y_valid_predict, y_test_predict = y_test_predict)
  return(barrier_result)
}

