
#' nnra
#' @description Estimates non crossing quantile regression with a neural network using rearrange method.
#' @param X train predictor data
#' @param y train response data
#' @param test_X test predictor data
#' @param valid_X validation predictor data
#' @param tau target quantiles
#' @param hidden_dim1 the number of nodes in the first hidden layer
#' @param hidden_dim2 the number of nodes in the second hidden layer
#' @param learning_rate learning rate in the optimization process
#' @param max_deep_iter the number of iterations
#' @param penalty the value of tuning parameter for ridge penalty on weights
#' @return y_predicted, y_test_predicted, y_valid_predited : predicted quantile based on train, test, and validation data, respectively

nnra = function(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, learning_rate, max_deep_iter, penalty)
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
  feature_vec = tf$nn$sigmoid(tf$matmul(hidden_layer_1, hidden_theta_2) + hidden_bias_2) ##

  ### output layer
  hidden_theta_4 = tf$Variable(tf$random_normal(shape(hidden_dim2, r)))
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


