# l1-p method
Non crossing quantile regression using neural network

### Required
- Python >= 3.6
- Tensorflow >= 1.14 & < 2

### Install tutorial
```
if(!require(Rcpp)) install.packages('Rcpp'); require(Rcpp)
if(!require(devtools)) install.packages('devtools'); require(devtools)
if(!require(tensorflow)) 
{
	devtools::install_github('rstudio/tensorflow')
	install_tensorflow()
	require(tensorflow)
}	
```

### Install tutorial with conda
```
if(!require(Rcpp)) install.packages('Rcpp'); require(Rcpp)
if(!require(devtools)) install.packages('devtools'); require(devtools)
if(!require(reticulate)) install.packages('reticulate'); require(reticulate)
if(!require(tensorflow)) 
{
	devtools::install_github('rstudio/tensorflow')
	install_tensorflow(version = '1.14')
}
reticulate::conda_list()
reticulate::use_condaenv(condaenv = 'names')
tensorflow::tf_config()
require(tensorflow)
```

### Execution & results
```
devtools::install_tensorflow('Monster-Moon/l1pm')
require(l1pm)
fit_result = l1_p(X = x_data,
                  y = y_data,
                  valid_X = x_valid_data,
                  test_X = x_test_data,
                  tau = tau_vec,
                  hidden_dim1 = 4,
                  hidden_dim2 = 4,
                  learning_rate = 0.005,
                  max_deep_iter = 5000,
                  lambda_obj = 5)

predicted_test = fit_result$y_test_predict  
```
