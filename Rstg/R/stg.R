#' STG: Feature Selection using Stochastic Gates 
#'
#' STG is a method for feature selection in neural network estimation problems.
#' The new procedure is based on probabilistic relaxation of the l0 norm of
#' features, or the count of the number of selected features.
#' STG simultaneously learns either a nonlinear regression
#' or classification function while selecting a small subset of features,
#' as described in Yamada, et al, ICML 2020.
#'
#' @param task_type string
#' choose 'regression', 'classification', or 'cox'
#' @param input_dim integer
#' The number of features of your data (input dimension)
#' @param output_dim integer
#' The number of classes for 'classification'. Should be 1 for 'regression' and 'cox'
#' @param hidden_dims vector of integers,optional,default:c(60, 20, 3)
#' architecture vector of the neural network
#' @param activation string
#' the type of activation functions.
#' @param sigma float
#' the noise level for the gaussian distribution
#' @param lam float
#' the regularization parameter
#' @param optimizer string
#' choose 'Adam' or 'SGD'
#' @param learning_rate float
#' @param batch_size int
#' @param freeze_onward integer, default:NULL
#' the network parameters will be frozen after 'freeze_onward' epoch.
#' This is to train the gate parameters.
#' @param feature_selection bool
#' @param weight_decay float
#' @param random_state integer
#' @param device string
#' 'cpu' or 'cuda' (if you have GPU)
#' @param ... Arguments passed to and from other methods
#' 
#' @export
#' 
stg <- function(...) {
    UseMethod("stg")
}

#' @rdname stg
#' @export
#'
stg.default <- function(
    task_type,
    input_dim,
    output_dim,
    hidden_dims,
    activation="relu",
    sigma=0.5,
    lam=0.1,
    optimizer="Adam",
    learning_rate=0.001,
    batch_size=100L,
    freeze_onward=NULL,
    feature_selection=TRUE,
    weight_decay=0.001,
    random_state = 123L,
    device = "cpu",
    ...
){
    # check installation; Do this later
    if (!reticulate::py_module_available(module = "stg") || (is.null(pystg))) load_pystg()
    #reticulate::use_python("/Users/yutaro/.pyenv/versions/Rstg/bin/python", required = TRUE)
    #reticulate::use_python("/Users/yutaro/.pyenv/versions/3.7.0/envs/Rstg/bin/python3.7", required = TRUE)
    #load_pystg()
    #pystg <- reticulate::import("stg")
    if (is.numeric(x = freeze_onward)){
        freeze_onward <- as.integer(x = freeze_onward)
    }
    input_dim <- as.integer(x = input_dim)
    output_dim <- as.integer(x = output_dim)
    #for(i in seq_along(hidden_dims)) {
    #    hidden_dims[[i]] <- as.integer(x = hidden_dims[[i]])
    #}
    hidden_dims <- as.integer(hidden_dims)
    random_state <- as.integer(x = random_state)
    batch_size <- as.integer(x = batch_size)
    
    model <- pystg$STG(
        task_type = task_type,
        input_dim = input_dim,
        output_dim = output_dim,
        hidden_dims = hidden_dims,
        activation = activation,
        optimizer = optimizer,
        learning_rate = learning_rate,
        batch_size = batch_size,
        feature_selection = feature_selection,
        sigma = sigma,
        lam = lam,
        random_state = random_state,
        device = device
    )
  return(model)
}