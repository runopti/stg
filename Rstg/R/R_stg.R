#' STochastic Gates (STG) based COX Proportional Hazards Regression Model
#'
#' coxstg extends cox proportional hazards regression model by using
#' a non-linear neural network feature selection model
#' (STochastic Gates (STG)), as described in Yamada, et al, ICML 2020.
#'
#' @param data string
#' path to the input survival data
#' @param hidden_layers_node vector of integers,optional,default:c(60, 20, 3)
#' architecture vector of the neural network
#' @examples
#' dataset <- "/data/jyc/cox_data/dataset/gaussian/gaussian_survival_data.h5"
#' coxstg(dataset,hidden_layers_node = c(60, 30, 20, 3),num_epoch = 3000,output_dir = "~/results/")
#' @export
#'
#'

stg <- function(data, ...) {
    UseMethod("stg", data)
}

stg.default <- function(
    data,
    task_type,
    input_dim,
    output_dim,
    hidden_dims,
    activation='relu',
    sigma=0.5,
    lam=0.1,
    optimizer='Adam',
    learning_rate=0.001,
    batch_size=100,
    freeze_onward=NULL,
    feature_selection=TRUE,
    weight_decay=0.001,
    random_state = 123,
    device = 'cpu',
    ...
){
    # check installation; Do this later
    #if (!reticulate::py_module_available(module = "stg") || (is.null(stg))) load_stg()
    reticulate::use_python("/Users/yutaro/.pyenv/versions/Rstg/bin/python", required = TRUE)
    pystg <- reticulate::import("stg")
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
    
    operator <- pystg$STG(
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
    result <- list(
        "operator" = operator,
        "pystg" = pystg
    )
  return(result)
}

#print.stg <- function(x, ...){
#    cat("fefe")
#}

# coxstg <- function(data, ...) {
#     UseMethod("coxstg", data)
# }

# coxstg.default <- function(data, hidden_layers_node = c(60, 20, 3), learning_rate = 0.001, learning_rate_decay = 1.0,
#     activation = "tanh", L2_reg = 0.0, L1_reg = 0.0, optimizer = "sgd", dropout_keep_prob = 1.0, feature_selection = TRUE,
#     seed = 1, sigma = 0.5, lam = 0.005, standardize = FALSE, num_epoch = 2000, iteration = 100, num_run = 5,
#     plot_loss = TRUE, plot_CI = TRUE, output_dir = "tmp") {
#     #### check whether stg has been loaded: suppose not what if pycoxstg has been loaded but the dependency
#     #### hasn't been checked:won't happen, because the python version is installed with requirements
#     #### checked. load_pycoxstg() will try to install stg and related pkgs (if included in setup.py) via
#     #### pip, which is not possible now. Therefore, assuming that the stg python package has been
#     #### successfully installed and loaded if (!reticulate::py_module_available(module='coxtf'))
#     #### load_pycoxstg()
#     #reticulate::use_python("/gpfs/loomis/home.grace/jy568/miniconda3/bin/python3", required = TRUE)
#     reticulate::use_python("/Users/yutaro/.pyenv/versions/Rstg/bin/python", required = TRUE)
#     coxtf <- reticulate::import("coxtf")

#     #### 1.params check:

#     #### 2.store the params
#     params <- list(hidden_layers_node = hidden_layers_node, learning_rate = learning_rate, learning_rate_decay = learning_rate_decay,
#         activation = activation, L2_reg = L2_reg, L1_reg = L1_reg, optimizer = optimizer, dropout_keep_prob = dropout_keep_prob,
#         feature_selection = feature_selection, seed = seed, sigma = sigma, lam = lam, standardize = standardize)
#     #### 3.run stg
#     coxtf$pycoxstg$R_interface(dataset, params, num_epoch, iteration, num_run, plot_loss, plot_CI, output_dir)

# }