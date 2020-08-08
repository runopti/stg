library(devtools)
setwd("/Users/yutaro/code/stg/")
install('Rstg')
setwd("/Users/yutaro/code/stg/Rstg")
document()

reticulate::use_python("/Users/yutaro/.pyenv/versions/3.7.0/envs/test-devR/bin/python3.7")
reticulate::import("stg")

#model = STG(task_type='regression',input_dim=X_train.shape[1], output_dim=1, hidden_dims=[500, 50, 10], activation='tanh',
#    optimizer='SGD', learning_rate=0.1, batch_size=X_train.shape[0], feature_selection=feature_selection, sigma=0.5, lam=0.1, random_state=1, device=device) 

#result <- stg(1, task_type='regression', input_dim=p_size, output_dim=1L, hidden_dims = c(500,50, 10), activation='tanh',
#            optimizer='SGD', learning_rate=0.1, batch_size=n_size, feature_selection=TRUE, sigma=0.5, lam=0.1, random_state=0.1)


# result <- stg(1, task_type='cox',input_dim=train_data['X'].shape[1], output_dim=1, hidden_dims=[60, 20, 3], 
#     activation='selu', optimizer='Adam', learning_rate=0.0005, batch_size=train_data['X'].shape[0], 
#     feature_selection=True, sigma=0.5, lam=0.004, random_state=1, device='cpu')

stg.model <- stg(task_type='cox',input_dim=10L, output_dim=1L, hidden_dims=c(60, 20, 3), 
    activation='selu', optimizer='Adam', learning_rate=0.0005, batch_size=4000L, 
    feature_selection=TRUE, sigma=0.5, lam=0.004, random_state=1L, device='cpu')

#### For loading toy dataset
pystg <- reticulate::import("stg")
datasets <- pystg$utils$load_cox_gaussian_data()
####

train_data <- datasets$train
test_data <- datasets$test

# get statistics to standardize data
norm_vals <- list(
        'mean' = apply(train_data$x, 2, mean),
        'std'  = apply(train_data$x, 2, sd)
        )
# reshape for python functions
norm_vals$mean <- matrix(norm_vals$mean, byrow=TRUE, nrow=1) #length(norm_vals$mean))
norm_vals$std <- matrix(norm_vals$std, byrow=TRUE, nrow=1) #length(norm_vals$std))

# standardize data 
train_data <- pystg$utils$standardize_dataset(datasets$train, norm_vals$mean, norm_vals$std)
valid_data <- pystg$utils$standardize_dataset(datasets$valid, norm_vals$mean, norm_vals$std)
test_data <- pystg$utils$standardize_dataset(datasets$test, norm_vals$mean, norm_vals$std)

# sort the samples by the survival time for correct partial likelihood calculation
tmp <- pystg$utils$prepare_data(train_data$x, list('e'=train_data$e, 't'=train_data$t))
train_data <- list('X'=tmp$x, 'E'=tmp$e, 'T'=tmp$t, 'ties'='noties')

tmp <- pystg$utils$prepare_data(valid_data$x, list('e'=valid_data$e, 't'=valid_data$t))
valid_data <- list('X'=tmp$x, 'E'=tmp$e, 'T'=tmp$t, 'ties'='noties')

tmp <- pystg$utils$prepare_data(test_data$x, list('e'=test_data$e, 't'=test_data$t))
test_data <- list('X'=tmp$x, 'E'=tmp$e, 'T'=tmp$t, 'ties'='noties')

#### Start training
stg.model$fit(train_data$X, list('E'=train_data$E, 'T'=train_data$T), nr_epochs=600L, 
            valid_X=valid_data$X, valid_y=list('E'=valid_data$E, 'T'=valid_data$T), print_interval=100L)

stg.model$save_checkpoint('r_test_cox_model.pth')

stg.model$evaluate(test_data$X, list('E'=test_data$E, 'T'=test_data$T))

