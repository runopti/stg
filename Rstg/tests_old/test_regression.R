library(devtools)
setwd("/Users/yutaro/code/stg/")
install('Rstg')
setwd("/Users/yutaro/code/stg/Rstg")
document()

n_size <- 1000L;
p_size <- 20L;

stg.model <- stg(task_type='regression', input_dim=p_size, output_dim=1L, hidden_dims = c(500,50, 10), activation='tanh',
            optimizer='SGD', learning_rate=0.1, batch_size=n_size, feature_selection=TRUE, sigma=0.5, lam=0.1, random_state=0.1)

## For creating simple datasets
pystg <- reticulate::import("stg")
datasets <- pystg$utils$create_sin_dataset(n_size, p_size);
print(dim(datasets[[1]]))
print(class(datasets[[1]]))

print(dim(datasets[[2]]))
print(class(datasets[[2]]))

train_test_split <- function(X, y, test_size){
    set.seed(12345)
    #getting training data set sizes of .20 (in this case 20 out of 100)
    test_idx <- sample(seq_len(length(y)), ceiling(length(y) * test_size))

    # Split data into test/train using indices
    X_test <- X[test_idx, ]
    y_test <- y[test_idx]
    X_train <- X[-test_idx, ]
    y_train <- y[-test_idx]
    return(list(X_train, X_test,
    y_train, y_test))
}

X_data <- datasets[[1]]
y_data <- datasets[[2]]


res_tmp = train_test_split(X_data, y_data, test_size=0.1)
X_train <- res_tmp[[1]];
X_test <- res_tmp[[2]];
y_train <- res_tmp[[3]];
y_test <- res_tmp[[4]];
print(dim(y_train))
print(class(y_train))
print(dim(y_test))
print(class(y_test))
y_test <- matrix(unlist(y_test), byrow=TRUE, nrow=length(y_test) ) 
print(dim(y_test))
print(class(y_test))
print(dim(X_train))
print(class(X_train))
res_list = train_test_split(X_train, y_train, test_size=0.2)
X_train <- res_list[[1]];
X_valid <- res_list[[2]];
y_train <- res_list[[3]];
y_valid <- res_list[[4]];

y_test <- matrix(unlist(y_test), byrow = TRUE, nrow = length(y_test))
y_train <- matrix(unlist(y_train), byrow = TRUE, nrow = length(y_train))
y_valid <- matrix(unlist(y_valid), byrow = TRUE, nrow = length(y_valid))
#y_train <- reticulate::np_array(data = y_train)
#y_test <- reticulate::np_array(data = y_test)
#y_valid <- reticulate::np_array(data = y_valid)

stg.model$fit(X_train, y_train, nr_epochs=5000L, valid_X=X_valid, valid_y=y_valid, print_interval=1000L)

stg.model$operator$save_checkpoint('r_test_model.pth')