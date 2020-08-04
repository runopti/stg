library(devtools)
setwd("/Users/yutaro/code/stg/")
install('Rstg')
setwd("/Users/yutaro/code/stg/Rstg")
document()

n_size <- 1000L;
p_size <- 20L;

#model = STG(task_type='regression',input_dim=X_train.shape[1], output_dim=1, hidden_dims=[500, 50, 10], activation='tanh',
#    optimizer='SGD', learning_rate=0.1, batch_size=X_train.shape[0], feature_selection=feature_selection, sigma=0.5, lam=0.1, random_state=1, device=device) 

result <- stg(1, task_type='regression', input_dim=p_size, output_dim=1L, hidden_dims = c(500,50, 10), activation='tanh',
            optimizer='SGD', learning_rate=0.1, batch_size=n_size, feature_selection=TRUE, sigma=0.5, lam=0.1, random_state=0.1)


result <- stg(1, task_type='cox',input_dim=train_data['X'].shape[1], output_dim=1, hidden_dims=[60, 20, 3], activation='selu',
    optimizer='Adam', learning_rate=0.0005, batch_size=train_data['X'].shape[0], feature_selection=True, 
    sigma=0.5, lam=0.004, random_state=1, device='cpu')

#pystg <- result$pystg 
gg <- result$pystg$utils$load_(n_size, p_size);