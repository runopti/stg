library(devtools)
setwd("/Users/yutaro/code/stg/")
install('Rstg')
setwd("/Users/yutaro/code/stg/Rstg")
document()

n_size <- 1000L;
p_size <- 20L;

#reticulate::py_config()

stg.model <- stg(task_type='regression', input_dim=p_size, output_dim=1L, hidden_dims = c(500,50, 10), activation='tanh',
            optimizer='SGD', learning_rate=0.1, batch_size=n_size, feature_selection=TRUE, sigma=0.5, lam=0.1, random_state=0.1)