library(devtools)
setwd("/Users/yutaro/code/stg/")
install('Rstg')
setwd("/Users/yutaro/code/stg/Rstg")
document()

n_size <- 1000L;
p_size <- 20L;

result <- stg(1, task_type='regression', input_dim=p_size, output_dim=1L, hidden_dims = c(500,50, 10), activation='tanh',
            optimizer='SGD', learning_rate=0.1, batch_size=n_size, feature_selection=TRUE, sigma=0.5, lam=0.1, random_state=0.1)

#pystg <- result$pystg 
#gg <- result$pystg$utils$create_sin_dataset(n_size, p_size);

result$operator$load_checkpoint('/Users/yutaro/code/stg/Rstg/r_test_model.pth')

#result$operator$predict()
