library(devtools)
setwd("/Users/yutaro/code/stg/")
install('Rstg')
setwd("/Users/yutaro/code/stg/Rstg")
document()

library(tidyverse)

n_size <- 1000L;
p_size <- 20L;

#model = STG(task_type='regression',input_dim=X_train.shape[1], output_dim=1, hidden_dims=[500, 50, 10], activation='tanh',
#    optimizer='SGD', learning_rate=0.1, batch_size=X_train.shape[0], feature_selection=feature_selection, sigma=0.5, lam=0.1, random_state=1, device=device) 

result <- stg(1, task_type='regression', input_dim=p_size, output_dim=1L, hidden_dims = c(500,50, 10), activation='tanh',
            optimizer='SGD', learning_rate=0.1, batch_size=n_size, feature_selection=TRUE, sigma=0.5, lam=0.1, random_state=0.1)

# Regression
gg <- result$pystg$utils$create_sin_dataset(n_size, p_size);
print(dim(gg[[1]]))
print(class(gg[[1]]))
print(dim(gg[[2]]))
print(class(gg[[2]]))
 
df_X <- as.data.frame(gg[[1]])
df_y <- as.data.frame(gg[[2]])
df_y <- setNames(df_y, c('Y'))
df <- cbind(df_X, df_y)
gp <- ggplot(data=df, mapping=aes(x=V1,y=V2, color=Y)) + geom_point(alpha=0.4, size=5) 
#gp <- gp + scale_color_brewer(palette="RdYlBu")
gp <- gp + scale_color_gradientn(colours = rainbow(5))
plot(gp)

# Classification
gg <- result$pystg$utils$create_twomoon_dataset(n_size, p_size);
print(dim(gg[[1]]))
print(class(gg[[1]]))
print(dim(gg[[2]]))
print(class(gg[[2]]))
 
df_X <- as.data.frame(gg[[1]])
df_y <- as.data.frame(gg[[2]])
df_y <- setNames(df_y, c('Y'))
df <- cbind(df_X, df_y)
gp <- ggplot(data=df, mapping=aes(x=V1,y=V2, color=Y)) + geom_point(alpha=0.4, size=5) 
#gp <- gp + scale_color_brewer(palette="RdYlBu")
gp <- gp + scale_color_gradientn(colours = rainbow(5))
plot(gp)

