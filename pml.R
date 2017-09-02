library('ggplot2') # visualisation
library('scales') # visualisation
library('grid') # visualisation
library('RColorBrewer') # visualisation
library('corrplot') # visualisation
library('alluvial') # visualisation
library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('lubridate') # date and time
library('geosphere') # geospatial locations
library('leaflet') # maps
library('leaflet.extras') # maps
library('maps') # maps
library('xgboost') # modelling
library('caret') # modelling
library('plyr')


#read data
train <- as.tibble(fread('~/Documents/coursera/machinelearning/week4/pml-training.csv',na.strings=c("NA","#DIV/0!","")))
test <- as.tibble(fread('~/Documents/coursera/machinelearning/week4/pml-testing.csv',na.strings=c("NA","#DIV/0!","")))

test$classe<-0
test$classe<-as.factor(test$classe)
train[train$classe=="A",]$classe<-1
train[train$classe=="B",]$classe<-2
train[train$classe=="C",]$classe<-3
train[train$classe=="D",]$classe<-4
train[train$classe=="E",]$classe<-5
train$classe<-as.factor(train$classe)

#forget the first 6 columns, they're not predictors
train <- train[,7:length(colnames(train))]
#remove near-zero variance columns
near_zero_var_comb<- nearZeroVar(train[], saveMetrics=TRUE)
train<- train[,near_zero_var_comb$nzv==FALSE]

#remove columns with >=50% NAs
na_Cols <- as.vector(apply(train, 2, function(train) length(which(!is.na(train)))))

# Drop columns that have more than 50% NAs
dropNAs <- c("problem_id")
for (i in 1:length(na_Cols)) {
  if (na_Cols[i] >= nrow(train)*.50) {
    dropNAs <- c(dropNAs, colnames(train)[i])
  }
}
test<- test[,(names(test) %in% dropNAs)]
train <- train[,(names(train) %in% dropNAs)]


trainIndex <- createDataPartition(train$classe, p = 0.8, list = FALSE, times = 1)

train <- train[trainIndex,]
valid <- train[-trainIndex,]
foo <- train %>% select(-classe)
bar <- valid %>% select(-classe)
test<-test %>% select(-classe)
dtrain <- xgb.DMatrix(as.matrix(foo),label = train$classe)
dvalid <- xgb.DMatrix(as.matrix(bar),label = valid$classe)
dtest <- xgb.DMatrix(as.matrix(test))

xgb_params <- list(colsample_bytree = .8, #variables per tree 
                   subsample = 0.75, #data subset per tree 
                   min_child_weight=10,
                   booster = "gbtree",
                   max_depth = 10, #tree levels
                   eta = 0.1, #shrinkage
                   eval_metric = "rmse", 
                   objective = "reg:linear",
                   seed = 4321
)


ntrees<-500

watchlist <- list(train=dtrain, valid=dvalid)
set.seed(4321)
gb_dt <- xgb.train(params = xgb_params,
                   data = dtrain,
                   print_every_n = 10,
                   watchlist = watchlist,
                   early_stopping_rounds = 10,
                   nrounds = ntrees)

xgb_cv <- xgb.cv(xgb_params,dtrain,early_stopping_rounds = 10, nfold = 5, nrounds=ntrees, print_every_n = 10)

imp_matrix <- as.tibble(xgb.importance(feature_names = colnames(train %>% select(-classe)), model = gb_dt))
imp_matrix %>%
  ggplot(aes(reorder(Feature, Gain, FUN = max), Gain, fill = Feature)) +
  geom_col() +
  coord_flip() +
  theme(legend.position = "none") +
  labs(x = "Features", y = "Importance")

test_preds <- round(predict(gb_dt,dtest))
test_preds[test_preds==1]<-"A"
test_preds[test_preds==2]<-"B"
test_preds[test_preds==3]<-"C"
test_preds[test_preds==4]<-"D"
test_preds[test_preds==5]<-"E"
test_preds