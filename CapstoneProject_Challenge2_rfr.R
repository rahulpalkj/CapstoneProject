global_poverty_2 = read.csv(file= "C:/Users/Rahulkj/Documents/Datasets/train_values_labels.csv", header = T)
global_poverty_2$bank_interest_rate = NULL
global_poverty_2$mm_interest_rate = NULL
global_poverty_2$mfi_interest_rate = NULL
global_poverty_2$other_fsp_interest_rate = NULL
global_poverty_2$row_id = NULL

#Partition the data
partition = createDataPartition(global_poverty_2[,'education_level'], times = 1, p = 0.8, list = F)
training = global_poverty_2[partition,] # create training feature sample
dim(training)

test = global_poverty_2[-partition,] #to create test sample features
dim(test)

#Scaling for all variables to be zero mean and unit variance
numcols = c('age', 'education_level', 'share_hh_income_provided', 'num_times_borrowed_last_year', 'borrowing_recency', 'num_shocks_last_year', 'avg_shock_strength_last_year', 'phone_technology', 'phone_ownership', 'num_formal_institutions_last_year', 'num_informal_institutions_last_year', 'num_financial_activities_last_year')
preProcValues = preProcess(training[,numcols], method = c('center','scale'))

training[,numcols] = predict(preProcValues, training[, numcols])
test[, numcols] = predict(preProcValues, test[, numcols])
head(training[, numcols])

#Random Forest Regression
library(randomForest)
rf <- randomForest(poverty_probability ~., data = training)

print(rf)
attributes(rf)

#Prediction
p1 <- data.frame(predict(rf, training))
p2 <- data.frame(predict(rf, test))

head(p1$predict.rf..training.)
head(global_poverty_2$poverty_probability)
head(p2$predict.rf..test.)

actual <- test$poverty_probability
predicted <- p2$predict.rf..test.

r2 <- 1 - (sum((actual-predicted)^2)/sum((actual-mean(actual))^2))
print (r2)