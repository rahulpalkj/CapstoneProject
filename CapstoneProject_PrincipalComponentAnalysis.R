global_poverty_2 = read.csv(file= "C:/Users/Rahulkj/Documents/Datasets/train_values_labels.csv", header = T)

global_poverty_2$bank_interest_rate = NULL
global_poverty_2$mm_interest_rate = NULL
global_poverty_2$mfi_interest_rate = NULL
global_poverty_2$other_fsp_interest_rate = NULL
global_poverty_2$row_id = NULL

dim(global_poverty_2)
str(global_poverty_2)

num_na = c('education_level','share_hh_income_provided')
global_poverty_2 = global_poverty_2[complete.cases(global_poverty_2[,num_na]),]

#Create Dummies
dummies = dummyVars(poverty_probability ~ ., data = global_poverty_2)

#Partition the data
partition = createDataPartition(global_poverty_2[,'education_level'], times = 1, p = 0.8, list = F)
training = global_poverty_2[partition,] # create training feature sample
training_label = global_poverty_2[partition, 'poverty_probability'] # subset training labels
training = predict(dummies, newdata = training) #transform categorical variables to dummyVars
dim(training)

test = global_poverty_2[-partition,] #to create test sample features
test_label = global_poverty_2[-partition, 'poverty_probability'] #subset of test label
test = predict(dummies, newdata = test)
dim(test)
head(training)

#Scaling for all variables to be zero mean and unit variance
numcols = c('age', 'education_level', 'share_hh_income_provided', 'num_times_borrowed_last_year', 'borrowing_recency', 'num_shocks_last_year', 'avg_shock_strength_last_year', 'phone_technology', 'phone_ownership', 'num_formal_institutions_last_year', 'num_informal_institutions_last_year', 'num_financial_activities_last_year')
preProcValues = preProcess(training[,numcols], method = c('center','scale'))

training[,numcols] = predict(preProcValues, training[, numcols])
test[, numcols] = predict(preProcValues, test[, numcols])
head(training[, numcols])

#PCA - Principal Component Analysis
pca_pp = prcomp(training)

var_exp = pca_pp$sdev**2/sum(pca_pp$sdev**2)
var_exp
summary(var_exp)

#Plot variance from PCA
plot_scree = function(pca_mod){
  ## Plot as variance explained
  df = data.frame(x = 1:length(var_exp), y = var_exp)
  ggplot(df, aes(x,y)) + geom_line(size = 1, color = 'blue') +
    xlab('Component number') + ylab('Variance explained') +
    ggtitle('Scree plot of variance explained vs. \n Principal Component')
}

plot_scree(pca_pp)

#Compute first 10 PCA components
pca_pp_10 = prcomp(training, rank = 10)

var_exp_10 = pca_pp_10$sdev**2/sum(pca_pp_10$sdev**2)
Nrow = nrow(pca_pp_10$rotation)
Ncol = ncol(pca_pp_10$rotation)
scaled_pca_pp_10 = data.frame(matrix(rep(0, Nrow*Ncol), nrow = Nrow, ncol = Ncol))

#Scale in rotation
for (i in 1:Nrow) {
  scaled_pca_pp_10[i,]= pca_pp_10$rotation[i,]*var_exp_10[1:Ncol]
}

#Print top 10 PCA
dim(scaled_pca_pp_10)
pca_pp_10$rotation[1:10,1]

#Compute and Evaluate Linear Regression
training_10 = training %*% as.matrix(scaled_pca_pp_10)
dim(training_10)

## Construct a data frame with the transformed features and label
training_10 = data.frame(training_10)
training_10[,'poverty_probability'] = training_label

#Create Linear Regression model for PCA_10
Lin_Reg_10 = lm(formula = poverty_probability ~.,
                data = training_10)
Lin_Reg_10$coefficients

#Test PCA_10
test_10 = test %*% as.matrix(scaled_pca_pp_10)
test_10 = data.frame(test_10)
test_10[,'poverty_probability'] = test_label
dim(test_10)


#Compute first 68 PCA components
pca_pp_68 = prcomp(training, rank = 68)

var_exp_68 = pca_pp_68$sdev**2/sum(pca_pp_68$sdev**2)
Nrow = nrow(pca_pp_68$rotation)
Ncol = ncol(pca_pp_68$rotation)
scaled_pca_pp_68 = data.frame(matrix(rep(0, Nrow*Ncol), nrow = Nrow, ncol = Ncol))

#Scale in rotation
for (i in 1:Nrow) {
  scaled_pca_pp_68[i,]= pca_pp_68$rotation[i,]*var_exp_68[1:Ncol]
}

#Print top 68 PCA
dim(scaled_pca_pp_68)
pca_pp_68$rotation[1:68,1]

#Compute and Evaluate Linear Regression
training_68 = training %*% as.matrix(scaled_pca_pp_68)
dim(training_68)

## Construct a data frame with the transformed features and label
training_68 = data.frame(training_68)
training_68[,'poverty_probability'] = training_label

#Create Linear Regression model for PCA_68
Lin_Reg_68 = lm(formula = poverty_probability ~.,
                data = training_68)
Lin_Reg_68$coefficients
summary(Lin_Reg_68)

