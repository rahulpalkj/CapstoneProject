global_poverty_2 = read.csv(file= "C:/Users/Rahulkj/Documents/Datasets/train_values_labels.csv", header = T, stringsAsFactors = F)
dim(global_poverty_2)
global_poverty_2$bank_interest_rate = NULL
global_poverty_2$mm_interest_rate = NULL
global_poverty_2$mfi_interest_rate = NULL
global_poverty_2$other_fsp_interest_rate = NULL
global_poverty_2$row_id = NULL

#Removing rows
num_na = c('education_level','share_hh_income_provided')
global_poverty_2 = global_poverty_2[complete.cases(global_poverty_2[,num_na]),]
dim(global_poverty_2)

#Scaling
global_poverty_2$age <- scale(global_poverty_2$age) 
global_poverty_2$num_times_borrowed_last_year <- scale(global_poverty_2$num_times_borrowed_last_year)
global_poverty_2$num_informal_institutions_last_year <- scale(global_poverty_2$num_informal_institutions_last_year)
global_poverty_2$num_shocks_last_year <- scale(global_poverty_2$num_shocks_last_year)
global_poverty_2$num_formal_institutions_last_year <- scale(global_poverty_2$num_formal_institutions_last_year)

#Converting to Factor
as.factor(global_poverty_2$education_level)
as.factor(global_poverty_2$phone_technology)
as.factor(global_poverty_2$phone_ownership)

#Feature Engineering
religion_categories = c('N' = 'N_O_P', 'O' = 'N_O_P', 'P' = 'N_O_P', 'Q', 'X')

out = rep('i', length.out = nrow(global_poverty_2) )
i = 1
for(x in global_poverty_2[,'religion']){
  out[i] = religion_categories[[x]]
  i = i+1
}
global_poverty_2[,'religion'] = out
table(global_poverty_2[,'religion'])
#Partition
set.seed(1955)
partition = createDataPartition(global_poverty_2[,'poverty_probability'], times = 1, p = 0.75, list = F)
training = global_poverty_2[partition,]
dim(training)
test = global_poverty_2[-partition,]
dim(test)
summary(global_poverty_2)
str(global_poverty_2)

#linear regression
regressor = lm(formula = poverty_probability ~ country	+ is_urban	+ age	+ married	+ education_level	+ literacy	+ can_add	+ can_calc_percents	+  employed_last_year	+ employment_type_last_year	+ share_hh_income_provided	+ income_friends_family_last_year	+ income_government_last_year	+ income_private_sector_last_year	+ num_times_borrowed_last_year	+ formal_savings	+ has_investment	+  num_shocks_last_year	+ borrowed_for_emergency_last_year	+ phone_technology	+ can_text	+ can_use_internet	+ can_make_transaction	+ phone_ownership	+ reg_bank_acct	+ active_mm_user	+ num_formal_institutions_last_year	+ num_financial_activities_last_year,
               data = training)
summary(regressor)

#Regularization L2
dummies = dummyVars(poverty_probability ~., data = global_poverty_2)

training_dummy= predict(dummies, newdata = training)
head(training_dummy)
print(dim(training_dummy))

test_dummy= predict(dummies, newdata = test)

glmnet_l2 = glmnet(x= training_dummy, y = training[,'poverty_probability'],
                   nlambda = 20, alpha = 0, family = 'gaussian')
plot(glmnet_l2, xlab = 'Inverse of regularization')

cv_plot = cv.glmnet(x= training_dummy, y = training[,'poverty_probability'],
                    nlambda = 20, alpha = 0, family = 'gaussian')
plot(cv_plot)



#Metrics
print_metrics_glm = function(df,score,label = 'poverty_probability'){
  resids = df[,label]-score
  resids2 = resids**2
  N = length(score)
  SSR = sum(resids2)
  SST = sum(mean(df[,label]-df[,label])**2)
  r2 = as.character(round(1-SSR/SST,4))
  cat(paste('R^2 = ', r2,'\n'))
}
score = predict(glmnet_l2, newx = test_dummy)[,20]
print_metrics_glm(test, score, label = 'poverty_probability')