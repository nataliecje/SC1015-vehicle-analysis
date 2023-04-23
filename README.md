# SC1015 Mini Project

**Goal:** To help vehicle insurance companies issue different insurance premium according to the different types of driver

**Problem:**
1) To predict whether a driver will make an insurance claim
2) To predict the claim cost for drivers who makes claim(s)

Our dataset was taken from Kaggle:
https://www.kaggle.com/datasets/lakshmanraj/vehicle-insurance-policy

Original dataset    : Vehicle_policies_2020.csv                                       
Cleaned dataset     : Prepped.csv

Please view the notebooks in the following order:                                    
1) Data Preparation                                                                  
2) Exploratory Data Analysis                                                             
3) Machine Learning                                                                                  

## Data Preparation

## Table of Contents
>1. Dropping of Unused Variables
>2. Derivation of New Variables
>2. Dealing with Empty Cells
>3. Removal of Outliers
>4. Re-scaling of Variables

**1. Dropping of Unused Variables**

Variables dropped: pol_number, date_of_birth, claim_office, annual_premium

These variables are not found useful in helping us to reach a solution for our problem statements. 


**2. Derivation of New Variable**

Variables derived: month, year, monthDiff, claims

month & year: derived from pol_eff_date; used for calculation of monthDiff
monthDiff: derived from month & year; represents the difference in months from pol_eff_date to current (1 April 2023) 
claims: derived from numclaims; indicated by “Yes” or “No”, to specify if a customer has made claims or not 


**3. Removal of Outliers**

Outliers can have a big impact on your statistical analyses and skew the results of any hypothesis test if they are inaccurate. These extreme values can impact your statistical power as well, making it hard to detect a true effect if there is one.


**4. Re-scaling of Variables**

Variables re-scaled: credit_score, traffic_index, veh_age

Data scaling is the process of transforming the values of the features of a dataset till they are within a specific range.

##### credit_score ranges from 301 to 850;
##### traffic_index ranges from 0 to 207;
##### veh_age ranges from 1 to 4;

As we have variables with data points far from each other, scaling is a technique to make them closer to each other or in simpler words, we can say that the scaling is used for making data points generalized so that the distance between them will be lower. Therefore, we have made a linear transformation of the data of some variables, namely credit_socre, traffic index, and veh_age, to map the minimum and maximum value to 1 and 100 respectively. This adjusts the numbers to make it easy to compare the values that are out of each other's scope, ultimately increasing the accuracy of the models.


## Exploratory Data Analysis

In this section, we will be exploring 3 predictor variables `gender`, `age` and `traffic` that are related to the 2 response variables `claims` and `cost` to find patterns or underlying relationships between the predictor and response variables. This is going to help us answer the initial question we posed in the subsequent sections. 
<br/> 
<br/>
Lastly, we will also be taking a look at the relationship between `numclaims` and `cost`.

<br/>

## Table of Contents 
>1. Classifying dataset into numerical and categorical <br>
>2. Exploring Predictor Variables <br>
>>2.1 `gender` (gender of individual) <br>
>>2.2 `age` (age category of individual) <br>
>>2.3 `traffic` (traffic index) <br>
>3. Exploring the Response Variables
>>3.1 `claims` (boolean variable indicating presence of an insurance claim) <br>
>>3.2 `cost` (cost of insurance claim when `claims` is present) <br>
>4. Exploring Relationship between predictor variables and `claims`
>>4.1 `gender` against `claims` <br>
>>4.2 `age` against `claims` <br>
>>4.3 `traffic` against `claims` <br>
>5. Exploring Relationship between predictor variables and `cost`
>>5.1 `gender` against `cost` <br>
>>5.2 `age` against `cost` <br>
>>5.3 `traffic` against `cost` <br>
>6. Exploring Relationship between `numclaims` and `cost`

<br/>


## Machine Learning

For all machine learning model train to test ratio will be 70 : 30.

Metrics used for classifications:                                     
- Classification accuracy = (TP + TN) / no. of samples                  
- True positive rate = true positives / all positives                     
- True negative rate = true negative / all negatives                     
- False positive rate = false positive / all negatives                   
- False negative rate = false negative / all positive                     

Metrics used for regression:                                     
- R^2 = 1 - (MSE / variance)
- Mesn Squared Error = RSS / no. of samples

** the results shown for the model are all used with the hyperparameters that performs the best

### Predicting whether a driver will make an insurance claim

Response variable: `claims` (Yes / No)                                      
Predictor variables: `gender`, `agecat,` `area`, `veh_body`, `credit_score`, `traffic_index`, `veh_age`, `veh_value`, `monthDiff`

**One Hot Encoding**

`OneHotEncoding` was first performed to convert categorical variabels to numerical.

It creates a binary representation of the categories and assigns "1" to the category, and "0" for others.

**Resampling**

There is a huge data imbalance of the response variable. To prevent the results from being skewed, we will resample the data

**Logistic Regression**

`LogisticRegression` is suitable for predicting binary categorical variables. It uses a logistic function (or sigmoid function), which maps any input value to a probability between 0 and 1. The input values are combined linearly with weights, and then the logistic function is applied to the result. During training, the logistic regression model adjusts its weights to maximize the likelihood of the correct class label given the input data. This is done using a cost function, such as the cross-entropy loss function, and an optimization algorithm, such as gradient descent.

Both train and test set yields a similar result that is fairly good.

Hyperparameters:                                                       
- max_iter = 400                                                         

> *Train set*                                                            
>> Classification accuracy           : 0.735                                          
>> True Positive Rate                : 0.736                                                     
>> True Negative Rate                : 0.734
>> False Positive Rate               : 0.266                                                         
>> False Negative Rate               : 0.264

> *Test set*                                                                  
>> Classification accuracy           : 0.737                                       
>> True Positive Rate                : 0.742                                      
>> True Negative Rate                : 0.733                                      
>> False Positive Rate               : 0.267                                      
>> False Negative Rate               : 0.258                                      

The almost identical results of train and test suggests the fitting of the model is good.

`GridSearchCV` was performed to determine the best solver. Solver is the algorith used by the model for optimisation. However, all solver yields the same score, as shown below:

> liblinear       : 0.736                                                                             
> newton_cg       : 0.736                                                                     
> lbfgs           : 0.736                                                                       
> sag             : 0.736                                                                       
> saga            : 0.736                                                                      

**Random Forest Classifier**

`RandomForest` generates multiple decision tree to make predictions about a target variable, and is suitable for complex datasets. Each tree is exposed to a different number of features and a different sample of the original dataset, and as such, every tree can be different. Each tree then makes a prediction, and finally taking the most popular result.

Hyperparameters:                                                                 
- n_estimators: 100                                                                
- max_depth: 10                                                                   

> *Train set*                                                                          
>> Classification accuracy           : 0.773                                      
>> True Positive Rate                : 0.784                                      
>> True Negative Rate                : 0.761                                      
>> False Positive Rate               : 0.239                                      
>> False Negative Rate               : 0.215                                      

> *Test set*                                                                                
>> Classification accuracy           : 0.763                                      
>> True Positive Rate                : 0.775                                      
>> True Negative Rate                : 0.752                                      
>> False Positive Rate               : 0.248                                      
>> False Negative Rate               : 0.225                                      

This model performs slightly better than the previous.

`GridSearchCV` was also used to determine the best parameters.

Max depth, which indicated the maximum depth of the tree, was determined to be best at 10.
N estimators, the number of trees in the forest, was determined to be 600.

However, after tuning the parameters, the results are almost the same.


### Predicting the claim costs of drivers who made claims

Response variable: `claimcst0` (numeric)

**Extract data**

We first filter out the data that have more than or equals to 1 claim.

**Dummy Encoding**

Similarly, we performed `DummyEncoding` to convert categorical variabels to numerical.

**Linear Regression**

`LinearRegression` shows the linear relationship between the predictor and response variable. It uses a straight line y = b0 + b1 * x. The goal of linear regression is to get the best values for b0 and b1 to find the best fit line.

Predictor variables: `agecat`, `area`, `veh_body`, `credit_score`, `traffic_index`, `monthDiff`, `numclaims`

> *Train set*                                                                              
>> Explained variance    : 0.970                                      
>> Mean Squared Error    : 934884.176                                      

> *Test set*                                                                                 
>> Explained variance    : -0.522                                      
>> Mean Squared Error    : 49792553.483                                      

Despite performing multiple times by varying the predictor variables, the R^2 and mean squared error are all not ideal. We have constant underfitting issues, which may suggest that the predictor and response variable relation are not linear.

**Ridge Regression**

Ridge Regression is suitable for variables with multicollinearity. It is a type of linear regression and that ads a penalty term to the cost function that is minimised during training. The penalty term is based on the sum of the squared values of the regression coefficients, and it is multiplied by a hyperparameter called lamda. This hyperparameter controls the strength of the penalty and helps to balance the tradeoff between model complexity and model fit. In other words, ridge regression adds a little bit of bias to the model to reduce the variance that comes from having too many predictors. This can result in better performance on new, unseen data.

Predictor variables: "agecat", "area", "veh_body", "credit_score", "traffic_index", "veh_age", "monthDiff", "numclaims"

> *Train set*                                                                  
>> Explained variance    : 0.822                                      
>> Mean Squared Error    : 5753073.318                                      

> *Test set*                                                              
>> Explained variance    : 0.302                                      
>> Mean Squared Error    : 21572444.483                                      

There is an overfitting issue here despite multiple tries of reducing and varying the predictor variables.

## Conclusion

Problem 1: A fairly good model was built using classification to predict if a driver will make claims or not

Problem 2: A model was built to predict the claim cost, but the model is not accurate.

In real life, an accident involves many factors, such as speed, and traffic density. In addition, claim cost can also vary from mechanic to mechanic. Furthermore, claim cost is also not proportional to the number of claims. All of these might be reasons why the regression was not well performed.

Limitations:                                                         
- Inaccurate prediction of claim cost                                         

Possible improvements:                                                       
- Due to the extremely large dataset, it might me better to split the dataset into multiple ones, perform the same machine learning, and take the average.
- Before predicting the claim cost, predict the number of claims that will be made first


## Work distribution

- Data Preparation            : Jie En                                                          
- EDA                         : Benjamin                                                                
- Machine Learning            : Jun Yu                                                                  
- Video                       : Everyone                                                                         
- README                      : Everyone                                                                  

