# SC1015 Mini Project

**Goal:** To help vehicle insurance companies issue different insurance premium according to the different types of driver

**Problem:**
1) To predict whether a driver will make an insurance claim
2) To predict the claim cost for drivers who makes claim(s)

Our dataset was taken from Kaggle:
https://www.kaggle.com/datasets/lakshmanraj/vehicle-insurance-policy

## Data Preparation


## Exploratory Data Analysis



## Machine Learning

For all machine learning model train to test ratio will be 70 : 30.

### Predicting whether a driver will make an insurance claim

Response variable: Claims (Yes / No)

**One Hot Encoding**

One Hot Encoding was first performed to convert categorical variabels to numerical.

It creates a binary representation of the categories and assigns "1" to the category, and "0" for others.

**Resampling**

There is a huge data imbalance of the response variable. To prevent the results from being skewed, we will resample the data

**Logistic Regression**

Logistic Regression is suitable for predicting binary categorical variables. It uses a logistic function to model the probability of the response variable as a function of the predictor variables.

Both train and test set yields a similar result that is fairly good.

*Train set*
Classification accuracy           : 0.735

True Positive Rate                : 0.736
True Negative Rate                : 0.734

False Positive Rate               : 0.266
False Negative Rate               : 0.264

*Test set*
Classification accuracy           : 0.737

True Positive Rate                : 0.742
True Negative Rate                : 0.733

False Positive Rate               : 0.267
False Negative Rate               : 0.258

The almost identical results of train and test suggests the fitting of the model is good.

Grid Search CV was performed to determine the best solver. Solver is the algorith used by the model for optimisation. However, all solver yields the same score, as shown below:

liblinear       : 0.736
newton_cg       : 0.736
lbfgs           : 0.736
sag             : 0.736
saga            : 0.736

**Random Forest Classifier**

Random Forest generates multiple decision tree to make predictions about a target variable, and is suitable for complex datasets.

Train set
Classification accuracy           : 0.773

True Positive Rate                : 0.784
True Negative Rate                : 0.761

False Positive Rate               : 0.239
False Negative Rate               : 0.215

Test set
Classification accuracy           : 0.763

True Positive Rate                : 0.775
True Negative Rate                : 0.752

False Positive Rate               : 0.248
False Negative Rate               : 0.225

This model performs slightly better than the previous.

Grid Search CV was also used to determine the best parameters.

Max depth, which indicated the maximum depth of the tree, was determined to be best at 10.
N estimators, the number of trees in the forest, was determined to be 600.

However, after tuning the parameters, the results are almost the same.


### Predicting the claim costs of drivers who made claims

Response variable: Claim cost (numeric)

**Extract data**

We first filter out the data that have more than or equals to 1 claim.

**Dummy Encoding**

Similarly, we performed Dummy Encoding to convert categorical variabels to numerical.

**Linear Regression**

Train set
Explained variance    : 0.970
Mean Squared Error    : 934884.176

Test set
Explained variance    : -0.522
Mean Squared Error    : 49792553.483

Despite performing multiple times by varying the predictor variables, the R^2 and mean squared error are all not ideal. We have constant underfitting issues, which may suggest that the predictor and response variable relation are not linear.

**Ridge Regression**

Ridge Regression is suitable for variables with multicollinearity.

Train set
Explained variance    : 0.822
Mean Squared Error    : 5753073.318

Test set
Explained variance    : 0.302
Mean Squared Error    : 21572444.483

There is an overfitting issue here despite multiple tries of reducing and varying the predictor variables.

## Conclusion

Problem 1: A fairly good model was built using classification to predict if a driver will make claims or not

Problem 2: A model was built to predict the claim cost, but the model is not accurate.

In real life, an accident involves many factors, such as speed, and traffic density. In addition, claim cost can also vary from mechanic to mechanic. Furthermore, claim cost is also not proportional to the number of claims. All of these might be reasons why the regression was not well performed.


