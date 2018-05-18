## List of Contents  
1. Data Set  
1. Outlier Detection  
1. Feature Selection  
1. Feature Enhancements  
1. Scaling  
1. Algorithms and Tuning  
1. Validate and Evaluate  
1. Attachments  

### Data Set:
The dataset contains the following features:  
['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'bonus', 'exercised_stock_options', 
'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 
'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 
'long_term_incentive', 'from_poi_to_this_person']    

Not all features are required to identify the POI, the type for POI is Boolean. There are 145 data points, out of them 17 are POI's and 127 are non-POI's. The remainder of 1, belongs to the key 'TOTAL' which is an incorrect data point. With such few data points it’s difficult to train the models.  Also, there is quite some missing data. The data was split 90% for training and 10% for testing for PCA and SVC and 70% training and 30% testing for just individual models.  

### Outlier Detection:
The POI name 'TOTAL' and persons greater than 10000000$ of salary are considered as outliers
and they were removed from the training/testing dataset. The exorbitant salary was removed by using conditional filtering. The field TOTAL came across while analyzing the dataset and sorting it based on various amounts field. There were other amounts which were relatively higher, but I decided to keep those instances as some of the belonged to POI and might be a factor in deciding if the person is a POI.  

### Feature Selection:
Initially used PCA to select the features and many a time the number of components it selected between 6 -10. The outcome of GridSearch for best parameters for number of components is 10. For my final models I decided to leave out PCA for feature selection as PCA is usually used when the dimensionality of the dataset is really high. In this case the number of features is limited. The decision to remove some of the features was based on the data, if there were too many instances with zeros for amounts and if the probability of that person being classified as POI or non-POI is evenly distributed then such features were removed. This step was performed manually. For this project, 11 features were used to start with including two new features that were added. I executed the models with and without the additional features, the classifiers performed poorly with the features that I removed from my final dataset.  

I used GridSearch with SelectKBest for Decision Tree Classifier to get insight into the features and drop if required. Listed below are the importance of the features. The from_poi_ratio and the to_poi_ratio are the most influential.     
[('0.0', 'salary'), ('0.006979077228975689', 'total_payments'), ('0.14633397974066384', 'bonus'), ('0.0', 'restricted_stock'), ('0.0', 'total_stock_value'), ('0.09282238992573852', 'expenses'), ('0.0', 'other'), ('0.06203624203533821', 'long_term_incentive'), ('0.04665642369741078', 'shared_receipt_with_poi'), ('0.3332459668308555', 'from_poi_ratio'), ('0.3119259205410175', 'to_poi_ratio')]  

The following features were selected.  
['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'bonus', 'exercised_stock_options',   
'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 
'loan_advances', 'from_messages', 'other', 'director_fees', 'deferred_income', 'long_term_incentive']  

### Feature Enhancements:  
 For few records, from_messages and to_messages do not stack with the from_poi_to_this_person and from_this_person_to_poi. This was probably some data error and had to be fixed by making few assumptions. Not the ideal way, but with no. of data points available, dropping instances was not an option other than the outliers.    

Two new features were created, 'from_poi_ratio' and 'to_poi_ratio'. Since using just the number of messages
from poi and to poi does not give a good insight. My assumption was, if majority of the communication is done
with the poi's then the probability of the person being poi is high. Whenever there was lack of data 
for 'from_this_person_to_poi' and 'from_poi_to_this_person', and if the person was a POI, the POI ration
was set to 1.   

### Scaling  
Surprisingly, using minmax scaler did not improve the result by much.  

### Algorithms and Tuning: 
####PCA and SVC:
Initially used PCA to reduce the number of features and then ran SVC, could not get both precision and 
recall to above 0.5. I used grid search to find the right number of components. The end result wasn't
satisfactory. Listed below are the is the parameter grid which GridSearchCV uses to identify the best 
performing parameters for PCA for feature selection and SVC for classification.  
 
param_grid = dict(clf__kernel=['sigmoid', 'rbf'],  
                  clf__C=[0.001, 0.1, 1, 10, 100, 1000, 10000, 1e3, 5e3, 1e4, 5e4, 1e5],    
                  clf__gamma=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],  
                  reduce_dim__n_components=[1, 2, 4, 6, 8, 10, 12, 13])    

Listed below are the parameters that were used for fitting the model by GridSearchCV and to make a prediction.   

Best params: 'clf__gamma': 0.01, 'reduce_dim__n_components': 10, 'clf__shrinking': True, 'clf__C': 100000.0, 
  'clf__class_weight': 'balanced', 'clf__kernel': 'rbf'.

The rbf kernel does not give any true positive predictions. The result did not vary even when 
the "c" value was set really high. The higher the "c" value, the higher the variance. The CV grid best performing parameters are listed below and this was used to predict if a person was POI or not.  

SVC(C=100000.0, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf', max_iter=-1, probability=False, random_state=41, shrinking=True, tol=0.001, verbose=False)

In my case, the best performing parameters does not do a good job with prediction as it does not get a single true positive prediction.  I used sigmoid function instead of rbf for kernel.

#### PCA and Linear SVC:
Listed below is the model and the best parameters that were used to build. The best parameters were identified using GridCV.  

LinearSVC(C=1000, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=41, tol=0.0001, verbose=0)

Best params  
{'clf__dual': True, 'clf__loss': 'hinge', 'clf__C': 1000, 'clf__class_weight': None, 'reduce_dim__n_components': 12}

Decision Tree classifier
Listed below are the parameters that were used for tuning.  
param_grid_t = dict(clf__criterion=['gini', 'entropy'],
                    clf__min_samples_split=[2, 6, 10, 20],
                    clf__max_depth=[2, 6, 10, 20, None],
                    clf__max_features=['sqrt', None])  

The best parameters selected were 
{'clf__criterion': 'entropy', 'clf__max_depth': 10, 'clf__max_features': None, 'clf__min_samples_split': 2}  

The decision tree model was built with the above identified parameters. 

#### Random Forest 
Listed below are the parameters that were used for tuning.  
{' bootstrap': [True],
  'max_depth': [20, 40, 60, 80, 100],
    'max_features': [2, 4, 6, 8, 10, 11],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2, 4, 6, 8, 10, 12],
    'n_estimators': [10, 20, 40, 60, 80, 100, 200]}

The best parameters selected where.
{'bootstrap': True, 'min_samples_leaf': 2, 'n_estimators': 10, 'min_samples_split': 10, 'max_features': 6, 'max_depth': 20

### Validate and Evaluate
Initially, 70% of the data was held for training and 30% was done for testing. I suspected that there weren’t enough data points for a model to converge. I started using 95% for training and 5% for testing, this meant my test data was not a good indicator for the performance of the model. Either I implement the n-fold logic or take advantage of the tester.py script which uses stratified shuffled split with default folds set to 1000.  The tester.py uses the stratified shuffled split and the default folds are set to 1000.    

The classifier needs to have a high precision value due to the nature of the problem. The model should not be false identifying as a POI. The false positive should be as low as possible. Having a relatively higher 
false negative is still acceptable at the cost of not falsely accusing a person as POI. So, a lower recall 
is acceptable for this problem.  
   
#### Decision Tree Classifier
Listed below are the precision and recall values of the classifier.  
	Accuracy: 0.88333	Precision: 0.56600	Recall: 0.53600	F1: 0.55059	F2: 0.54174  
	Total predictions: 15000	True positives: 1072	False positives:  822	False negatives:  928	True negatives: 12178
	
Out of the total 15000 predictions, the true positives where 1072. the false positives are on the bit higher side. 56% of the times, the classifier detects if a person is POI. 

#### Random Forest 
Listed below are the precision and recall values. The recall value does not meet the project min required guidelines.  
	Accuracy: 0.87907	Precision: 0.60265	Recall: 0.27300	F1: 0.37577	F2: 0.30653  
	Total predictions: 15000	True positives:  546	False positives:  360	False negatives: 1454	True negatives: 12640  

Out of total 150000 predictions, 360 persons were identified as POI but in reality, they weren’t, and 1454 were identified as non-POI but they were. Out of the total positives identified, 60% of the predictions were correct.
Models for this type of problem, should reduce the no. of false positives to as low as possible.
The model also did a fairly good job in identify non-POI’s correctly.   

#### PCA with SVC 
Listed below are the precision and recall values for Linear SVC with PCA.  
Accuracy: 0.72340	Precision: 0.18755	Recall: 0.32250	F1: 0.23718	F2: 0.28193  
	Total predictions: 15000	True positives:  645	False positives: 2794	False negatives: 1355	True negatives: 10206  

The true positives are comparable to Random forest, but this model, also classifies a fairly large number of persons as POI’s incorrectly. A coin toss fares lot better than this model. This model also classifies 1355 persons as non-POI’s incorrectly. 

### Attachments
poi_id.py - Contains Random Forest and Decision Tree Classifier.  
poi_id_1.py - Contains PCA with SVC   .
my*.pkl – Classifier, dataset and feature list.  
Results.xlsx – Few of the results that were captured and the dataset.  

Note: In order to run GridSearchCV in poid_id.py, the import line for GridSearchCV needs to be uncommented.
