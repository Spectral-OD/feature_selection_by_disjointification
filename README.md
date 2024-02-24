# Feature Selection By Disjointification
 
## M.Sc MLDS Project
Code for performing feature selection by disjointification, as well as benchmarking/testing the resulting selection.
Data loaded for the experiment is from gean expression data, with loader function implemented and demonstrated.
Follow the notenooks to track project progress and stages

## Known Issues
- logistic regression cannot work at present without setting a max_iter parameter. This is temporarily addressed by a kwarg with default value 10000.

## Backlog
- Rename class Disjointification to e.g. DisjointificationModel and spinoff to separate file DisjointificationModel.py
- Major refactorization of code to eliminate distinction of regression from classification
  - replace attributes regression_correlation_method, classification correlation method to a list of functions and/or strings
  - replace number_of_features_tested_log, number_of_features_tested_log with a list of integers
  - replace correlation_ranking_log, correlation_ranking_lin with a list of floats
  - y_test_log, ...train_log, _pred_log ... _pred_lin - replace with lists
  - logistic_regressor, linear_regressor, log_score, lin_score - lists
  - smartly treat test_size, correlation_thresholds, min_num_features as either a single numeric or an itterable of numerics

- implement runtime duration calculation for Disjointification class
  - add 'fmt' attribute for datettime format storage
  - add 'start_time' attribute set during .run()
  - add 'get_duration' by calculating the time delta between start and end times