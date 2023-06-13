

# **Prediction Problem**

**Our exploratory data analysis on this dataset can be found**: [here](https://zhoutianning.github.io/Protein_and_Rating/)

* Our **prediction problem** is predicting predicting the calories of recipes based on nutritions with Random Forest Regressor.

* Our **response variable** is the calories of a recipe. We chose it because we think there is a correlation between nutritions and calories. For instance, a recipe with more total fat is likely to have higher calories. Also if we can predict the calories of a recipe based on the nurtion, it would be helpful for people who want to lose weight.

* Moreover, we chose **R-squared** and **RMSE** as the metric to examine our model. These two metrics can effectively reflect how well our model fits. 
  * R-squared is the coefficient of determination, which showcases the proportion of variance in y that the linear model explains. 
  * On the other hand, RMSE measures how far predictions fall from true values using Euclidean distance. 
  * In contrast, MAE and MSE vary based on whether the values of response variable is scaled or not. In addition, MAE is not differentiable at 0. Hence, they could not be used as meaningful model performance indicator.

# Data Cleanning

We extract and create columns for the nutritions , which we choose as the independent features from the nutrition column.

```python
# Read in the raw dataset
recipe = pd.read_csv('RAW_recipes.csv')

# Extract and create columns for the nutritions we want to investigate from the nutrition column
recipe['calories'] = recipe['nutrition'].apply(lambda x: x[1:-1]).str.split(',').apply(lambda x: x[0]).astype(float)
recipe['total_fat'] = recipe['nutrition'].apply(lambda x: x[1:-1]).str.split(',').apply(lambda x: x[1]).astype(float)
recipe['sugar'] = recipe['nutrition'].apply(lambda x: x[1:-1]).str.split(',').apply(lambda x: x[2]).astype(float)
recipe['protein'] = recipe['nutrition'].apply(lambda x: x[1:-1]).str.split(',').apply(lambda x: x[4]).astype(float)
recipe['carbohydrates'] = recipe['nutrition'].apply(lambda x: x[1:-1]).str.split(',').apply(lambda x: x[6]).astype(float)
recipe.head()
```

|      |                               name |     id | minutes | contributor_id |  submitted |                                              tags |                                     nutrition | n_steps |                                             steps |                                       description |                                       ingredients | n_ingredients | calories | total_fat | sugar | protein | carbohydrates |
| ---: | ---------------------------------: | -----: | ------: | -------------: | ---------: | ------------------------------------------------: | --------------------------------------------: | ------: | ------------------------------------------------: | ------------------------------------------------: | ------------------------------------------------: | ------------: | -------: | --------: | ----: | ------: | ------------: |
|    0 |  1 brownies in the world best ever | 333281 |      40 |         985201 | 2008/10/27 | ['60-minutes-or-less', 'time-to-make', 'course... |      [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0] |      10 | ['heat the oven to 350f and arrange the rack i... | these are the most; chocolatey, moist, rich, d... | ['bittersweet chocolate', 'unsalted butter', '... |             9 |    138.4 |      10.0 |  50.0 |     3.0 |           6.0 |
|    1 | 1 in canada chocolate chip cookies | 453467 |      45 |        1848091 |  2011/4/11 | ['60-minutes-or-less', 'time-to-make', 'cuisin... |  [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] |      12 | ['pre-heat oven the 350 degrees f', 'in a mixi... | this is the recipe that we use at my school ca... | ['white sugar', 'brown sugar', 'salt', 'margar... |            11 |    595.1 |      46.0 | 211.0 |    13.0 |          26.0 |
|    2 |             412 broccoli casserole | 306168 |      40 |          50969 |  2008/5/30 | ['60-minutes-or-less', 'time-to-make', 'course... |     [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0] |       6 | ['preheat oven to 350 degrees', 'spray a 2 qua... | since there are already 411 recipes for brocco... | ['frozen broccoli cuts', 'cream of chicken sou... |             9 |    194.8 |      20.0 |   6.0 |    22.0 |           3.0 |
|    3 |             millionaire pound cake | 286009 |     120 |         461724 |  2008/2/12 | ['time-to-make', 'course', 'cuisine', 'prepara... | [878.3, 63.0, 326.0, 13.0, 20.0, 123.0, 39.0] |       7 | ['freheat the oven to 300 degrees', 'grease a ... |  why a millionaire pound cake? because it's su... | ['butter', 'sugar', 'eggs', 'all-purpose flour... |             7 |    878.3 |      63.0 | 326.0 |    20.0 |          39.0 |
|    4 |                      2000 meatloaf | 475785 |      90 |        2202916 |   2012/3/6 | ['time-to-make', 'course', 'main-ingredient', ... |    [267.0, 30.0, 12.0, 12.0, 29.0, 48.0, 2.0] |      17 | ['pan fry bacon , and set aside on a paper tow... | ready, set, cook! special edition contest entr... | ['meatloaf mixture', 'unsmoked bacon', 'goat c... |            13 |    267.0 |      30.0 |  12.0 |    29.0 |           2.0 |

# Baseline Model

### Our Independent Features and Model

* We choose ['total_fat', 'sugar', 'protein', 'carbohydrates'] as our nutrition features and ['calories'] as our dependent variable.

  ```python
  # Get the features and response variables from the dataframe
  X = recipe[['total_fat', 'sugar', 'protein', 'carbohydrates']]
  y = recipe['calories']
  ```

* Since all features we choose are quantitative, we directly apply a random forest regressor as our baseline model.

  A Random Forest Regressor is a supervised machine learning algorithm that is used for regression tasks. It is an ensemble learning method that combines multiple decision trees to make predictions. Random forests are called "random" because they introduce randomness into the tree-building process. 

  ```python
  # Since all features are quantitative, we directly apply a random forest regressor as our baseline model
  baseline_model = Pipeline([('rfr', RandomForestRegressor(n_estimators=50, max_depth=3))])
  ```

### Performance of our model

* In order to evaluate whether our model would perform well on unseen data, we split the data into training and testing sets, and we fit the model with trainning data.

  ```python
  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  
  # Fit the model with the training data and examine the performance of the model on the training set with R-squared
  baseline_model.fit(X_train, y_train)
  baseline_model.score(X_train, y_train)
  ```

  ```python
  0.8778365991185519
  ```

  The $R^2$ for our trainning data set is 0.878, which means around 87.8% of the trainning data can be explained by our regression model.

* Then we evaluate how the model can perform on our testing unseen data.

  ```python
  # Examine the performance of the model on the testing set with R-squared
  baseline_model.score(X_test, y_test)
  ```

  ```python
  0.8751488687009604
  ```

  The $R^2$ for our testing data set is 0.875, which means around 87.5% of the testing data can be explained by our regression model. The  $R^2$  is only a little smaller, which means our model can perform well on unseen data.

* We create a funtion to calculate RMSE for predicted and actual calories.

  ```python
  def rmse(actual, pred):
      return np.sqrt(np.mean((actual - pred) ** 2))
  ```

  ```python
  # RMSE of the model on the training set
  rmse(y_train, baseline_model.predict(X_train))
  ```

  ```python
  219.59147708023818
  ```

  ```python
  # RMSE of the model on the testing set
  rmse(y_test, baseline_model.predict(X_test))
  ```

  ```python
  236.38588008151456
  ```

  We can see the RMSE for testing data is higher than the trainning data but the difference is not large, we is accepatable. As a result, according to the overall performace of our model reflected by $R^2$ and RMSE, we believe our current model is good, but still have room for improvement.

# **Final Model**

## Feature engineering

We add quantiled protein, quantitled sugar, squared carbohydrates and squared total_fat as our new features.

Here are the ditribution of value of protein and sugar.

<iframe src="protein_dist.html" width=500 height=400 frameBorder=0></iframe>

<iframe src="sugar_dist.html" width=500 height=400 frameBorder=0></iframe>

As we can see, The histogram of protein and surgar values are both extremely left skewed because most of the data are in the same range excepet some outliers, and in order to capture the outliers we choose to quantile the values for protein and sugar.

We also choose to square carbohydrates and and total fat because we believe these two elements has big correlation with calories. So we use square function to enlarge the influence of carbohydrates and total fat.

```python
# Pipline for feature engineering and training the model.
# Build the final model with feature engineering
# Transform protein and sugar into quantiles
# Use square function to enlarge the influence of carbohydrates and total fat
square_tran = FunctionTransformer(np.square)
ct = ColumnTransformer([
                    ('quantile', QuantileTransformer(n_quantiles=100), ['protein', 'sugar']),
                    ('square', square_tran, ['carbohydrates', 'total_fat']),
                    ], remainder='passthrough')
final_model = Pipeline([('preprocess', ct), ('rfr', RandomForestRegressor(n_estimators=50, max_depth=3))])
```

```python
# R-squared of final model on training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
final_model.fit(X_train, y_train)
final_model.score(X_train, y_train)
```

```python
0.8840652941487283
```

```python
final_model.score(X_test, y_test)
```

```python
0.9003295819175974
```

In the final model, the $R^2$ of trainning data is 0.884 $R^2$ of testing data is 0.900, they are both higher than the corresponding values in the baseline model, which indicates more data can be explained after the feature enginerring.

```python
rmse(y_train, final_model.predict(X_train))
```

```python
219.04738286929597
```

```python
rmse(y_test, final_model.predict(X_test))
```

```python
195.61502529868255
```

In the final model, the RMSE of trainning data and RMSE of testing data are both lower than the corresponding values in the baseline model, which indicate less predicting error. In conclusion, our feature engineering help us improve the model's performance.

## **Tuning** of Random Forest Regressor

Though the final model has better performance on both training and testing set, we want to search for the best hyperparameters to further improve our **Random Forest Regressor model**.  It is an ensemble learning method that combines multiple decision trees to make predictions. Two hyperparameters that can impact the performance of our model are **n_estimators**, which is the number of trees in the forest and **max_depth**, which is the maximum depth of the tree. Hence, we apply **Grid Search CV** to find the best value of the two hyperparameters. It is a method for systematically searching for the optimal combination of hyperparameters to improve the model's performance.

```python
best_model = GridSearchCV(final_model, param_grid={'rfr__max_depth': [3, 7, 11, 15], 'rfr__n_estimators': [50, 100, 150, 200]}, cv=5)
best_model.fit(X_train, y_train)
best_model.best_score_
```

```python
0.9398996108533421
```

```python
best_model.best_params_
```

```python
{'rfr__max_depth': 11, 'rfr__n_estimators': 100}
```

By performing Grid Search CV, we found the best hyperparameters for **max_depth: 11 and n_estimators: 100**. 

## Tunned final model

```python
tunned_final_model = Pipeline([('preprocess', ct), ('rfr', RandomForestRegressor(max_depth=11, n_estimators=200))])
tunned_final_model.fit(X_train, y_train)
tunned_final_model.score(X_train, y_train)
```

```python
0.9891193634856745
```

```python
tunned_final_model.score(X_test, y_test)
```

```python
0.9880027215942357
```

```python
rmse(y_train, tunned_final_model.predict(X_train))
```

```python
67.1302974070314
```

```python
rmse(y_test, tunned_final_model.predict(X_test))
```

```python
66.6032341975582
```

The $R^2$ for the tunned final model for unseen data is 0.988 which is pretty high and 99.0% of the unseen data can be explained by the model.

# Fairness Analysis

* We want to investigate whether our tunned final model performs the same for recipes with high protein and recipes with low protein. The evaluation metrics we chose is RMSE.

* **Null hypothesis**: The tunned final model performs the same (i.e. has same RMSE) for recipes with high protein and recipes with low protein.

* **Alternative hypothesis**: The tunned final model performs differently (i.e. has different RMSE) for recipes with high protein and recipes with low protein.

* Our test statistic is the **absolute difference between the RMSE**of high protein recipes and low protein recipes. We use **0.05** as our significance level.

  

We first train the tunned_final_model on the trainning data we splited by cross validation and generate all the information needed for permutation test in one dataframe.

```python
# train the tunned_final_model on the trainning data we splited by cross validation
perm_X = recipe[['total_fat', 'sugar', 'protein', 'carbohydrates']]
perm_y = recipe['calories']
perm_X_train, perm_X_test, perm_y_train, perm_y_test = train_test_split(perm_X, perm_y, test_size=0.2)
tunned_final_model.fit(perm_X_train, perm_y_train)
y_pred = final_model.predict(perm_X_test)
```

```python
# Gather all the information needed for permutation test in one dataframe
results = perm_X_test.copy()
results['prediction'] = y_pred
results['calories'] = perm_y_test
```

```python
# Find the best value to split recipes as high protein and low protein
results['protein'].describe()
```

```pytho
count    16757.000000
mean        33.628872
std         61.062931
min          0.000000
25%          6.000000
50%         18.000000
75%         49.000000
max       3605.000000
Name: protein, dtype: float64
```

We add a categorical column by classifying recipes with fat greater than or equal to the median (18) as high protein, and others as low protein.

```python
# Classify recipes with fat greater than or equal to the median (18) as high protein, and others as low protein
results['high_protein'] = results['protein'] >= 18
results.head()
```

|       | total_fat | sugar | protein | carbohydrates | prediction | calories | high_protein |
| ----: | --------: | ----: | ------: | ------------: | ---------: | -------: | -----------: |
| 37284 |      24.0 |  10.0 |     0.0 |           1.0 |    154.417 |    153.2 |        False |
| 82349 |      25.0 | 156.0 |     2.0 |          14.0 |    324.683 |    322.4 |        False |
| 34107 |      53.0 |  15.0 |    40.0 |           9.0 |    520.924 |    534.1 |         True |
| 74226 |      32.0 |  99.0 |     0.0 |           8.0 |    306.046 |    286.3 |        False |
| 45062 |       7.0 |  11.0 |    40.0 |           5.0 |    185.123 |    182.1 |         True |

The get the observed absolute difference in RMSE between high protein and low protein recipes by grouping by the data set into two groups and calculated RMSE seprately and get the absolute difference.

```python
# Observed difference in RMSE between high protein and low protein recipes
obs_rmse = abs(results.groupby('high_protein').apply(lambda x: rmse(x['calories'], x['prediction'])).diff().iloc[-1])
obs_rmse
```

```python
40.54544925214688
```

Our observed difference in RMSE is 40.54544925214688.

## Permutation test

```python
# Permutation test
diff_rmse_ls = []
shuffled = results.copy()
for _ in range(100):
    shuffled['high_protein'] = np.random.permutation(shuffled['high_protein'])
    shuffled['prediction'] = final_model.predict(shuffled[['total_fat', 'sugar', 'protein', 'carbohydrates']])
    diff_rmse = abs(shuffled.groupby('high_protein').apply(lambda x: rmse(x['calories'], x['prediction'])).diff().iloc[-1])
    diff_rmse_ls.append(diff_rmse)
```

```python
# P-value of RMSE
p_val_rmse = (np.array(diff_rmse_ls) >= obs_rmse).mean()
p_val_rmse
```

```python
0.62
```

Our p value is 0.62.

```python
# visualization of the permutation test 
fig_rmse = pd.Series(diff_rmse_ls).plot(kind='hist', histnorm='probability', nbins=20,
                            title='Difference in RMSE (High fat - Low fat)')
fig_rmse.add_vline(x=obs_rmse, line_color='red')
fig_rmse.update_layout(xaxis_range=[-5, 100])
fig_rmse.add_annotation(text='<span style="color:red">Observed Difference in RMSE</span>', \
    x=25,showarrow=False, y=0.17)
```

<iframe src="perm_rmse.html" width=500 height=400 frameBorder=0></iframe>

## Conclusion

In our shuffled case, our p value 0.62 is greater than our significance level, so we fail to reject the null and the tunned final model performs the same for recipes with high protein and recipes with low protein. This indicates that our model performs fairly on both recipes with high protein and low protein. However, in other shuffled cases, we may reach other conclusions.
