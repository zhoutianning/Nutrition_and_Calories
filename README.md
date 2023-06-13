

# **Prediction Problem**

* Our **prediction problem** is predicting predicting the calories of recipes based on nutritions with Random Forest Regressor.

* Our **response variable** is the calories of a recipe. We chose it because we think there is a correlation between nutritions and calories. For instance, a recipe with more total fat is likely to have higher calories. Also if we can predict the calories of a recipe based on the nurtion, it would be helpful for people who want to lose weight.

* Moreover, we chose **R-squared** and **RMSE** as the metric to examine our model. These two metrics can effectively reflect how well our model fits. 
  * R-squared is the coefficient of determination, which showcases the proportion of variance in y that the linear model explains. 
  * On the other hand, RMSE measures how far predictions fall from true values using Euclidean distance. 
  * In contrast, MAE and MSE vary based on whether the values of response variable is scaled or not. In addition, MAE is not differentiable at 0. Hence, they could not be used as meaningful model performance indicator.

# Data Cleanning

* We extract and create columns for the nutritions , which we choose as the independent features from the nutrition column.

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

  
