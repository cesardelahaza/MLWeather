# XGBoost

References on implementation: [here](https://kirenz.github.io/regression/docs/xgboost-regression.html).

XGBoost  (eXtreme Gradient Boosting) is a machine learning library which implements supervised machine learning models under the Gradient Boosting framework.

## Gradient Boosting

Reference: _Hands-on Machine Learning with Scikit-Learn, Keras & Tensorflow_, by Aurélien Géron.

### Boosting

Boosting (originally called hypothesis boosting) refers to any ensemble method that can combine several weak learners into a strong learner. The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor.

---

There are many boosting methods available, such as _AdaBoost_ and _gradient boosting_. We start with AdaBoost.

### AdaBoost

One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfit. This results in new predictors focusing more and more on the hard cases. This sequential learning technique has some similarities with gradient descent, except that instead of tweaking a single predictor's parameters to minimize a cost function, AdaBoost adds predictors to the ensemble, gradually making it better.

There is one important drawback to this sequential learning technique: training cannot be parallelized since each predictor can only be trained after the previous predictor has been trained and evaluated. As a result, it does not scale.

To make predictions, AdaBoost simply computes the predictions of all the predictors and weighs them using the predictor weights.

### Gradient Boosting

Gradient boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor.

## [XGBoost](https://github.com/dmlc/xgboost)

XGBoost builds on gradient boosting by adding several improvements that make it faster and more robust:

- **Regularization**: Unlike standard gradient boosting, XGBoost includes L1 and L2 regularization terms directly in its objective function, which helps prevent overfitting.
- **Efficient split finding**: Instead of evaluating every possible split, XGBoost uses a histogram-based approximation, which allows parallelization at the node level (not the tree level, since trees are still built sequentially).
- **Handling missing values**: XGBoost learns automatically which direction to send missing values at each split.
- **Tree pruning**: Trees are grown to a maximum depth and then pruned backwards, removing splits that don't improve the objective.

### XGBoost for regression

Although XGBoost is often associated with classification tasks, it supports regression natively. The objective function for regression is typically the mean squared error (MSE):

$$\mathcal{L} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \Omega(f)$$

where $\Omega(f)$ is the regularization term over the tree structure.

### Some configurable hyperparameters

| Hyperparameter | Description |
|---|---|
| `n_estimators` | Number of trees (boosting rounds) |
| `learning_rate` | Shrinks each tree's contribution; lower values need more trees |
| `max_depth` | Maximum depth of each tree; controls model complexity |
| `subsample` | Fraction of training samples used per tree; reduces overfitting |

## Appendix: Key Concepts

### Overfitting

A model overfits when it memorizes the training data instead of learning general patterns. An overfit model performs well on training data but poorly on new, unseen data.

### Regularization

Regularization is a technique to prevent overfitting. It works by adding a penalty to the model's objective function when its parameters become too large or complex, discouraging the model from fitting the training data too closely.

In XGBoost, the penalty applies to the tree structure: it penalizes having too many leaves and having leaf values that are too extreme.

### L1 and L2 Regularization

L1 and L2 are two ways of measuring the regularization penalty:

- **L1 (Lasso):** penalizes the sum of the absolute values of the parameters. This can shrink some parameters exactly to zero, effectively removing features from the model.
- **L2 (Ridge):** penalizes the sum of the squares of the parameters. This smooths the values without eliminating them entirely.

In XGBoost, L1 acts on the number and values of the leaves, and L2 smooths the leaf weights.

### Histogram-based Split Finding

To build each tree, XGBoost needs to find the best split point for each feature. Instead of evaluating every possible value (which is computationally expensive), it groups the data into bins and only evaluates cut points between bins. This makes the process faster and allows parallelization within each node.

### Missing Values

At each split, XGBoost automatically learns whether missing values should go to the left or right branch, choosing whichever minimizes the error. No imputation is needed before training.

### Tree Pruning

XGBoost grows each tree to `max_depth` and then prunes it backwards, removing splits whose net gain (after regularization) is negative. This is safer than stopping early, because a split that looks bad in isolation may precede a useful one.