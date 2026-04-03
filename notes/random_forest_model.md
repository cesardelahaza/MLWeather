# Random Forest

## Decision Trees

A decision tree is a supervised learning model that splits the data recursively based on feature values, creating a tree structure where each internal node represents a decision rule and each leaf represents a predicted value (in regression, the mean of the samples that reach that leaf).

Decision trees are simple and interpretable, but they tend to overfit: a deep tree can memorize the training data instead of learning general patterns.

## Ensemble Methods

Ensemble methods combine multiple models to produce a better prediction than any single model alone. The two main strategies are:

- **Bagging:** trains multiple models independently on different random subsets of the data, then averages their predictions. Reduces variance.
- **Boosting:** trains models sequentially, each correcting the errors of the previous one. Reduces bias.

Random Forest uses bagging.

## Random Forest

Random Forest is an ensemble of decision trees trained using a technique called **bagging** (Bootstrap Aggregating). Each tree is trained on a random bootstrap sample of the training data (sampled with replacement). Additionally, at each split, only a random subset of features is considered. This decorrelates the trees, making the ensemble more robust than bagging alone.

For regression, the final prediction is the average of all trees' predictions.

### Random Forest for Regression

Random Forest naturally supports regression by averaging the outputs of all trees. It is non-parametric, meaning it makes no assumptions about the distribution of the data, and it handles non-linear relationships well.

### Key Hyperparameters

| Hyperparameter | Description |
|---|---|
| `n_estimators` | Number of trees in the forest; more trees reduce variance but increase training time |
| `max_depth` | Maximum depth of each tree; `None` grows trees until leaves are pure |
| `max_features` | Number of features considered at each split; typically `sqrt` for classification, `1/3` for regression |
| `min_samples_leaf` | Minimum number of samples required at a leaf; higher values smooth the predictions |
| `bootstrap` | Whether to use bootstrap samples; `True` by default |