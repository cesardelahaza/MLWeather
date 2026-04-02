# Linear Model

Reference: _The Elements of Statistical Learning. Data Mining, Inference, and Prediction_.

----------

Given a vector of inputs $X^t = (X_1,X_2,...,X_p)$, we predict the output $Y$ via the model

$$
\hat{Y} = \hat{\beta}_0 + \sum_{i=1}^pX_i\hat{\beta}_i
$$

The term $\hat{\beta}_0$ is the intercept, also known as the _bias_ in machine learning. Often it is convenient to include the constant variable $1$ in $X$, include $\hat{\beta}_0$ in the vector of coefficients $\hat{\beta}$, and then write the linear model in vector form as 

$$
\hat{Y} = X^t\hat{\beta}
$$

Here we are modeling a single output, so $\hat{Y}$ is a scalar; in general $\hat{Y}$ can be a $K$-vector, in which case $\beta$ would be a $p\times K$ matrix of coefficients. In the $(p+1)$-dimensional input-output space, $(X,\hat{Y})$ represents a hyperplane. 

We assume that the intercept is included in $\hat{\beta}$.

Viewed as a function over the $p$-dimensional input space, $f(X)=X^t\beta$ is linear, and the gradient $f'(X)=\beta$ is a vector in input space that points in the steepest uphill direction.

How do we fit the linear model to a set of training data? There are many different methods, but by far the most popular is the method of _least squares_. in this approach, we pick the coefficients $\beta$ to minimize the residual sum of squares

$$
\text{RSS}(\beta) = \sum_{i=1}^N(y_i-x_i^t\beta)^2
$$

RSS$(\beta)$ is a quadratic function of the parameters, and hence its minimum always exists, but may not be unique. The solution is easiest to characterize in matrix notation. We can write

$$
\text{RSS}(\beta) = (\bold{y}-\bold{X}\beta)^t(\bold{y}-\bold{X}\beta),
$$
where $\bold{X}$ is an $N\times p$ matrix with each row an input vector, and $\bold{y}$ is an $N$-vector of the outputs in the training set. Differentiating w.r.t. $\beta$ we get the _normal equations_

$$
\bold{X}^t(\bold{y}-\bold{X}\beta)=0
$$

If $\bold{X}^t\bold{X}$ is nonsingular, then the unique solution is given by

$$
\hat{\beta} = (\bold{X}^t\bold{X})^{-1}\bold{X}^t\bold{y}
$$
and the fitted value at the $i$th input $x_i$ is $\hat{y}_i=\hat{y}(x_i)=x_i^t\hat{b}$. The entire fitted surface is characterized by the $p$ parameters $\hat{\beta}$. Intuitively, it seems that we do not need a very large data set to fit such a model.