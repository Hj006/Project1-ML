# Project 1

## Team Members:

- **Aakash Shivanandappa Gowda**  
  A20548984
- **Dhyan Vasudeva Gowda**  
  A20592874
- **Hongi Jiang**  
  A20506636
- **Purnesh Shivarudrappa Vidhyadhara**  
  A20552125

## 1. Overview

This project implements an **Elastic Net regression model** (ElasticNetModel) that combines L1 (Lasso) and L2 (Ridge) regularization techniques. It is designed to handle high-dimensional datasets, feature selection, and multicollinearity problems. The model uses **gradient descent** for optimization, function consists of the MSE loss, L1 loss, L2 loss.

## 2. Class and Function Descriptions

### `ElasticNetModel`

The `ElasticNetModel` class implements the Elastic Net regression model. Below are the key methods and their descriptions:

#### `__init__(self, lambdas=0.1, thresh=0.5, max_iter=1000, tol=1e-4, learning_rate=0.01)`

Initializes the Elastic Net model with the following parameters:

- **`lambdas`** (_float_, default=0.1): Penalty coefficient for regularization.
- **`thresh`** (_float_, default=0.5): Mixing parameter between L1 and L2 regularization. Value ranges from 0 to 1, where 0 means L1 regularization only and 1 means L2 regularization only.
- **`max_iter`** (_int_, default=1000): Maximum number of iterations for gradient descent.
- **`tol`** (_float_, default=1e-4): Tolerance for the stopping condition. If changes in weights are smaller than `tol`, training stops.
- **`learning_rate`** (_float_, default=0.01): Step size for gradient descent updates.

#### `fit(self, X, y)`

Trains the Elastic Net regression model using gradient descent on the provided feature matrix `X` and target values `y`.

- **Parameters**:
  - **`X`** (_numpy array_): Feature matrix of the training dataset (without intercept).
  - **`y`** (_numpy array_): Target values corresponding to the training dataset.
- **Returns**:
  - An instance of `ElasticNetResults`, containing the trained weight coefficients and intercept.

### `ElasticNetResults`

This class stores the results after training the Elastic Net model.

#### `predict(self, X)`

Predicts target values for the provided feature matrix `X`.

- **Parameters**:
  - **`X`** (_numpy array_): Feature matrix of the test dataset.
- **Returns**:
  - Predicted target values.

### Example Usage

```python
from elasticnet/models import ElasticNetModel

# Initialize the model
model = ElasticNetModel(lambdas=0.1, thresh=0.5, max_iter=1000, tol=1e-4, learning_rate=0.01)

# Fit the model on training data
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)
```

### `generate_negative_data`

The `generate_negative_data` function generates synthetic data with two distinct patterns: 
1. The first half of the features has a **monotonic increasing trend**.
2. The second half of the features has a **linear decreasing trend** (i.e., negative slope).

This function can be useful for testing model's ability in negative correlations data.

#### Parameters:
- **`range_x`** (_tuple_): The range of feature values (min, max).
- **`noise_scale`** (_float_): Standard deviation of the Gaussian noise added to the data.
- **`size`** (_int_): Number of samples in the dataset.
- **`num_features`** (_int_): Total number of features (half with increasing trend, half with decreasing trend).
- **`seed`** (_int_): Random seed for reproducibility.

#### Returns:
- **`X`** (_numpy array_): Generated feature matrix with both increasing and decreasing trends.
- **`y`** (_numpy array_): Target values with contributions from the features and added noise.

#### Example Usage:

```python
from generate_negative_regression_data import generate_negative_data

# Generate negative slope data
X, y = generate_negative_data(range_x=(0, 10), noise_scale=1.0, size=100, num_features=6, seed=42)
```

---

### `generate_rotated_positive_data`

The `generate_rotated_positive_data` function generates synthetic data with two patterns:
1. The first half of the features follows a **monotonic increasing trend**.
2. The second half exhibits a **wavy (S-shaped) pattern**, adjusted by a rotation matrix to create a slanted shape.

This function generates more complex data to test model's ability.

#### Parameters:
- **`range_x`** (_tuple_): Specifies the range of feature values (min, max).
- **`noise_scale`** (_float_): Standard deviation of the Gaussian noise added to the data.
- **`size`** (_int_): Number of samples in the dataset.
- **`num_features`** (_int_): Total number of features (half with increasing trend, half with a wavy pattern).
- **`seed`** (_int_): Random seed for reproducibility.
- **`rotation_angle`** (_float_, default=45): Angle (in degrees) to rotate the wavy pattern, introducing a slanted S-shape.
- **`mode`** (_int_, default=0): Determines the scaling factors for the feature values.

#### Returns:
- **`X`** (_numpy array_): Generated feature matrix with increasing trends and rotated S-shaped patterns.
- **`y`** (_numpy array_): Target values influenced by the features and noise.

#### Example Usage:

```python
from generate_positive_regression_data import generate_rotated_positive_data

# Generate positive data with rotation and S-shaped curves
X, y = generate_rotated_positive_data(range_x=(0, 10), noise_scale=1.0, size=100, num_features=6, seed=42, rotation_angle=45, mode=0)
```

---


# 1. Linear regression with ElasticNet regularization (combination of L2 And L1 regularization)

## Q1.What does the model you have implemented do and when should it be used?

### ElasticNetModel Overview

The model we've crafted, called **ElasticNetModel**, is an implementation of Elastic Net regression. What makes it special is that it not only uses MSE as the loss function, but also combines L1 (Lasso) and L2 (Ridge) regularization techniques. Here's a clearer view on how we implement this model:

### Loss Function

The loss function consists of the MSE loss, L1 loss, and L2 loss. The loss function is shown as follows:

$\text{Loss} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^m |w_j| + \lambda_2 \sum_{j=1}^m w_j^2$

### Then use the gradient descent algorithm to calculate the gradient of the loss function and then update.

- **Gradient calculation results of MSE:**:

$\nabla_{\text{MSE}} = -\frac{2}{n} X^T (\hat{y} - y)$

- **Gradient calculation results of L1:**:

$\nabla_{\text{L1}} = \lambda_1 \cdot \text{sign}(w)$

- **Gradient calculation results of L2:**:

$\nabla_{\text{L2}} = 2 \cdot \lambda_2 \cdot w$

- **Total Gradient**:

$\nabla_{\text{Total}} = -\frac{2}{n} X^T (\hat{y} - y) + \lambda_1 \cdot \text{sign}(w) + 2 \cdot \lambda_2 \cdot w$

- **Weight Update**:

$w = w - \eta \cdot \nabla_{\text{Total}}$

$\eta \text{ is the learning rate, } \lambda_1 \text{ and } \lambda_2 \text{ are the penalty coefficients}$

### ElasticNetModel Overview

In summary, our model implements hybrid regularization by combining L1 and L2 penalties. This simplifies feature selection and improves model stability. Additionally, we uses gradient descent to efficiently update parameters and weights.

### What the Model Does:

- **Hybrid Regularization**: Elastic Net stands out because it uses a mix of L1 and L2 penalties.

  - **L1 penalty** helps thin out the less essential features, essentially cleaning up the model to focus on what really matters, making it simpler and easier to understand. This is especially useful when you’re dealing with loads of potential predictors and want to avoid a model that’s too complex to interpret or too tailored to the training data.
  - **L2 penalty** helps when features are intertwined, which is quite common. It does this by distributing the influence of these features more evenly, which helps keep the model stable and reliable.

- **Gradient Descent Optimization**: This technique is crucial for fine-tuning the model in big datasets where older methods might be too slow. It adjusts the model gradually and methodically to hone in on the best possible performance, based on the rate of learning and how precise we want the model to be.

### When to Use This Model:

- **Broad Application Spectrum**: This model shines in any scenario where you need precise and sturdy predictions. It’s fantastic for fields like economics or healthcare, where the data and relationships can be quite complex.

- **Managing Complexity and Ensuring Accuracy**: It’s perfect for diving into complicated, noisy data and coming out with clear insights. Whether it’s studying genes or predicting trends in big industries, this model helps manage a lot of information without losing focus.

The **ElasticNetModel** isn’t just about making predictions; it’s also about deeply understanding the data. It’s a powerful tool for anyone who needs to tackle complex data sets and derive meaningful conclusions, not just numbers. This model’s ability to simplify while handling complex relationships makes it invaluable for researchers and analysts who regularly face challenging modeling situations.

## Q2.How did you test your model to determine if it is working reasonably correctly?

### Model Testing and Validation Strategy

To ensure our model is both robust and accurate, we’ve put a thorough testing and validation strategy into play. Let’s walk through each step of this strategy to see how it contributes to a full evaluation of the model’s performance:

### 1. Synthetic Data Generation

- **Method**: We created synthetic datasets using a method called `generate_rotated_s_wave_data` that clearly defines how features relate to target values.
- **Why It Matters**: This setup lets us dive deep into the model’s ability to recognize and learn the patterns we’ve set up. By working with synthetic data, where the outcomes are already known, we can rigorously test how well the model mimics these set relationships, giving us a clear picture of its learning precision.

### 2. Training on Generated Data

- **What We Did**: The model spent a good deal of time learning from these synthetic datasets. During this phase, we kept a close eye on its parameters, checking how well the learned coefficients and intercepts matched the expected patterns.
- **Why It Matters**: This step is crucial for confirming that the model understands the underlying data dynamics. It helps validate that the model’s foundation is solid and that it’s tuned to respond appropriately to the nuances it will face in real scenarios.

### 3. Predictions on Unseen Data

- **Method**: We tested the model’s predictions using a new set of data that it hadn’t seen during training.
- **Why It Matters**: This is key for assessing how well the model can apply what it’s learned to new, unknown datasets—a must for ensuring the model is genuinely useful in real-world settings where data variability is the norm.

### 4. Model Evaluation Metrics

- **Metrics Used**: We measured the model’s accuracy and reliability using several standard regression metrics like:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)
- **Why They’re Important**: These metrics provide hard numbers that tell us how accurately the model predicts new data. They help us understand different aspects of performance, such as the average size of errors and how much of the variance in our target variable the model can explain.

### 5. Visual Comparisons of Predicted vs. Actual Values

- **How We Do It**: We use scatter plots where each point shows a predicted value against its actual value. The closer these points are to the diagonal line (where predicted equals actual), the better the model’s predictions.
- **The Benefits**: These visuals offer an instant, gut-level indication of the model’s accuracy. They’re particularly good for spotting whether the model tends to over or underestimate across the board (systematic bias) or if it’s inconsistent (high variance). They also let us see at a glance if some features yield more reliable predictions than others.

### 6. Analysis and Iteration

- **Ongoing Process**: Based on what we learn from the visual data and other metrics, we continuously refine the model. This might mean tweaking the model’s settings to better align predictions with actual results, especially where we notice gaps.
- **Why Keep At It**: These refinements help the model adapt to the subtleties of new data, ensuring it stays relevant and effective. This ongoing cycle of feedback and improvement is vital for the model to remain at the top of its game as it encounters new challenges.

### Insights from Graphs:

Observing how “Feature 2 vs. Target” shows a tighter cluster around the diagonal than “Feature 6 vs. Target” suggests Feature 2 is being handled more precisely. This might lead us to look into whether Feature 6 needs more preprocessing or whether the model should employ advanced feature engineering strategies to better capture complex data patterns.

By weaving together these detailed testing methods, we not only affirm that the model is theoretically sound but also ensure it’s equipped to perform reliably in a range of real-world applications. This rigorous validation and refinement process is crucial for building top-quality predictive models that can rise to new challenges and deliver dependable results.

## Q3.What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

### Model Optimization Parameters

To optimize the performance of our model, we’ve provided several adjustable parameters. Here’s a breakdown of each parameter, explaining their function and how adjustments can enhance model performance:

### 1. Lambdas

- **Function**: Lambdas are central to our model’s regularization strategies, utilized in both L1 (Lasso) and L2 (Ridge) techniques. This helps prevent overfitting, a common issue where the model learns noise as if it were true data signals.
- **Impact of Adjustment**: Increasing the lambdas value amplifies the penalty on the model’s coefficients, effectively pulling them towards zero. This adjustment is beneficial when dealing with complex datasets that could potentially lead to misleading noise.
- **When We Use It**: If we observe our model overfitting to training data, boosting the lambdas value can help mitigate this, promoting better generalization to new, unseen data.

### 2. Thresh

- **Function**: Thresh determines the cutoff for feature selection, which simplifies the model by focusing only on impactful features.
- **Impact of Adjustment**: A higher thresh value is likely to eliminate less significant features by setting their coefficients to zero, which can enhance both the interpretability and performance of the model on new data.
- **When We Use It**: If our dataset has an abundance of features, adjusting thresh helps us streamline the model, concentrating on the most influential factors for predictions.

### 3. Max_iter

- **Function**: This parameter sets the maximum number of iterations our algorithm will execute before stopping.
- **Impact of Adjustment**: Allowing more iterations gives the model additional opportunities to refine its weights to the data, potentially improving its accuracy.
- **When We Use It**: If our model hasn’t yet reached its optimal state, increasing `max_iter` may be necessary. However, this could lead to longer training times, requiring a balance between precision and efficiency.

### 4. Tol

- **Function**: Tol, or tolerance, defines how close the model needs to fit the training data trends before ceasing further adjustments.
- **Impact of Adjustment**: Setting a smaller `tol` value means the model won’t stop learning until it achieves a closer fit, which can enhance accuracy but also extend training duration.
- **When We Use It**: This is particularly valuable when high precision is crucial. However, setting the tolerance too tight might lead to diminishing returns, especially in terms of prolonged training without significant gains.

### 5. Learning_rate

- **Function**: In gradient descent optimization, this parameter controls the step size the model takes towards minimizing the loss function.
- **Impact of Adjustment**: Lowering the learning rate results in smaller, more cautious steps, potentially leading to more precise model optimization.
- **When We Use It**: If the model is overshooting the minimal loss point, reducing the learning rate can help achieve a more stable and accurate convergence. However, too low a rate may slow down the training process excessively.

By fine-tuning these parameters, we can adjust our model to better fit the specific characteristics of the data and the demands of the task at hand, striking an optimal balance between accuracy, computational efficiency, and model simplicity.

## Q4.Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

### Specific Inputs That May Challenge the ElasticNet Model

### 1. Highly Nonlinear Data

- **The Challenge**: Elastic Net is designed to see the world in straight lines, assuming a straightforward, linear relationship between features and the target variable. However, life often isn’t that simple, and data can show complex, curved relationships that the model struggles to grasp.
- **What Happens**: When faced with these complex patterns, Elastic Net might not hit the mark with its predictions since it can’t effectively capture the twists and turns in variable interactions.
- **What We Can Do**:
  - **Feature Engineering**: To help the model better understand the underlying patterns, we can transform the input features. This might mean squaring or cubing features, or introducing interaction terms, and even applying mathematical transformations like logarithms or exponentials to straighten out those nonlinear relationships.
  - **Kernel Methods**: Borrowing a trick from the support vector machine playbook, we can use kernel methods to project our data into a space where the once-complex relationships look linear, making it easier for Elastic Net to do its job.
  - **Hybrid Models**: Sometimes, two heads (or models) are better than one. By teaming up Elastic Net with nonlinear models like decision trees or neural networks, we can combine their strengths, enhancing the model’s ability to predict accurately across a wider range of scenarios.

### 2. Noisy Data

- **The Challenge**: Noisy data is like trying to listen to a quiet radio station during a thunderstorm. The static (noise) can drown out the music (true patterns), making it hard for Elastic Net to hear what’s important.
- **What Happens**: If the data is too noisy, Elastic Net might start learning the static instead of the music, leading to models that perform well on their training data but falter in the real world.
- **What We Can Do**:
  - **Increase Regularization**: By turning up the lambdas knob, we increase the penalties for complexity, encouraging the model to focus on the loud and clear signals and ignore the static.
  - **Threshold Tuning**: Adjusting the thresh parameter helps us be stricter about which features we let into the model, cutting out those that are more noise than signal.
  - **Noise Reduction Techniques**: Before we even feed data into our model, we can clean it up—filtering out oddball values or smoothing out jumpy signals—to make sure we’re working with the cleanest, clearest data possible.

By understanding these limitations and employing smart strategies to mitigate them, we can enhance the capabilities of Elastic Net, making it a more versatile and reliable tool in our predictive modeling toolkit.
