# Penguin Species Prediction Using Logistic Regression

## Live Application

The trained model is available as an interactive Streamlit application.

[**Try the Penguin Species Prediction App**](https://adelie-and-gentoo-penguin-species-prediction.streamlit.app/)

The application allows users to enter penguin measurements and predict whether a penguin belongs to the **Adelie** or **Gentoo** species.

---

## Project Overview

This project uses **Logistic Regression** to classify penguins as either **Adelie** or **Gentoo** using physical measurements and island information.

The project follows an end-to-end machine learning workflow, including:

- Data inspection and cleaning
- Exploratory data analysis
- Feature selection
- Train-test splitting
- Data preprocessing
- Logistic Regression modeling
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Model interpretation
- Classification threshold testing
- Model saving
- Deployment using Streamlit

The project was developed as a practical exercise to understand how Logistic Regression works in a real machine learning workflow.

---

## About the Dataset

This project uses a binary classification version of the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/).

The dataset contains information about penguins observed in the **Palmer Archipelago**, a group of islands located off the northwestern coast of the Antarctic Peninsula.

The goal of this project is to classify two penguin species:

- **Adelie Penguin (*Pygoscelis adeliae*)**
- **Gentoo Penguin (*Pygoscelis papua*)**

The dataset used in this project contains **274 observations**.

### Gentoo Penguins

Gentoo penguins are generally larger than Adelie penguins.

Some common characteristics include:

- Larger body size
- Longer flippers
- Orange-red bills
- A white stripe across the top of the head
- Strong swimming ability

Their diet mainly consists of krill, fish, and squid.

### Adelie Penguins

Adelie penguins are generally smaller than Gentoo penguins.

Some common characteristics include:

- Black and white plumage
- Shorter flippers compared with Gentoo penguins
- Dark bills
- Wide distribution around the Antarctic coast

Their diet mainly consists of krill, together with some fish and squid.

---

## Dataset Variables

The dataset contains the following variables:

| Variable | Description |
|---|---|
| `species` | Penguin species |
| `island` | Island where the penguin was observed |
| `bill_length_mm` | Bill length in millimetres |
| `bill_depth_mm` | Bill depth in millimetres |
| `flipper_length_mm` | Flipper length in millimetres |
| `body_mass_g` | Body mass in grams |
| `year` | Year of observation |

The target variable is:

```text
species
```

The two target classes are:

```text
Adelie
Gentoo
```

---

## Features Used

The final model uses the following features:

```text
island
bill_length_mm
bill_depth_mm
flipper_length_mm
body_mass_g
```

The `year` variable was excluded from the model because it was not considered a useful biological feature for identifying penguin species.

---

## Data Quality Checks

Before modeling, the dataset was checked for:

- Missing values
- Duplicate rows
- Data types
- Numerical distributions
- Categorical values
- Target class distribution

The dataset contained:

```text
274 rows
7 columns
```

No missing values or duplicate rows were found.

The target distribution was:

```text
Adelie: 151
Gentoo: 123
```

The two classes were therefore reasonably balanced.

---

## Train-Test Split

The dataset was divided into training and testing sets using:

```text
80% training data
20% testing data
```

Stratification was used to preserve the proportion of Adelie and Gentoo penguins in both sets.

The final split was:

```text
Training observations: 219
Testing observations: 55
```

A fixed `random_state` was used to make the split reproducible.

---

## Data Preprocessing

The preprocessing steps were handled using a Scikit-learn pipeline.

### Categorical Feature

The `island` feature was transformed using:

```python
OneHotEncoder(drop="first")
```

This converts the categorical island values into numerical values that Logistic Regression can use.

### Numerical Features

The numerical features were standardized using:

```python
StandardScaler()
```

The numerical features were:

- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`

### ColumnTransformer

A `ColumnTransformer` was used to apply different preprocessing steps to categorical and numerical features.

The preprocessing steps were then combined with Logistic Regression inside one complete Scikit-learn pipeline.

This helps ensure that the same preprocessing is applied during training, testing, cross-validation, and prediction.

It also helps prevent data leakage.

---

## Model

The machine learning algorithm used in this project is:

**Logistic Regression**

Logistic Regression was selected because the project is a **binary classification problem**.

The model predicts one of two classes:

```text
Adelie
Gentoo
```

---

## Baseline Model Performance

The initial Logistic Regression model was trained using its default parameters.

### Training Performance

```text
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
```

### Test Performance

```text
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
```

The confusion matrix showed that all **55 test observations** were correctly classified.

---

## Cross-Validation

To check whether the model performed consistently across different parts of the training data, **Stratified 3-Fold Cross-Validation** was used.

The following metrics were evaluated:

- Accuracy
- Precision
- Recall
- F1 Score

The model achieved:

```text
Mean Accuracy: 1.00
Mean Precision: 1.00
Mean Recall: 1.00
Mean F1 Score: 1.00
```

The results showed that the model performed consistently across all three folds.

---

## Hyperparameter Tuning

`GridSearchCV` was used to test different values of the Logistic Regression regularization parameter `C`.

The following values were tested:

```text
0.01
0.1
1
10
100
```

All tested values achieved the same mean cross-validation F1 score.

The selected value was:

```text
C = 0.01
```

The tuned model also achieved perfect performance on the held-out test set.

---

## Model Interpretation

The fitted Logistic Regression coefficients were inspected to understand how each feature influenced the model's predictions.

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| Dream Island | -0.085 | 0.918 |
| Torgersen Island | -0.082 | 0.921 |
| Bill Length | 0.343 | 1.410 |
| Bill Depth | -0.413 | 0.662 |
| Flipper Length | 0.394 | 1.483 |
| Body Mass | 0.341 | 1.406 |

In the fitted model, Gentoo is the second class.

Therefore:

- Positive coefficients shift predictions toward **Gentoo**
- Negative coefficients shift predictions toward **Adelie**

The model learned that:

- Longer flippers are associated with Gentoo predictions
- Greater body mass is associated with Gentoo predictions
- Longer bills are associated with Gentoo predictions
- Greater bill depth is associated with Adelie predictions

These are associations learned by the model and should not be interpreted as causal relationships.

---

## Classification Threshold Testing

The default Logistic Regression classification threshold is:

```text
0.5
```

Different thresholds were tested to understand how changing the threshold affects precision and recall.

### Threshold 0.3

```text
Accuracy: 0.964
Precision: 0.926
Recall: 1.00
F1 Score: 0.962
```

### Threshold 0.5

```text
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
```

### Threshold 0.7

```text
Accuracy: 0.873
Precision: 1.00
Recall: 0.72
F1 Score: 0.837
```

For this dataset, the default threshold of `0.5` produced the best overall performance on the test set.

---

## Model Saving

The complete trained Scikit-learn pipeline was saved using `joblib`.

The saved object contains:

- OneHotEncoder
- StandardScaler
- ColumnTransformer
- Logistic Regression model

The saved model is located at:

```text
models/penguin_logistic_regression.pkl
```

Saving the complete pipeline means that raw user input can be passed directly to the model without manually repeating the preprocessing steps.

---

## Streamlit Application

A Streamlit application was developed to make the trained model interactive.

Users can enter:

- Island
- Bill length
- Bill depth
- Flipper length
- Body mass

The application then returns:

- Predicted penguin species
- Adelie probability
- Gentoo probability

### Live App

[**Open the Penguin Species Prediction App**](https://adelie-and-gentoo-penguin-species-prediction.streamlit.app/)

---

## Project Structure

```text
supervised_logistic_classification_penguin_project/
│
├── .venv/
│
├── app/
│   └── app.py
│
├── data/
│   └── penguins_binary_classification.csv
│
├── models/
│   └── penguin_logistic_regression.pkl
│
├── notebooks/
│   └── 01_logistic_regression_penguins.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

> The `.venv` folder is used locally and is excluded from Git using `.gitignore`.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- Git
- GitHub

---

## Running the Project Locally

Clone the repository.

Create and activate a Python virtual environment.

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application from the project root:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

## Possible Uses

### Machine Learning Education

The project demonstrates a complete Logistic Regression classification workflow.

### Ecological Data Analysis

The penguin measurements can be used to explore physical differences between species.

### Biodiversity and Conservation Learning

The project demonstrates how machine learning methods can be applied to ecological and biodiversity datasets.

### Model Deployment Practice

The project demonstrates how a trained Scikit-learn model can be saved and deployed as an interactive web application using Streamlit.

---

## Limitations

This project uses a relatively small dataset and only includes two penguin species.

The perfect classification performance on this dataset does not mean that the model would always achieve perfect performance on completely new penguin populations.

The species in this dataset are also relatively easy to separate using the available measurements.

The project is primarily intended to demonstrate an end-to-end machine learning classification workflow.

---

## Conclusion

This project demonstrates the complete process of building and deploying a binary classification model using Logistic Regression.

The workflow covers:

- Data inspection
- Preprocessing
- Model training
- Cross-validation
- Hyperparameter tuning
- Evaluation
- Model interpretation
- Threshold analysis
- Model saving
- Web deployment

The final result is a working machine learning application that can classify **Adelie** and **Gentoo** penguins using island information and physical measurements.

## Try the Model

[**Launch the live Streamlit application**](https://adelie-and-gentoo-penguin-species-prediction.streamlit.app/)