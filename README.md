### L2 Regularized Logistic Regression Model Implementation

The python file l2_regularized_logistic_regression.py is a from-scratch (using numpy) implementation of L2 Regularized Logistic Regression (Logistic Regression with the Ridge penalty).  The accompanying demo .ipynb files provide the following examples of using the from-scratch model:

- Classifying some simple, simulated data.
- Classifying tumors in the Wisconsin Breast Cancer Dataset as benign or cancerous.  This demo also includes how to implement my from-scratch K fold cross-validation method to find the optimal lambda penalty for the regression model.
- Comparing my model built with numpy to scikit-learn's L2 Regularized Logistic Regression model.

Note that the dataset comes prepackaged with scikit-learn and is imported in the demo files.  The core functionality of the l2_regularized_logistic_regression.py module demonstrated in the example notebooks includes:

- Training the model
- Visualizing the training process
- Examining misclassification error on simulated and real data
- Setting the optimal L2 penalty using an implementation of K fold cross-validation

**Dataset:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

**Author:** Joel Stremmel (jstremme@uw.edu)

**Credits:** University of Washington DATA 558 with Zaid Harchaoui and Corinne Jones

