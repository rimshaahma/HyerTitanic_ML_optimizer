Titanic Survival Prediction & Hyperparameter Tuning Application
This project provides an interactive GUI to predict Titanic survival using machine learning models, specifically Logistic Regression and Decision Trees. The application also includes functionality for hyperparameter tuning to optimize the model’s performance. The GUI is built using Tkinter, and it allows users to upload a dataset, train the model, and view the best-performing hyperparameters and accuracy.

Features
Upload CSV: Upload your Titanic dataset (train.csv).
Start Training & Tuning: Initiates model training and hyperparameter optimization.
Model Performance: Displays the best hyperparameters found during tuning and the model’s accuracy.
Technical Stack
Python: Programming language used.
Tkinter: GUI toolkit.
pandas: Data manipulation and analysis.
scikit-learn: Machine learning models and tools.
matplotlib: Data visualization (if you wish to include any plots in future versions).
Installation
Step 1: Clone the Repository
Clone the repository to your local machine using Git:

bash
Copy code
git clone https://github.com/yourusername/titanic-ml.git
Step 2: Install Dependencies
Ensure you have Python 3.x installed. Then, install the required libraries using pip:

bash
Copy code
pip install pandas scikit-learn matplotlib tkinter
How to Use
Upload the Titanic Dataset:
Click the "Upload CSV" button.
Select your train.csv file (you can download it from Kaggle Titanic Dataset).
Start Model Training and Hyperparameter Tuning:
Click the "Start Training & Tuning" button.
The program will train a model and optimize hyperparameters using GridSearchCV and RandomizedSearchCV.
View Results:
After tuning, the best hyperparameters and model accuracy will be displayed.
Model and Hyperparameter Tuning
Logistic Regression
A type of regression used for binary classification problems. In this case, it predicts whether a passenger survived (Survived=1) or not (Survived=0). Logistic Regression uses a logistic function to model the probability that a given input point belongs to a certain class.

Decision Tree Classifier
A decision tree is a non-linear model that makes decisions based on answering a series of "if-else" questions. It builds a tree where each node represents a question about the features, and each branch represents the answer.

Hyperparameter Tuning
Hyperparameters are the settings or configurations that govern the model training process. For example:

Logistic Regression Hyperparameters: C (regularization strength), solver (algorithm to use), and penalty (regularization type).
Decision Tree Hyperparameters: max_depth (maximum depth of the tree), min_samples_split (minimum samples required to split a node), and min_samples_leaf (minimum samples required in each leaf).
GridSearchCV
GridSearchCV is a method used to tune the hyperparameters of a model. It exhaustively searches through a manually specified hyperparameter grid and evaluates all possible combinations to find the best model.

Example: You could search for the best C value for Logistic Regression from the list [0.01, 0.1, 1, 10, 100].
RandomizedSearchCV
RandomizedSearchCV is similar to GridSearchCV, but it samples a fixed number of random combinations of hyperparameters from the specified grid. It’s faster than GridSearchCV because it doesn’t evaluate all possible combinations, but instead, evaluates only a subset.

Cross-Validation
In this project, 5-fold cross-validation is used. This means the dataset is split into 5 subsets. The model is trained on 4 subsets and tested on the remaining one, and this is repeated 5 times (each subset is used as a test set once). The average of these tests provides a robust estimate of model performance.

Accuracy
The accuracy is the proportion of correctly predicted labels (Survived=1 or Survived=0) from the total number of predictions.

Code Structure
1. titanic_model.py
Contains functions for:

Loading the data (load_data).
Preprocessing the data (filling missing values, encoding categorical variables, etc.).
Performing hyperparameter tuning using GridSearchCV and RandomizedSearchCV (tune_model).
2. titanic_eda_gui.py
The main GUI application. It:

Allows users to upload a CSV file.
Runs the model training and hyperparameter tuning process.
Displays the best hyperparameters and model accuracy.
Output
After performing hyperparameter tuning, the GUI will display the following information:

Best Hyperparameters: The hyperparameters that resulted in the best model performance.
Accuracy: The accuracy score of the model on the test dataset.
Here’s an example output:

css
Copy code
Best Hyperparameters: {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'}
Accuracy: 0.82
Screenshots
Here’s a screenshot of the application in action:


Conclusion
This project demonstrates the process of building an interactive GUI application for Titanic survival prediction and hyperparameter optimization. It uses machine learning models like Logistic Regression and Decision Trees and optimizes them using techniques like GridSearchCV and RandomizedSearchCV.

Troubleshooting
Issue: The model doesn’t seem to train or gives an error.

Solution: Ensure the dataset is correctly formatted, and there are no missing or incorrectly typed columns in your CSV file.
Issue: The hyperparameter tuning is taking too long.

Solution: Reduce the number of hyperparameters in the search grid or use RandomizedSearchCV instead of GridSearchCV for faster results.
License
This project is licensed under the MIT License.

How to add the output image to your GitHub repository:
Place the Image in the Repository: Move the output.png image to your repository folder (e.g., inside a folder called assets).
Update the README: Update the image path in the README file to reflect the correct location. For example:
markdown
Copy code
![Output Screenshot](assets/output.png)
