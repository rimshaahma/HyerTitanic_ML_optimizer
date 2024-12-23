import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Function to load data from CSV
def load_data(file_path):
    # Read the CSV data
    data = pd.read_csv(file_path)
    
    # Fill missing values (example, you can modify based on your dataset)
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # Drop unnecessary columns
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Convert categorical variables into dummy variables
    data = pd.get_dummies(data, drop_first=True)
    
    # Extract features and target
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    return X, y

# Function to tune model using GridSearchCV and RandomizedSearchCV
def tune_model(X, y, model_type='logistic'):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features (important for models like Logistic Regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Choose the model
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l2', 'none']
        }
        search = GridSearchCV(model, param_grid, cv=5)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10]
        }
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, random_state=42)
    
    # Perform the hyperparameter search
    search.fit(X_train, y_train)
    
    # Get the best parameters and accuracy
    best_params = search.best_params_
    best_model = search.best_estimator_
    
    # Predict on the test data
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return best_params, accuracy
