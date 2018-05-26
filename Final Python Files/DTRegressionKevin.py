# Load the required Python machine learning packages
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ModelEvaluation

def CreateModel():
    # Using the pandas dataframes to import the dataset
    data = pd.read_csv('student-maths.csv')
    data.head()
    
    # Extract the values of the columns and label as X and y 
    X = ((data.iloc[:, 0:6]).values).astype(float)
    y = ((data.iloc[:, 6]).values).astype(float)
    
    # Split the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)
    
    # Infer the decision tree regressor
    rgrs = DecisionTreeRegressor(max_depth=4, max_features=None, random_state=100)
    
    # Standardising features by removing mean and scaling
    pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', rgrs)])
    
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)) # Fit the train data and transform into decision tree
    pipeline.fit(X_train, y_train.ravel()) 
    y_pred = y_scaler.inverse_transform(pipeline.predict(X_test)) # Reverses the scale
    
    # Apply K-Fold Cross Validation
    accuracies = cross_val_score(rgrs, X_train, y_train, cv=10)
    
    
    # Export the decision tree into a dot file to use on WebGraphviz
    export_graphviz(rgrs, out_file = 'features.dot',
                                    feature_names=['age','studytime','failures','absences', 
                                                   'g1','g2'])
        
    ModelEvaluation.ModelEvaluation(y_test, y_train, y_pred, "DECISION TREE REGRESSION", accuracies)
    
