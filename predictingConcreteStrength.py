#  Author: Priyadharsshini Sakrapani

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def Q1_results():
    train_df = pd.read_csv("train.csv")
    # Split the training data into features and target
    X = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]

    cvrse5 = 0
    cvr25 = 0

    # Define the linear regression model
    lr = LinearRegression()

    # Validation approach
    val_splits = [0.6, 0.5, 0.4, 0.3, 0.2]
    for i, split in enumerate(val_splits):
        # Split the training data into train/validation sets
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=(1-split), random_state=42)

        # Train the model on the training + validation sets
        lr.fit(X_train1, y_train1)
    #   lr.fit(X_val, y_val)

        # Make predictions on the test set
        y_pred = lr.predict(X_test1)
        rse = np.sqrt(mean_squared_error(y_test1, y_pred))
        r2 = r2_score(y_test1, y_pred)

    
        # Print the results
        print(f'Validation Approach ({i+1}): Train/Validation/Test Split: {split:.1f}/{1-split:.1f}/0.2')
        print(f'RSE: {rse:.2f}')
        print(f'R2: {r2:.2f}\n')
    
    # Define the number of folds for cross-validation
    kfolds = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    print("\nScores using cross-validation approach")

    # Train the linear regression model using k-fold cross-validation
    for k in kfolds:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        rse = np.sqrt(-cross_val_score(lr,X_train1,y_train1,cv=kf,scoring='neg_mean_squared_error').mean())
        r2 = cross_val_score(lr,X_train1,y_train1,cv=kf,scoring='r2').mean()
    
        if k == 5:
            cvrse5 = rse
            cvr25 = r2
    
        # Print the performance metrics for the current value of k
        print(f"CV RSE for {k} folds:", rse)
        print(f"CV R2 for {k} folds:", r2,"\n")

def Q2_results():
    global ridge_rse
    global ridge_r2
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    # Splitting the features and target variable
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    X = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.8, random_state=42)

    # Perform hyperparameter tuning using GridSearchCV
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 300, 500, 700, 1000]}
    ridge = Ridge()
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Calculate the RSE and R-squared for each alpha value
    rse_values = []
    r2_values = []
    for alpha in param_grid['alpha']:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train1, y_train1)
        y_pred = ridge.predict(X_test1)
        rse = np.sqrt(mean_squared_error(y_test1, y_pred))
        r2 = r2_score(y_test1, y_pred)
        rse_values.append(rse)
        r2_values.append(r2)

    # Print the RSE and R-squared for each alpha value
    for alpha, rse, r2 in zip(param_grid['alpha'], rse_values, r2_values):
        print(f'alpha = {alpha}: RSE = {rse:.4f}, R-squared = {r2:.4f}')

    # Print the best hyperparameters and the corresponding score
    best_alpha = grid_search.best_params_['alpha']
    print("\nBest alpha:", best_alpha)

    # Train the final Ridge regression model using the best hyperparameters
    final_ridge = Ridge(alpha=best_alpha)
    final_ridge.fit(X_train, y_train)

    # Calculate the RSE and R^2 of the final Ridge regression model on the test set
    y_pred = final_ridge.predict(X_test)
    ridge_rse = np.sqrt(mean_squared_error(y_test, y_pred))
    ridge_r2 = r2_score(y_test, y_pred)

    # Compare the performance of the Ridge regression model with that of the simple linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_rse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)


    # Plot the performance of the models explored during the alpha hyperparameter tuning phase as a function of alpha
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 300, 500, 700, 1000]
    plt.plot(alphas, rse_values, "o-", label="Ridge")
    # plt.plot([0, 1000], [lr_rse, lr_rse], "--", label="Linear")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("RSE")
    plt.title("Ridge Regression")
    plt.legend()
    plt.show()

    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 300, 500, 700, 1000]
    plt.plot(alphas, r2_values, "o-", label="Ridge")
    # plt.plot([0, 1000], [lr_rse, lr_rse], "--", label="Linear")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("r-Squared")
    plt.title("Ridge Regression")
    plt.legend()
    plt.show()

    # Print the RSE and R^2 of the Ridge regression model
    print("Ridge Regression RSE:", ridge_rse)
    print("R2 score for Ridge Regression:", ridge_r2)

    # Print the RSE and R^2 of the Linear regression model using CV approach
    print("Linear Regression RSE:", lr_rse)
    print("R2 score for Linear Regression:", lr_r2)

def Q3_results():
    global lasso_rse
    global lasso_r2
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    # Splitting the features and target variable
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    X = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.8, random_state=42)
    # Perform hyperparameter tuning using GridSearchCV
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 300, 500, 700, 1000]}
    lasso = Lasso()
    grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring='r2')
    grid_search_lasso.fit(X_train, y_train)

    # Calculate the RSE and R-squared for each alpha value
    lasso_rse_values = []
    lasso_r2_values = []
    for alpha in param_grid['alpha']:
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train1, y_train1)
        y_pred = lasso.predict(X_test1)
        rse = np.sqrt(mean_squared_error(y_test1, y_pred))
        r2 = r2_score(y_test1, y_pred)
        lasso_rse_values.append(rse)
        lasso_r2_values.append(r2)

    # Print the RSE and R-squared for each alpha value
    for alpha, rse, r2 in zip(param_grid['alpha'], lasso_rse_values, lasso_r2_values):
        print(f'alpha = {alpha}: RSE = {rse:.4f}, R-squared = {r2:.4f}')

    # Print the best hyperparameters and the corresponding score
    best_alpha = grid_search_lasso.best_params_['alpha']
    print("\nBest alpha:", best_alpha)

    # Train the final Lasso regression model using the best hyperparameters
    final_lasso = Lasso(alpha=best_alpha)
    final_lasso.fit(X_train, y_train)

    # Calculate the RSE and R^2 of the final Lasso regression model on the test set
    y_pred = final_lasso.predict(X_test)
    lasso_rse = np.sqrt(mean_squared_error(y_test, y_pred))
    lasso_r2 = r2_score(y_test, y_pred)

    # Compare the performance of the Ridge regression model with that of the simple linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_rse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)


    # Plot the performance of the models explored during the alpha hyperparameter tuning phase as a function of alpha
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 300, 500, 700, 1000]
    plt.plot(alphas, lasso_rse_values, "o-", label="Lasso")
    # plt.plot([0, 1000], [lr_rse, lr_rse], "--", label="Linear")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("RSE")
    plt.title("Lasso Regression")
    plt.legend()
    plt.show()

    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 300, 500, 700, 1000]
    plt.plot(alphas, lasso_r2_values, "o-", label="Lasso")
    # plt.plot([0, 1000], [lr_rse, lr_rse], "--", label="Linear")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("RSquared")
    plt.title("Lasso Regression")
    plt.legend()
    plt.show()

    # Print the RSE and R^2 of the Lasso regression model
    print("Lasso Regression RSE:", lasso_rse)
    print("R2 score for Lasso Regression:", lasso_r2)
    print("\n")

    # Print the RSE and R^2 of the Ridge regression model
    print("Ridge Regression RSE:", lasso_rse)
    print("R2 score for Ridge Regression:", ridge_r2)
    print("\n")

    # Print the RSE and R^2 of the Linear regression model using CV approach
    print("Linear Regression RSE:", lr_rse)
    print("R2 score for Linear Regression:", lr_r2)

def predictCompressiveStrength(Xtest, data_dir):
    
    path = data_dir + "train.csv"
    # path1 = data_dir + "test.csv"

    train_df = pd.read_csv(path)

    test_df = Xtest
    # test_df = pd.read_csv(path1)

    # Remove missing data and outliers using z score
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    z_scores_train = np.abs((train_df - train_df.mean()) / train_df.std())
    z_scores_test = np.abs((test_df - test_df.mean()) / test_df.std())
    train_df = train_df[(z_scores_train < 3).all(axis=1)]
    test_df = test_df[(z_scores_test < 3).all(axis=1)]


    # Splitting the train and test after outlier removal
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # selecting the best model
    models = [
        LinearRegression(),
        Ridge(alpha=0.1),
        Lasso(alpha=0.1),
        SVR(kernel='linear'),
        RandomForestRegressor(),
        GradientBoostingRegressor()
    ]

    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(type(model).__name__, "RSE:", rse)
        print(type(model).__name__, "R2 score:", r2) 
        print("\n")

    best_model = min(models, key=lambda model: np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
    print(f"Best model: {type(best_model).__name__}")

    # Evaluate the best model
    best_model.fit(X_train, y_train)
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print("\nTrain set evaluation:")
#     print("RSE:", np.sqrt(mean_squared_error(y_train, y_pred_train))
    print("RSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))

#     print("RSE:", np.sqrt(mean_squared_error((y_train - y_pred_train))))
    print("R2 score:", r2_score(y_train, y_pred_train))

    print("\nTest set evaluation:")
#     print("RSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print("RSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print("R2:", r2_score(y_test, y_pred_test))

    rse_randomForest = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_randomForest = r2_score(y_test, y_pred_test)


    fig, ax = plt.subplots()
    ax.plot(['Random Forest(Final Improved)', 'Lasso', 'Ridge'], [r2_randomForest, lasso_r2, ridge_r2])
    ax.set_ylim([0, 1])
    ax.set_ylabel('R-squared score')
    ax.set_title('Comparison of R-squared scores for three regression models')
    

    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(['Random Forest(Final Improved)', 'Lasso', 'Ridge'], [rse_randomForest, lasso_rse, ridge_rse])
    ax.set_ylabel('RSE')
    ax.set_title('Comparison of RSE for three regression models')
    
    print("Improved RSE score:", rse_randomForest)
    print("Improved R2 score:", r2_randomForest)
    return y_pred_test

if __name__ == "__main__":  
    Q1_results()
    Q2_results()
    Q3_results()
    # predictCompressiveStrength(Xtest,"file:///Users/priyadharsshinis/" )





