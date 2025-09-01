import pandas as pd
from sklearn import preprocessing

def get_adult_data():
# Define column names
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    # Load the .txt file (comma-separated)
    df = pd.read_csv('../datasets/adult/adult.data', names=columns, sep=r'\s*,\s*', engine='python', skiprows=1)
    df_test = pd.read_csv("../datasets/adult/adult.test",names=columns, sep=r'\s*,\s*', engine='python', skiprows=1)


    # Create training and testing variables
    Y_train = (df["income"] == ">50K").astype(int)
    Y_sen_train = (df["sex"] == "Female").astype(int)   # Encode sex: Female -> 1, Male -> 0
    X_train = df.drop(columns=["sex", "income", "fnlwgt"])      # Prepare X_train (features without sensitive attribute and without target)
    X_train = pd.get_dummies(X_train, drop_first=True)      # One-hot encode categorical variables

    Y_test = (df_test["income"] == ">50K.").astype(int)
    Y_sen_test = (df_test["sex"] == "Female").astype(int)
    X_test = df_test.drop(columns=["sex", "income", "fnlwgt", "capital-gain", "capital-loss"])
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    print(Y_test.unique())
    # Align columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1)

    # Fill NaNs from missing categories in test with 0
    X_test = X_test.fillna(0)
    Y_test = Y_test.loc[X_test.index]
    Y_sen_test = Y_sen_test.loc[X_test.index]
    
    #Standardize numeric feature to mean=0 and std=1
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(exclude=["int64", "float64"]).columns

    # Initialize scaler
    scaler =  preprocessing.StandardScaler()

    # Fit and transform only numeric columns
    X_train_num_scaled = scaler.fit_transform(X_train[num_cols])

    # Convert back to DataFrame to keep column names
    X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=num_cols, index=X_train.index)

    # Concatenate scaled numeric columns with untouched categorical columns
    X_train_scaled = pd.concat([X_train_num_scaled, X_train[cat_cols]], axis=1)

    X_test_num_scaled = scaler.transform(X_test[num_cols])
    X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=num_cols, index=X_test.index)
    X_test_scaled = pd.concat([X_test_num_scaled, X_test[cat_cols]], axis=1)


    return X_train_scaled, Y_train, Y_sen_train, X_test_scaled, Y_test, Y_sen_test