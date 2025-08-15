import numpy as np
import pandas as pd
import random
from IPython.display import display, HTML
from sklearn.preprocessing import OrdinalEncoder
import sys
sys.path.insert(1, '/home/ptr@itd.local/code/fairness_triangle/tools')  # Update this path as needed
from preprocessing import *
from sklearn.utils import resample

def get_hdma_data(perc_original, balanced_y=False, balanced_y_bar=False):
    random.seed(42) # For reproducibility
    sample_df = pd.read_csv("../datasets/2024_public_lar.csv", skiprows=lambda i: i > 0 and random.random() > perc_original)
    
    # Remove columns with >35% missing values
    missing_ratio = sample_df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.35].index.tolist()
    cols_to_drop = cols_to_drop + ['lei', 'state_code','county_code', 'applicant_age_above_62'] 
    sample_df_clean = sample_df.drop(columns=cols_to_drop)
    must_have_val = [
        'activity_year',
        'co_applicant_ethnicity_1',
        'co_applicant_race_1',
        'census_tract',
        'loan_amount',
        'property_value',
        'income',
        'debt_to_income_ratio',
        'applicant_credit_score_type',
        'action_taken',
        'aus_1',
        'applicant_race_1',
        'applicant_ethnicity_1',
        'applicant_sex'
    ]


    sample_df_clean = sample_df_clean.dropna(subset=must_have_val)
    
    #Make Y bar binary (m/w)
    sample_df_clean['applicant_sex'].unique()
    sample_df_clean = sample_df_clean[sample_df_clean['applicant_sex'].isin([1, 2])]
    sample_df_clean['applicant_sex'] = sample_df_clean['applicant_sex'].replace({1: 0, 2: 1})
    
    #Make Y binary (loan originated/not originated)
    sample_df_clean['action_taken'] = sample_df_clean['action_taken'].apply(lambda x: 1 if x == 1 else 0)
    
    # Convert loan_term to int and remove NaN
    sample_df_clean['loan_term'] = pd.to_numeric(sample_df_clean['loan_term'], errors='coerce')
    sample_df_clean = sample_df_clean.dropna(subset=['loan_term'])
    sample_df_clean['loan_term'] = sample_df_clean['loan_term'].astype(int)
    
    # Convert loacombined_loan_to_value_ration_term to float and remove NaN
    sample_df_clean['combined_loan_to_value_ratio'] = pd.to_numeric(sample_df_clean['combined_loan_to_value_ratio'], errors='coerce')
    sample_df_clean = sample_df_clean.dropna(subset=['combined_loan_to_value_ratio'])

    # Convert interest_rate to float and remove NaN
    sample_df_clean['interest_rate'] = pd.to_numeric(sample_df_clean['interest_rate'], errors='coerce')
    sample_df_clean = sample_df_clean.dropna(subset=['interest_rate'])

    # Convert loan_term to int and remove NaN
    sample_df_clean['property_value'] = pd.to_numeric(sample_df_clean['property_value'], errors='coerce')
    sample_df_clean = sample_df_clean.dropna(subset=['property_value'])
    sample_df_clean['property_value'] = sample_df_clean['property_value'].astype(int)

    # Encoding Ordinal and Categorical Features
    sample_df_clean['total_units'] = sample_df_clean['total_units'].astype(str).str.strip()
    category_order = [['1', '2', '3', '4', '5-24', '25-49', '50-99', '>149']]
    encoder = OrdinalEncoder(categories=category_order)
    sample_df_clean['total_units'] = encoder.fit_transform(sample_df_clean[['total_units']]).astype(int)
    
    
    sample_df_clean['applicant_age'] = sample_df_clean['applicant_age'].replace('8888', np.nan)
    sample_df_clean = sample_df_clean.dropna(subset=['applicant_age'])
    age_order = [['<25', '25-34', '35-44', '45-54', '55-64', '65-74', '>74']]
    encoder = OrdinalEncoder(categories=age_order)
    sample_df_clean['applicant_age'] = encoder.fit_transform(sample_df_clean[['applicant_age']]).astype(int)


    sample_df_clean['co_applicant_age'] = sample_df_clean['co_applicant_age'].replace(['8888', '9999'], np.nan)
    sample_df_clean = sample_df_clean.dropna(subset=['co_applicant_age'])
    age_order = [['<25', '25-34', '35-44', '45-54', '55-64', '65-74', '>74']]
    encoder = OrdinalEncoder(categories=age_order)
    sample_df_clean['co_applicant_age'] = encoder.fit_transform(sample_df_clean[['co_applicant_age']]).astype(int)
    

    sample_df_clean['debt_to_income_ratio'] = sample_df_clean['debt_to_income_ratio'].apply(parse_dti)
    if sample_df_clean['debt_to_income_ratio'].isna().sum() == 0:
        sample_df_clean['debt_to_income_ratio'] = sample_df_clean['debt_to_income_ratio'].astype(int)


    sample_df_clean['derived_race'] = sample_df_clean['derived_race'].astype(str).str.strip()
    race_dummies = pd.get_dummies(sample_df_clean['derived_race'], prefix='derived_race', drop_first=True)
    sample_df_clean = pd.concat([sample_df_clean.drop(columns=['derived_race']), race_dummies], axis=1)
    
 
    sample_df_clean['derived_loan_product_type'] = sample_df_clean['derived_loan_product_type'].astype(str).str.strip()
    loan_type_dummies = pd.get_dummies(sample_df_clean['derived_loan_product_type'], prefix='loan_product_type', drop_first=True)
    sample_df_clean = pd.concat([sample_df_clean.drop(columns=['derived_loan_product_type']), loan_type_dummies], axis=1)
    

    sample_df_clean['derived_ethnicity'] = sample_df_clean['derived_ethnicity'].astype(str).str.strip()
    ethnicity_dummies = pd.get_dummies(sample_df_clean['derived_ethnicity'], prefix='derived_ethnicity', drop_first=True)
    sample_df_clean = pd.concat([sample_df_clean.drop(columns=['derived_ethnicity']), ethnicity_dummies], axis=1)


    sample_df_clean['derived_sex'] = sample_df_clean['derived_sex'].astype(str).str.strip()
    sex_dummies = pd.get_dummies(sample_df_clean['derived_sex'], prefix='derived_sex', drop_first=True)
    sample_df_clean = pd.concat([sample_df_clean.drop(columns=['derived_sex']), sex_dummies], axis=1)


    sample_df_clean['derived_dwelling_category'] = sample_df_clean['derived_dwelling_category'].astype(str).str.strip()
    dwelling_dummies = pd.get_dummies(sample_df_clean['derived_dwelling_category'], prefix='dwelling_category', drop_first=True)
    sample_df_clean = pd.concat([sample_df_clean.drop(columns=['derived_dwelling_category']), dwelling_dummies], axis=1)


    sample_df_clean = sample_df_clean.dropna(subset=['conforming_loan_limit'])
    sample_df_clean['conforming_loan_limit'] = sample_df_clean['conforming_loan_limit'].map({'C': 1,'NC': 0})

    if(balanced_y):
        df_1 = sample_df_clean[sample_df_clean['action_taken'] == 1]
        df_0 = sample_df_clean[sample_df_clean['action_taken'] == 0]

        # Downsample majority class to match minority class size
        df_1_balanced = resample(df_1, replace=False, n_samples=len(df_0), random_state=42)

        # Combine and shuffle
        sample_df_clean = pd.concat([df_1_balanced, df_0]).sample(frac=1, random_state=42).reset_index(drop=True)
        
    if(balanced_y_bar):
        df_1 = sample_df_clean[sample_df_clean['applicant_sex'] == 1]
        df_0 = sample_df_clean[sample_df_clean['applicant_sex'] == 0]

        # Downsample majority class to match minority class size
        df_0_balanced = resample(df_0, replace=False, n_samples=len(df_1), random_state=42)

        # Combine and shuffle
        sample_df_clean = pd.concat([df_0_balanced, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)


    Y = sample_df_clean['action_taken'].astype(int)
    Y_sen = sample_df_clean['applicant_sex'].astype(int)
    X =sample_df_clean.drop(columns=["action_taken", "applicant_sex"])
    
    return X, Y, Y_sen





