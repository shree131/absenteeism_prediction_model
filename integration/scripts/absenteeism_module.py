# coding: utf-8

# ================================
# Imports
# ================================
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# ================================
# Custom Scaler
# ================================
class CustomScaler(BaseEstimator, TransformerMixin):
    """
    A custom scaler that applies standard scaling to specific numerical columns.
    """
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# ================================
# Absenteeism Model Class
# ================================
class absenteeism_model:
    """
    A class that loads a saved logistic regression model and scaler, 
    preprocesses new data, and generates absenteeism predictions.
    """
    def __init__(self, model_file, scaler_file):
        # Load model and scaler from disk
        with open('model', 'rb') as model_f, open('scaler', 'rb') as scaler_f:
            self.reg = pickle.load(model_f)
            self.scaler = pickle.load(scaler_f)
            self.data = None

    def load_and_clean_data(self, data_file):
        """
        Load and preprocess new data for prediction.
        """
        df = pd.read_csv(data_file, delimiter=',')
        self.df_with_predictions = df.copy()

        # Rename columns
        column_names = ['id', 'reason_for_absence', 'date_of_absence', 'transportation_expense_dollars', 'distance_to_work_miles', 'age',
            'daily_work_load_average', 'body_mass_index', 'education', 'children', 'pets'
        ]
        df.columns = column_names

        # Drop unneeded column
        df = df.drop(['id'], axis=1)

        # Add placeholder column
        df['absenteeism_time_hours'] = 'NaN'

        # Create dummy variables for 'reason_for_absence'
        reason_columns = pd.get_dummies(df['reason_for_absence'], drop_first=True, dtype=int)
        reason_group_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_group_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_group_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_group_4 = reason_columns.loc[:, 22:28].max(axis=1)

        # Add all reason groups into the main df
        grouped_reasons = pd.DataFrame({
            'reason_group_1': reason_group_1,
            'reason_group_2': reason_group_2,
            'reason_group_3': reason_group_3,
            'reason_group_4': reason_group_4,
        })

        df = df.drop(['reason_for_absence'], axis=1)
        df = pd.concat([df, grouped_reasons], axis=1)

        # Extract date features
        df['date_of_absence'] = pd.to_datetime(df['date_of_absence'], format='%d/%m/%Y')
        df['day_of_week'] = df['date_of_absence'].dt.weekday
        df['month'] = df['date_of_absence'].dt.month
        df.drop(['date_of_absence'], axis=1, inplace=True)

        # Final column order
        column_names_reordered = [
            'reason_group_1', 'reason_group_2', 'reason_group_3', 'reason_group_4',
            'day_of_week', 'month', 'transportation_expense_dollars', 'distance_to_work_miles',
            'age', 'daily_work_load_average', 'body_mass_index', 'education',
            'children', 'pets', 'absenteeism_time_hours'
        ]
        df = df[column_names_reordered]

        # Simplify education categories
        df['education'] = df['education'].map(lambda x: 0 if x == 1 else 1)

        # Fill any NaN values
        df = df.fillna(value=0)

        # Drop unused or insignificant features
        df = df.drop(['absenteeism_time_hours'], axis=1)
        df = df.drop(['day_of_week', 'daily_work_load_average', 'distance_to_work_miles', 'education'], axis=1)

        # Preprocess using saved scaler
        self.preprocessed_data = df.copy()
        self.data = self.scaler.transform(df)

    def predicted_probability(self):
        """
        Returns the probability of excessive absenteeism.
        """
        if self.data is not None:
            return self.reg.predict_proba(self.data)[:, 1]

    def predicted_output_category(self):
        """
        Returns predicted category for excessive absenteeism (0 or 1).
        """
        if self.data is not None:
            return self.reg.predict(self.data)

    def predicted_outputs(self):
        """
        Appends prediction results to the data.
        """
        if self.data is not None:
            self.preprocessed_data['probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data