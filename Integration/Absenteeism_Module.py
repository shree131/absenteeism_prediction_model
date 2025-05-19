
# coding: utf-8

# In[1]:


# import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# the custom scaler class 
class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
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
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# create the special class that we are going to use from here on to predict new data
class absenteeism_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
            
            # import the data
            df = pd.read_csv(data_file,delimiter=',')
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            # drop the 'ID' column
            df = df.drop(['id'], axis = 1)
            # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
            df['absenteeism_time_hours'] = 'NaN'

            # create a separate dataframe, containing dummy values for ALL avaiable reasons
            reason_columns = pd.get_dummies(df['reason_for_absence'], drop_first = True)
            
            # split reason_columns into 4 types
            reason_type_1 = reason_columns.loc[:,1:14].max(axis=1)
            reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
            reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
            reason_type_4 = reason_columns.loc[:,22:].max(axis=1)
            
            # to avoid multicollinearity, drop the 'Reason for Absence' column from df
            df = df.drop(['reason_for_absence'], axis = 1)
            
            # concatenate df and the 4 types of reason for absence
            df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)
            
            # assign names to the 4 reason type columns
            # note: there is a more universal version of this code, however the following will best suit our current purposes             
            column_names = ['date', 'transportaion_expense_dollars', 'distance_to_work_miles', 'age',
                           'daily_work_load_average', 'body_mass_index', 'education',
                            'children', 'pets', 'absenteeism_time_hours', 'reason_group_1',
                            'reason_group_2', 'reason_group_3', 'reason_group_4']
            df.columns = column_names

            # re-order the columns in df
            column_names_reordered = ['reason_group_1', 'reason_group_2','reason_group_3', 'reason_group_4', 'date', 
                                     'transportaion_expense_dollars', 'distance_to_work_miles', 'age', 'daily_work_load_average',
                                     'body_mass_index', 'education', 'children', 'pets', 'absenteeism_time_hours']
            df = df[column_names_reordered]
      
            # convert the 'Date' column into datetime
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

            # create a list with month values retrieved from the 'Date' column
            list_months = []
            for i in range(df.shape[0]):
                list_months.append(df['date'][i].month)

            # insert the values in a new column in df, called 'Month Value'
            df['month'] = list_months

            # create a new feature called 'Day of the Week'
            df['day_of_week'] = df['date'].apply(lambda x: x.weekday())


            # drop the 'Date' column from df
            df = df.drop(['date'], axis = 1)

            # re-order the columns in df
            column_names_upd = ['reason_group_1', 'reason_group_2', 'reason_group_3', 'reason_group_4', 
                                'day_of_week', 'month', 'transportaion_expense_dollars', 'distance_to_work_miles', 
                                'age', 'daily_work_load_average', 'body_mass_index', 'education', 'children', 'pets', 'absenteeism_time_hours']
            df = df[column_names_upd]


            # map 'Education' variables; the result is a dummy
            df['education'] = df['education'].map({1:0, 2:1, 3:1, 4:1})

            # replace the NaN values
            df = df.fillna(value=0)

            # drop the original absenteeism time
            df = df.drop(['absenteeism_time_hours'],axis=1)
            
            # drop the variables we decide we don't need
            df = df.drop(['day_of_week','daily_work_load_average','distance_to_work_miles'],axis=1)
            
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
        
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data

