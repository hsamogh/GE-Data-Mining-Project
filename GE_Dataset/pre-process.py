#importing pandas as pd
import numpy.core.umath
import pandas as pd
from matplotlib import pyplot as pp

# reading the training data into a dataframe
train_data = pd.read_csv('train_data.csv')

#######################################################################################################
# cleaning the gender column . we replace 'na' values with a gender that occurs maximum number of times
#######################################################################################################

# getting counts of number of males
male_count = len(train_data[train_data['gender'] == 'M'])

# getting counts of number of females
female_count = len(train_data[train_data['gender'] == 'F'])

# variable gender_default stores default gender to be applied in case of missing gender value
gender_default = ''


# assigning default gender based on counts
if male_count > female_count:
    gender_default = 'M'
else:
    gender_default = 'F'

# adding the default gender to the rows that have 'na' in gender
train_data.loc[pd.isnull(train_data['gender']),'gender' ] = gender_default



#######################################################################################################
# cleaning values of marital_status
#######################################################################################################

# getting counts of 'yes' for marital_status
marital_status_yes = len(train_data[train_data['marital_status'] == 'Yes'])

# getting counts of 'no' for marital_status
marital_status_no = len(train_data[train_data['marital_status'] == 'No'])

# setting the deafult marital status based on counts
if marital_status_yes > marital_status_no:
    default_marital_status = 'Yes'
else:
    default_marital_status = 'No'

# adding default values to missing marital_status
train_data.loc[pd.isnull(train_data['marital_status']),'marital_status'] = default_marital_status


########################################################################################################
# Cleaning values of dependents
########################################################################################################

# For dependents column , we are changing the value of 3+ to 3
train_data.loc[train_data['dependents'] == '3+','dependents'] = '3'

# The dependent column is of type string. converting it to integer and replacing 'na'
train_data.dependents = pd.to_numeric(train_data['dependents'], errors='coerce')

# the default value for dependent is considered to be median
median_dependent = train_data['dependents'].median()

# replacing 'na' values in dependents with median
train_data.loc[pd.isnull(train_data['dependents']),'dependents'] = median_dependent


########################################################################################################
# Cleaning values of qualification
########################################################################################################

# getting counts of graduates in qualification
graduate_count = len(train_data[train_data['qualification'] == 'Graduate'])

# getting counts of 'not graduates' in qualification
not_graduate_count = len(train_data[train_data['qualification'] == 'Not Graduate'])

# setting the deafult qualification based on counts
if graduate_count > not_graduate_count:
    qualification_default = 'Graduate'
else:
    qualification_default = 'Not Graduate'

# adding default values to missing marital_status
train_data.loc[pd.isnull(train_data['qualification']),'qualification'] = qualification_default

########################################################################################################
# Cleaning values of is_self_employed
########################################################################################################

# getting counts of 'yes' for marital_status
self_employed_yes = len(train_data[train_data['is_self_employed'] == 'Yes'])

# getting counts of 'no' for marital_status
self_employed_no = len(train_data[train_data['is_self_employed'] == 'No'])

# setting the deafult marital status based on counts
if self_employed_yes > self_employed_no:
    default_se_status = 'Yes'
else:
    default_se_status = 'No'

# adding default values to missing marital_status
train_data.loc[pd.isnull(train_data['is_self_employed']),'is_self_employed'] = default_se_status

########################################################################################################
# Cleaning values of applicant_income
########################################################################################################


#since applicant_income does not have any outliers , we can safetly consider mean as default value
default_applicant_income = train_data['applicant_income'].mean()

#replacing applicant_income with default applicant_income
train_data.loc[pd.isnull(train_data['applicant_income']), 'applicant_income'] = default_applicant_income

########################################################################################################
# Cleaning values of co_applicant_income
########################################################################################################


# selecting value at 0.99 quantile to eliminate outliers
q = train_data['co_applicant_income'].quantile(0.99)
default_co_applicant_income = train_data[train_data['co_applicant_income']<q]['co_applicant_income'].mean()

#replacing applicant_income with default co_applicant_income
train_data.loc[pd.isnull(train_data['co_applicant_income']), 'co_applicant_income'] = default_applicant_income

########################################################################################################
# Cleaning values of loan_amount
########################################################################################################

# selecting loan amount at 0.99 percent quantile. used to eliminate outliers
q = train_data['loan_amount'].quantile(0.99)
# computing default value
default_loan_amount = train_data[train_data['loan_amount']<q]['loan_amount'].mean()
# replacing nulls with default amount
train_data.loc[pd.isnull(train_data['loan_amount']), 'loan_amount'] = default_loan_amount

########################################################################################################
# Cleaning values of loan_amount_term
########################################################################################################

# selecting loan amount at 0.99 percent quantile. used to eliminate outliers
q = train_data['loan_amount_term'].quantile(0.99)
# computing default value
default_loan_amount_term = train_data[train_data['loan_amount_term']<q]['loan_amount_term'].mean()
# replacing nulls with default loan_amount_term
train_data.loc[pd.isnull(train_data['loan_amount_term']), 'loan_amount_term'] = default_loan_amount_term

########################################################################################################
# Cleaning values of credit history
########################################################################################################

# counting number of 1's for credit_history
one_count = len(train_data[train_data['credit_history'] == 1])

# counting number of 0's for credit_history
zero_count = len(train_data[train_data['credit_history'] == 0])

# setting default value
if zero_count > one_count :
    default_credit_history = 0
else:
    default_credit_history = 1

# assigning default values to missing records
train_data.loc[pd.isnull(train_data['credit_history']),'credit_history'] = default_credit_history


########################################################################################################
# Cleaning values of property area
########################################################################################################


# assigning default values to missing records . use knn to assign values later
train_data.loc[pd.isnull(train_data['property_area']),'property_area'] = 'unknown'

########################################################################################################
# Cleaning  data ends
########################################################################################################


# creating a data frame for predictions
train_predict = pd.read_csv('train_prediction.csv')

# joining training prediction data frame with training dataframe to get label

total_dataset = train_data.join(train_predict.set_index(['loan_id']),on=['loan_id'],how='right')

total_dataset.to_csv('processed_data.csv',index=False)
