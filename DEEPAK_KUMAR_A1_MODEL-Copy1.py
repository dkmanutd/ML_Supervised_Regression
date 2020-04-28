# timeit

# Student Name : Deepak Kumar
# Cohort       : Haight

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

import pandas as pd                                  # data science essentials
from sklearn.model_selection import train_test_split # train/test split
import sklearn.linear_model                          # linear regression models
import numpy as np                                   # calculation essentials

################################################################################
# Load Data
################################################################################

file = 'Apprentice Chef Dataset.xlsx'
original_df = pd.read_excel(file)                    # reading in the dataset

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

## Categorization of data 

"""
# CONTINUOUS OR INTERVAL

REVENUE
AVG_TIME_PER_SITE_VISIT
AVG_PREP_VID_TIME
AVG_CLICKS_PER_VISIT
FOLLOWED_RECOMMENDATIONS_PCT


# BINARY

CROSS_SELL_SUCCESS
MOBILE_NUMBER
PACKAGE_LOCKER
REFRIGERATED_LOCKER
TASTES_AND_PREFERENCES


# COUNT

TOTAL_MEALS_ORDERED
UNIQUE_MEALS_PURCH
CONTACTS_W_CUSTOMER_SERVICE
PRODUCT_CATEGORIES_VIEWED
CANCELLATIONS_BEFORE_NOON
CANCELLATIONS_AFTER_NOON
MOBILE_LOGINS
PC_LOGINS
WEEKLY_PLAN
EARLY_DELIVERIES
LATE_DELIVERIES
LARGEST_ORDER_SIZE
MASTER_CLASSES_ATTENDED
TOTAL_PHOTOS_VIEWED


# DISCRETE
LARGEST_ORDER_SIZE
MEDIAN_MEAL_RATING
"""

# Setting outlier thresholds:

REVENUE_hi = 2250
AVG_TIME_PER_SITE_VISIT_hi = 250
AVG_PREP_VID_TIME_lo = 50
AVG_PREP_VID_TIME_hi = 275
AVG_CLICKS_PER_VISIT_lo = 8
AVG_CLICKS_PER_VISIT_hi = 20
FOLLOWED_RECOMMENDATIONS_PCT_hi = 50
TOTAL_MEALS_ORDERED_hi = 150
UNIQUE_MEALS_PURCH_hi = 9
CONTACTS_W_CUSTOMER_SERVICE_hi = 12
PRODUCT_CATEGORIES_VIEWED_lo = 1
PRODUCT_CATEGORIES_VIEWED_hi = 10
CANCELLATIONS_BEFORE_NOON_hi = 7
MOBILE_LOGINS_lo = 4.5
MOBILE_LOGINS_hi = 6.5
PC_LOGINS_lo = 1
PC_LOGINS_hi = 2
WEEKLY_PLAN_hi = 20
EARLY_DELIVERIES_hi = 5
LATE_DELIVERIES_hi = 10
LARGEST_ORDER_SIZE_lo = 1
LARGEST_ORDER_SIZE_hi = 7
MASTER_CLASSES_ATTENDED_hi = 2
TOTAL_PHOTOS_VIEWED_hi = 600
MEDIAN_MEAL_RATING_hi = 4

## Feature Engineering (outlier thresholds)

# Developing features (columns) for outliers:

#Revenue
original_df['out_REVENUE'] = 0
condition_hi = original_df.loc[0:,'out_REVENUE'][original_df['REVENUE'] > REVENUE_hi]

original_df['out_REVENUE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#AVG_TIME_PER_SITE_VISIT
original_df['out_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_TIME_PER_SITE_VISIT']\
                              [original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]

original_df['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#AVG_PREP_VID_TIME
original_df['out_AVG_PREP_VID_TIME'] = 0
condition_hi = original_df.loc[0:,'out_AVG_PREP_VID_TIME']\
                              [original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]
condition_lo = original_df.loc[0:,'out_AVG_PREP_VID_TIME']\
                              [original_df['AVG_PREP_VID_TIME'] < AVG_PREP_VID_TIME_lo]

original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#AVG_CLICKS_PER_VISIT
original_df['out_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT']\
                              [original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]
condition_lo = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT']\
                              [original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]

original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#FOLLOWED_RECOMMENDATIONS_PCT
original_df['out_FOLLOWED_RECOMMENDATIONS_PCT'] = 0
condition_hi = original_df.loc[0:,'out_FOLLOWED_RECOMMENDATIONS_PCT']\
                              [original_df['FOLLOWED_RECOMMENDATIONS_PCT'] > FOLLOWED_RECOMMENDATIONS_PCT_hi]

original_df['out_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#TOTAL_MEALS_ORDERED
original_df['out_TOTAL_MEALS_ORDERED'] = 0
condition_hi = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED']\
                              [original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]

original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#UNIQUE_MEALS_PURCH
original_df['out_UNIQUE_MEALS_PURCH'] = 0
condition_hi = original_df.loc[0:,'out_UNIQUE_MEALS_PURCH']\
                              [original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]

original_df['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#CONTACTS_W_CUSTOMER_SERVICE
original_df['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = original_df.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE']\
                              [original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]

original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#PRODUCT_CATEGORIES_VIEWED
original_df['out_PRODUCT_CATEGORIES_VIEWED'] = 0
condition_hi = original_df.loc[0:,'out_PRODUCT_CATEGORIES_VIEWED']\
                              [original_df['PRODUCT_CATEGORIES_VIEWED'] > PRODUCT_CATEGORIES_VIEWED_hi]
condition_lo = original_df.loc[0:,'out_PRODUCT_CATEGORIES_VIEWED']\
                              [original_df['PRODUCT_CATEGORIES_VIEWED'] < PRODUCT_CATEGORIES_VIEWED_lo]

original_df['out_PRODUCT_CATEGORIES_VIEWED'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_PRODUCT_CATEGORIES_VIEWED'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#CANCELLATIONS_BEFORE_NOON
original_df['out_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = original_df.loc[0:,'out_CANCELLATIONS_BEFORE_NOON']\
                              [original_df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_hi]

original_df['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#MOBILE_LOGINS
original_df['out_MOBILE_LOGINS'] = 0
condition_hi = original_df.loc[0:,'out_MOBILE_LOGINS']\
                              [original_df['MOBILE_LOGINS'] > MOBILE_LOGINS_hi]
condition_lo = original_df.loc[0:,'out_MOBILE_LOGINS']\
                              [original_df['MOBILE_LOGINS'] < MOBILE_LOGINS_lo]

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#PC_LOGINS
original_df['out_PC_LOGINS'] = 0
condition_hi = original_df.loc[0:,'out_PC_LOGINS']\
                              [original_df['PC_LOGINS'] > PC_LOGINS_hi]
condition_lo = original_df.loc[0:,'out_PC_LOGINS']\
                              [original_df['PC_LOGINS'] < PC_LOGINS_lo]

original_df['out_PC_LOGINS'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_PC_LOGINS'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#WEEKLY_PLAN
original_df['out_WEEKLY_PLAN'] = 0
condition_hi = original_df.loc[0:,'out_WEEKLY_PLAN']\
                              [original_df['WEEKLY_PLAN'] > WEEKLY_PLAN_hi]

original_df['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#EARLY_DELIVERIES
original_df['out_EARLY_DELIVERIES'] = 0
condition_hi = original_df.loc[0:,'out_EARLY_DELIVERIES']\
                              [original_df['EARLY_DELIVERIES'] > EARLY_DELIVERIES_hi]

original_df['out_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#LATE_DELIVERIES
original_df['out_LATE_DELIVERIES'] = 0
condition_hi = original_df.loc[0:,'out_LATE_DELIVERIES']\
                              [original_df['LATE_DELIVERIES'] > LATE_DELIVERIES_hi]

original_df['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#LARGEST_ORDER_SIZE
original_df['out_LARGEST_ORDER_SIZE'] = 0
condition_hi = original_df.loc[0:,'out_LARGEST_ORDER_SIZE']\
                              [original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]
condition_lo = original_df.loc[0:,'out_LARGEST_ORDER_SIZE']\
                              [original_df['PRODUCT_CATEGORIES_VIEWED'] < LARGEST_ORDER_SIZE_lo]

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#MASTER_CLASSES_ATTENDED
original_df['out_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = original_df.loc[0:,'out_MASTER_CLASSES_ATTENDED']\
                              [original_df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_hi]

original_df['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#TOTAL_PHOTOS_VIEWED
original_df['out_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = original_df.loc[0:,'out_TOTAL_PHOTOS_VIEWED']\
                              [original_df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_hi]

original_df['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#MEDIAN_MEAL_RATING
original_df['out_MEDIAN_MEAL_RATING'] = 0
condition_hi = original_df.loc[0:,'out_MEDIAN_MEAL_RATING']\
                              [original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_hi]

original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)


import numpy as np

# Creating new columns
original_df['log_REVENUE'] = np.log(original_df['REVENUE'])


# Setting trend-based thresholds 
AVG_TIME_PER_SITE_VISIT_change_hi = 300
AVG_PREP_VID_TIME_change_hi = 300
CONTACTS_W_CUSTOMER_SERVICE_change_hi = 10
TOTAL_MEALS_ORDERED_change_hi = 25 


# Feature Engineering:

original_df['change_AVG_TIME_PER_SITE_VISIT'] = 0
condition = original_df.loc[0:,'change_AVG_TIME_PER_SITE_VISIT']\
            [original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_change_hi]

original_df['change_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

original_df['change_AVG_PREP_VID_TIME'] = 0
condition = original_df.loc[0:,'change_AVG_PREP_VID_TIME']\
            [original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_change_hi]

original_df['change_AVG_PREP_VID_TIME'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

original_df['change_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition = original_df.loc[0:,'change_CONTACTS_W_CUSTOMER_SERVICE']\
            [original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_change_hi]

original_df['change_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

original_df['change_TOTAL_MEALS_ORDERED'] = 0
condition = original_df.loc[0:,'change_TOTAL_MEALS_ORDERED']\
            [original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_change_hi]

original_df['change_TOTAL_MEALS_ORDERED'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


# Setting change-at thresholds
TOTAL_PHOTOS_VIEWED_change_at = 0 # zero inflated
MEDIAN_MEAL_RATING_change_at_3 = 3
MEDIAN_MEAL_RATING_change_at_4 = 4

# Feature Engineering:

original_df['change_TOTAL_PHOTOS_VIEWED'] = 0
condition = original_df.loc[0:,'change_TOTAL_PHOTOS_VIEWED']\
            [original_df['TOTAL_PHOTOS_VIEWED'] == TOTAL_PHOTOS_VIEWED_change_at]

original_df['change_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

original_df['change_MEDIAN_MEAL_RATING_3'] = 0
condition = original_df.loc[0:,'change_MEDIAN_MEAL_RATING_3']\
            [original_df['MEDIAN_MEAL_RATING'] == MEDIAN_MEAL_RATING_change_at_3]

original_df['change_MEDIAN_MEAL_RATING_3'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

original_df['change_MEDIAN_MEAL_RATING_4'] = 0
condition = original_df.loc[0:,'change_MEDIAN_MEAL_RATING_4']\
            [original_df['MEDIAN_MEAL_RATING'] == MEDIAN_MEAL_RATING_change_at_4]

original_df['change_MEDIAN_MEAL_RATING_4'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)



# making a copy of Apprentice Chef dataset
original_df_explanatory = original_df.copy()

# dropping REVENUE and any text features from the explanatory variable set
original_df_explanatory = original_df.drop(['REVENUE', 'NAME', 'EMAIL',
                                           'FIRST_NAME', 'FAMILY_NAME', 
                                           'log_REVENUE', 'out_REVENUE'], 
                                            axis = 1)


# placeholder list for the e-mail domains
placeholder_lst = []

# looping over each email address
for index, col in original_df.iterrows():
    
    # splitting email domain at '@'
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)     

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)


# renaming columns
email_df.columns = ['concatenate' , 'email_domain']

# concatenating email_domain with Apprentice Chef DataFrame
original_df = pd.concat([original_df, email_df.loc[: ,'email_domain']],
                   axis = 1) # because we are concatenating by column


# creating email domains
professional_domains = ['@mmm.com', '@amex.com', '@apple.com', '@boeing.com',
                       '@caterpillar.com', '@chevron.com', '@cisco.com',
                       '@cocacola.com', '@disney.com','@dupont.com',
                       '@exxon.com', '@ge.org', '@goldmansacs.com',
                       '@homedepot.com', '@ibm.com', '@intel.com', '@jnj.com',
                       '@jpmorgan.com', '@mcdonalds.com', '@merck.com',
                       '@microsoft.com', '@nike.com', '@pfizer.com', '@pg.com',
                       '@travelers.com', '@unitedtech.com', '@unitedhealth.com',
                       '@verizon.com', '@visa.com', '@walmart.com']
personal_domains  = ['@gmail.com', '@yahoo.com', '@protonmail.com']
junk_domains = ['@me.com', '@aol.com', '@hotmail.com', '@live.com', '@msn.com',
                '@passport.com']

# placeholder list
placeholder_lst = []

# looping to group observations by domain type
for domain in original_df['email_domain']:
        if '@' + domain in professional_domains:
            placeholder_lst.append('professional')
        elif '@' + domain in personal_domains:
            placeholder_lst.append('personal')
        elif '@' + domain in junk_domains:
            placeholder_lst.append('junk')
        else:
            print('Unknown')


# concatenating with original DataFrame
original_df['domain_group'] = pd.Series(placeholder_lst)


# checking results
original_df['domain_group'].value_counts()


# One Hot encoding the categorical variable 'domain_group'

one_hot_domain_group = pd.get_dummies(original_df['domain_group'])

# dropping categorical variables after they've been encoded
original_df = original_df.drop('domain_group', axis = 1)

# joining codings together
original_df = original_df.join([one_hot_domain_group])


################################################################################
# Train/Test Split
################################################################################

# preparing explanatory variable data
original_df_data = original_df.drop(['REVENUE', 'out_REVENUE', 'log_REVENUE',
                                     'NAME', 'EMAIL', 'FIRST_NAME', 'FAMILY_NAME', 
                                     'email_domain'],
                                     axis = 1)


# preparing response variable data, and using logarithmic values of Revenue
original_df_target = original_df.loc[:, 'log_REVENUE']


# preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
            original_df_data,
            original_df_target,
            test_size = 0.25,
            random_state = 222)

original_df_data = pd.concat([X_train, y_train], axis = 1)


################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# INSTANTIATING a model object for Linear Regression
lr = sklearn.linear_model.LinearRegression()

# FITTING to the training data
lr_fit = lr.fit(X_train, y_train)

# PREDICTING on new data
lr_pred = lr_fit.predict(X_test)


################################################################################
# Final Model Score (score)
################################################################################

# SCORING the results
test_score = lr.score(X_test, y_test).round(4)