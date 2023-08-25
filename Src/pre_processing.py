from sklearn.preprocessing import LabelEncoder
from Src.config import big_mart_data

'''show first 5 rows of the dataframe'''
print(big_mart_data.head())

'''number of data points and number of features'''
print(big_mart_data.shape)

'''getting some info about the dataset'''
big_mart_data.info()

'''
#### categorical features :
- item_identifier
- item_fat_content
- Outlet_Identifier
- Item_Type 
- Outlet_Size
- Outlet_Type 
- Outlet_Location_Type 
'''
'''checking for missing values'''
print(big_mart_data.isnull().sum())

'''
# handling missing values
{Outlet_Size                  2410
Item_Weight                  1463}
'''
'''
# Mean --> average values
# Mode --> most repeated value
'''

# mean value of "Item_Weight" column
print(big_mart_data['Item_Weight'].mean())

# filling the missing values in "Item_Weight" column with "Mean" value
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean, inplace=True)

# check the missing data if correct
print(big_mart_data.isnull().sum())

'''replacing the missing values in "Outlet_Size" with mode'''
mode_of_otlet_size = big_mart_data.pivot_table(values='Outlet_Size',
                                               columns='Outlet_Type',
                                               aggfunc=(lambda x: x.mode(0)))
print(mode_of_otlet_size)

missing_values = big_mart_data['Outlet_Size'].isnull()
'''true = value is absent and false = is present'''
print(missing_values)

'''locate the missing values'''
big_mart_data.loc[missing_values, 'Outlet_Size'] = \
    big_mart_data.loc[missing_values, 'Outlet_Type'].apply(lambda x: mode_of_otlet_size)

'''checking for missing values'''
print(big_mart_data.isnull().sum())

'''
#### DATA ANALYSIS
'''
# statistical measures about the data

print(big_mart_data.describe())

'''replace title of item fat content'''
print(big_mart_data['Item_Fat_Content'].value_counts())

big_mart_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}},
                      inplace=True)

print(big_mart_data['Item_Fat_Content'].value_counts())

'''Label Encoding'''
encoder = LabelEncoder()


big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])

big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])

big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])

big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])

big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'].astype(str))

big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])

big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])


print(big_mart_data.head())

X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']

print(X)
print(Y)

