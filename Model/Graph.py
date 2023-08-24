import matplotlib.pyplot as plt
import seaborn as sns
from Src.pre_processing import big_mart_data
import pandas as pd
'''Numerical features'''

'''covert float64 data of item_weigh in numeric for seaborn processing'''
big_mart_data['Item_Weight'] = pd.to_numeric(big_mart_data['Item_Weight'], errors='coerce')

sns.set()
'''displot(past method) to hisplot(feature method) => (kde=True) for average line of graph'''
big_data = big_mart_data
# Item_Weight distribution
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Weight'], kde=True)
plt.show()

# Item_Visibility distribution
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Visibility'], kde=True)
plt.show()

# Item_MRP distribution
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_MRP'], kde=True)
plt.show()

# Outlet_Sales distribution
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Outlet_Sales'], kde=True)
plt.show()


'''Outlet_let_Establishment_Year column'''
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.show()

'''categorical features'''

# Item_Fat_Content column
plt.figure(figsize=(6, 6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()

# Item_Type column
plt.figure(figsize=(30, 6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()

# # Outlet_Size column
# plt.figure(figsize=(6, 6))
# sns.countplot(x='Outlet_Size', data=big_mart_data)
# plt.title('Item_Type count')
# plt.show()


