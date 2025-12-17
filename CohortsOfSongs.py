import pandas as pd #used for data manipulation
import numpy as np # used for its numerical operations
import matplotlib.pyplot as plt # used for plotting the data 
import seaborn as sns # also used for plotting 
from scipy import stats # used for its statistical functions

# Loading the dataset: 

data = pd.read_csv('C:/Users/valen/OneDrive/Desktop/marketing_data.csv') # loading the CSV file. Change the path to your file location

data.columns = data.columns.str.strip()  # Removes the leading and trailing spaces from the column names

# Converting Income to numeric: 

data['Income'] = data['Income'].replace(r'[\$,]', '', regex=True).astype(float) 

'''Cleans and converts the Income column by removing the dollar sign and its commas so the values can 
be converted from a string to a float type. '''

# Displaying the first few rows of the dataset: 

print(data.head())

#Checking the data types and null values: 

print(data.info())

# Converting the Dt_Customer to datetime format: 

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], errors='coerce')

# Checking for the null values in income: 

print(data['Income'].isnull().sum())

# Imputing the missing Income values based on Education and Marital_Status: 

def impute_income(row, income_mean):
    if pd.isnull(row['Income']):
        return income_mean.get((row['Education'], row['Marital_Status']), np.nan)
    return row['Income']

    '''This defines a function in order to fill in missing 'Income' values usign the mean income for 
    each combination. '''

# Calculating the mean income for each Education and Marital_Status combination: 

income_mean = data.groupby(['Education', 'Marital_Status'])['Income'].mean() # calculates the mean income for each group 

# Imputing missing Income values: 

data['Income'] = data.apply(lambda row: impute_income(row, income_mean), axis=1) # applies the imputation function in order to fill in missing values

# Checking for unique values in Education and Marital_Status: 

print(data['Education'].unique())
print(data['Marital_Status'].unique())

# Cleaning Education and Marital_Status: 

data['Education'] = data['Education'].str.strip().str.title()
data['Marital_Status'] = data['Marital_Status'].str.strip().str.title()

# Creating variables for the total number of children, age, and the total spending: 

data['Total_Children'] = data['Kidhome'] + data['Teenhome']
data['Age'] = 2023 - data['Year_Birth']
data['Total_Spending'] = data[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                               'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Deriving the total purchases from the number of transactions: 

data['Total_Purchases'] = data[['NumDealsPurchases', 'NumWebPurchases', 'NumStorePurchases']].sum(axis=1)

'''This creates a new column for the total number of children at home'''


# Generating box plots and histograms for the distributions: 

def plot_distributions(data, column):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=data[column])
    plt.title(f'Box Plot of {column}')
    plt.subplot(1, 2, 2)
    sns.histplot(data[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.tight_layout()
    plt.show()

# Plotting the distributions for Total_Spending: 

plot_distributions(data, 'Total_Spending')

# Finding the outliers using the Z-score method: 

def remove_outliers(data, column):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < 3]

# Removing those outliers from Total_Spending: 

data = remove_outliers(data, 'Total_Spending')

# Generating a heatmap to illustrate correlations (only showing the numeric columns): 

plt.figure(figsize=(18, 14))    
numeric_data = data.select_dtypes(include=[np.number])
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    square=True,
    annot_kws={"size": 8}  
)
plt.title('Correlation Heatmap', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

'''The reason I made the heatmap so big is because I wanted to fit all the numbers in it so its 
easier to see the relationship between the variables.'''


# Hypothesis Testing: 


# Hypothesis 1: Older individuals may prefer in-store shopping: 
# Visulizing store purchases by age group: 

def test_hypothesis_1(data):
    data = data.copy()
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-39', '40-49', '50-59', '60+'])
    older_customers = data[data['Age'] > 50]
    younger_customers = data[data['Age'] <= 50]
    older_store_purchases = older_customers['NumStorePurchases'].mean()
    younger_store_purchases = younger_customers['NumStorePurchases'].mean()
    print(f'Older Customers Store Purchases: {older_store_purchases}')
    print(f'Younger Customers Store Purchases: {younger_store_purchases}')
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='AgeGroup', y='NumStorePurchases', data=data)
    plt.title('Store Purchases by Age Group')
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.show()


# Hypothesis 2: Customers with children prefer online shopping: 
# Visualizing online purchases by the number of children at home: 

def test_hypothesis_2(data):
    customers_with_children = data[data['Total_Children'] > 0]
    customers_without_children = data[data['Total_Children'] == 0]
    online_purchases_with_children = customers_with_children['NumWebPurchases'].mean()
    online_purchases_without_children = customers_without_children['NumWebPurchases'].mean()
    print(f'Online Purchases with Children: {online_purchases_with_children}')
    print(f'Online Purchases without Children: {online_purchases_without_children}')
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Total_Children', y='NumWebPurchases', data=data)
    plt.title('Web Purchases by Number of Children')
    plt.tight_layout()
    plt.show()


# Hypothesis 3: 
# Comparing the average store and web purchases to see if web sales might be cannibalizing store sales: 

def test_hypothesis_3(data):
    store_purchases = data['NumStorePurchases'].mean()
    web_purchases = data['NumWebPurchases'].mean()
    print(f'Store Purchases: {store_purchases}')
    print(f'Web Purchases: {web_purchases}')
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Store', 'Web'], y=[store_purchases, web_purchases])
    plt.title('Comparison of Store and Web Purchases')
    plt.ylabel('Average Purchases')
    plt.tight_layout()
    plt.show()


# Hypothesis 4: united states vs. rest of the world in total purchase: 

def test_hypothesis_4(data):
    us_customers = data[data['Country'].str.upper().isin(['US', 'USA', 'UNITED STATES'])]
    non_us_customers = data[~data['Country'].str.upper().isin(['US', 'USA', 'UNITED STATES'])]
    us_total_purchases = us_customers['Total_Purchases'].sum()
    non_us_total_purchases = non_us_customers['Total_Purchases'].sum()
    print(f'Total Purchases in the US: {us_total_purchases}')
    print(f'Total Purchases outside the US: {non_us_total_purchases}')
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['United States', 'Rest of World'], y=[us_total_purchases, non_us_total_purchases])
    plt.title('Total Purchases: US vs Rest of World')
    plt.ylabel('Total Purchases')
    plt.tight_layout()
    plt.show()


# Running the hypothesis tests and displaying them:

test_hypothesis_1(data)
test_hypothesis_2(data)
test_hypothesis_3(data)
test_hypothesis_4(data)

# Visualizing the top-performing products and finding whihc product generate the most revenue: 

def top_performing_products(data):
    product_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                       'MntSweetProducts', 'MntGoldProds']
    total_revenue = data[product_columns].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=total_revenue.index, y=total_revenue.values)
    plt.title('Top-Performing Products by Revenue')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

top_performing_products(data)

# Finding the correlation between age and acceptance rate of the last campaign: 

def age_acceptance_correlation(data):
    plt.figure(figsize=(10, 6)) # Visualizing the data 
    sns.scatterplot(x='Age', y='AcceptedCmp3', data=data, alpha=0.6)
    plt.title('Correlation between Age and Acceptance Rate of Last Campaign')
    plt.xlabel('Age')
    plt.ylabel('Acceptance Rate of Last Campaign')
    plt.tight_layout()
    plt.show()

age_acceptance_correlation(data)


# Finding the country with the highest number of customers who accepted the last campaign: 

def country_highest_acceptance(data):
    acceptance_counts = data[data['AcceptedCmp3'] == 1]['Country'].value_counts()
    top_country = acceptance_counts.idxmax()
    top_count = acceptance_counts.max()
    print(f'Country with highest acceptance: {top_country} ({top_count} customers)')
    plt.figure(figsize=(10, 6))
    sns.barplot(x=acceptance_counts.index, y=acceptance_counts.values)
    plt.title('Number of Customers Accepted Last Campaign by Country')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

country_highest_acceptance(data)


# Finding the pattern in the number of children at home and total: 

def children_expenditure_pattern(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Total_Children', y='Total_Spending', data=data, alpha=0.6)
    plt.title('Pattern of Number of Children at Home and Total Expenditure')
    plt.xlabel('Total Number of Children')
    plt.ylabel('Total Expenditure')
    plt.tight_layout()
    plt.show()

children_expenditure_pattern(data)

# Looking at the education background of customers who had complaints in the last two years: 

def complaints_education(data):
    complaints = data[data['Complain'] == 1]
    education_counts = complaints['Education'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=education_counts.index, y=education_counts.values)
    plt.title('Educational Background of Customers Who Complained in Last Two Years')
    plt.ylabel('Number of Complaints')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

complaints_education(data)


# Applying ordinal and one hot encoding after all the functions that require the original columns: 

data = pd.get_dummies(data, columns=['Education', 'Marital_Status'], drop_first=True)

# Saving the cleaned and processed data to a new CSV file: 

data.to_csv('cleaned_marketing_data.csv', index=False)
print("Data has been cleaned and saved to 'cleaned_marketing_data.csv'.")