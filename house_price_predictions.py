import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error





# Load the dataset
dataset = pd.read_excel('HousePricePrediction.xlsx')

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(dataset.head(5))

# Display the shape of the dataset
print("Shape of the dataset:", dataset.shape)


# Data PreProcessing

obj= (dataset.dtypes == 'object') # Identify categorical variables
object_cols = list(obj[obj].index)# Get the list of categorical variables
print("Categorical variables:", len(object_cols)) # Print the number of categorical variables

int_= (dataset.dtypes == 'int') #
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

flt = (dataset.dtypes == 'float')
flt_cols = list(flt[flt].index)
print("Float variables:", len(flt_cols))


# Exploratory Data Analysis

# Visualizing the distribution of numerical features
numerical_dataset = dataset.select_dtypes(include = ['number'])

plt.figure(figsize=(12,6)) # Figure size for the plot
sns.heatmap(numerical_dataset.corr(), cmap='BrBG', fmt=".2f", linewidths=2, annot=True) # Correlation heatmap 
plt.title('Correlation Heatmap of Numerical Features') 
plt.tight_layout() # Adjust layout to prevent overlap
plt.show()


# Visualizing the distribution of categorical features
# Count unique values for each categorical column
unique_counts = {col: dataset[col].nunique() for col in object_cols}

# Convert to a DataFrame for easier plotting
unique_df = pd.DataFrame({
    'Feature': list(unique_counts.keys()),
    'UniqueValues': list(unique_counts.values())
}) #

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(data=unique_df, x='Feature', y='UniqueValues') # Bar plot of unique values
plt.title("Number of Unique Values in Categorical Features")
plt.xticks(rotation=90) # Rotate x-axis labels for better readability
plt.tight_layout() # Adjust layout to prevent overlap
plt.show()


# To findout the actual count of each category we can plot the bargraph of each four features separately.

# Plotting the distribution of categorical features

import math # for calculating number of rows in the plot grid

num_plots = len(object_cols) # Number of categorical features
cols = 4 # Number of columns in the plot grid
rows = math.ceil(num_plots / cols) # Calculate number of rows needed

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4)) # Create subplots
fig.suptitle('Categorical Features Distribution', fontsize=22) # Set the main title of the figure
plt.subplots_adjust(top=0.95, hspace=0.5)# Adjust layout to prevent overlap

# Flatten axes for easy iteration
axes = axes.flatten() # Flatten the axes array for easy iteration

for i, col in enumerate(object_cols): # Iterate over each categorical column
    y = dataset[col].value_counts() # Get the count of each category in the column
    sns.barplot(x=y.index, y=y.values, ax=axes[i]) # Create a bar plot for the category counts
    axes[i].set_title(col) # Set the title of the subplot
    axes[i].tick_params(axis='x', rotation=90) # Rotate x-axis labels for better readability

# Hide any unused axes
for j in range(i + 1, len(axes)): # Iterate over remaining axes
    fig.delaxes(axes[j]) # Delete unused axes

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent overlap
plt.show() # Display the plots


# Data Cleaning

dataset.drop(['Id'], axis = 1, inplace= True) # Drop the 'Id' column as it is not needed for analysis

dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean()) # Fill missing values in 'SalePrice' with the mean value

new_dataset = dataset.dropna() # Drop rows with any missing values

print(new_dataset.isnull().sum() )# Check for any remaining missing values

#One-Hot Encoding for Label Categorical Features

s = (new_dataset.dtypes == 'object') # Identify categorical variables
object_cols = list(s[s].index) # Get the list of categorical variables

print(object_cols) # Print the list of categorical variables
print("Categorical variables after cleaning:", len(object_cols)) # Print the number of categorical variables

# Initialize OneHotEncoder

OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Create an instance of OneHotEncoder  
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols])) # Apply OneHotEncoder to the categorical columns
OH_cols.index = new_dataset.index # Set the index of the new DataFrame to match the original dataset
OH_cols.columns = OH_encoder.get_feature_names_out() # Set the column names of the new DataFrame 
df_final = new_dataset.drop(object_cols, axis=1) # Drop the original categorical columns from the dataset
df_final = pd.concat([df_final, OH_cols], axis=1) # Concatenate the new one-hot encoded columns with the original dataset

# Display the final dataset shape and a preview of the first few rows
print("Original dataset shape:", dataset.shape)
print("After cleaning and encoding shape:", df_final.shape)
print("Final dataset preview:")
print(df_final.head())

#Splitting the dataset into training and testing sets

X = df_final.drop(['SalePrice'], axis=1) # # X is the feature set excluding the target variable 'SalePrice'
Y = df_final['SalePrice'] # Y is the target variable 'SalePrice'

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)# Split the dataset into training and validation sets


# Model Training and Evaluation

model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train) # Fit the model on the training data
Y_pred = model_SVR.predict(X_valid) # Predict the target variable on the validation set

print ("Mean Absolute Error:", mean_absolute_error(Y_valid, Y_pred)) # Calculate and print the Mean Absolute Error
