import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

int_= (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Categorical variables:", len(num_cols))

flt = (dataset.dtypes == 'float')
flt_cols = list(flt[flt].index)
print("Categorical variables:", len(flt_cols))
