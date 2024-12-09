# %% [markdown]
# # Import Libraries

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# # SET UP

# %%
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# %% [markdown]
# # Load Datasets

# %%
df = pd.read_csv('/Users/Jai/Documents/Git_remote/DataScience/Capstone/data_sources/dataset1/US_healthcare_data-2122020.csv')

# %%
df.head()

# %%
df.info()

# %%
df["readmitted"].value_counts()

# %%
df.describe()

# %%
%matplotlib inline
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()

# %%
# to make this notebook's output identical at every run
np.random.seed(42)

# %% [markdown]
# # EDA

# %% [markdown]
# # Plotting the distribution of the target variable

# %%

sns.countplot(x='readmitted', data=df)
plt.title('Distribution of Readmitted (Target Variable)')
plt.show()

# %% [markdown]
# # Checking Missing Values

# %%

print("\nMissing Values Per Column:")
print(df.isnull().sum())  # Checking for missing values


# %% [markdown]
# # Visualize missing values

# %%

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# %% [markdown]
# # Categorical Variable Exploration

# %%

print("\nDistribution of Categorical Variables:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col} Value Counts:")
    print(df[col].value_counts())

# %% [markdown]
# # Bar plots for categorical variables

# %%

for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=90)
    plt.show()

# %% [markdown]
# # Numerical Variable Exploration

# %%

print("\nSummary of Numerical Variables:")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# %% [markdown]
# # Correlation Analysis

# %%

print("\nCorrelation Matrix:")
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# # Analyze relationships with the target variable

# %%

# Plot correlation between numerical variables and the target variable
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='readmitted', y=col, data=df)
    plt.title(f'{col} vs. Readmitted')
    plt.show()


