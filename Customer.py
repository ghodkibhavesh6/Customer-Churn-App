# take required lib
import pandas as pd
import pickle

# load the data set
df = pd.read_csv("Customer Churn data.csv")

# remove useless columns
del df["CustomerID"]

# handel missing valuses
# number values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Tenure"] = df["Tenure"].fillna(df["Tenure"].mean())
df["Usage Frequency"] = df["Usage Frequency"].fillna(df["Usage Frequency"].mean())
df["Support Calls"] = df["Support Calls"].fillna(df["Support Calls"].mean())
df["Payment Delay"] = df["Payment Delay"].fillna(df["Payment Delay"].mean())
df["Total Spend"] = df["Total Spend"].fillna(df["Total Spend"].mean())
df["Last Interaction"] = df["Last Interaction"].fillna(df["Last Interaction"].mean())

#Alphabetical values
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Subscription Type"] = df["Subscription Type"].fillna(df["Subscription Type"].mode()[0])
df["Contract Length"] = df["Contract Length"].fillna(df["Contract Length"].mode()[0])

# remove the row where churn is missing
df = df.dropna(subset=["Churn"])

# Convert Churn (float to int)
df["Churn"] = df["Churn"].astype(int)

# convert categorical to numbers 
df = pd.get_dummies(df, drop_first=True)

# select feature and label (input and output)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# select the algorithm and train the model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained successfully!")

# save model
pickle.dump(model, open("churn_model.pkl", "wb"))
pickle.dump(X.columns, open("model_columns.pkl", "wb"))

print("Model saved successfully!")
