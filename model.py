
### Import Libraries

import pandas as pd
import numpy as np
import warnings
import pickle
import pickle

warnings.filterwarnings('ignore')

### Import Datset
df = pd.read_csv("booking.csv")
# Changing datatype to Datetime
df['date of reservation'] = pd.to_datetime(df['date of reservation'], errors='coerce')

# Extract day, month, and year from the date
df['year'] = df['date of reservation'].dt.year
df['month'] = df['date of reservation'].dt.month
df['day'] = df['date of reservation'].dt.day


# Dropping Booking_ID Column and date of reservation Column
df.drop(columns=['Booking_ID','date of reservation'] , inplace=True)
df.dropna(inplace=True)


### Encoding columns
encoding_columns = df.select_dtypes(include=["object"]).columns

# Perform one-hot encoding using pandas.get_dummies
df_encoded = pd.get_dummies(df, columns=encoding_columns)


### Splitting Data and Data Preprocessing
# Identify features and target
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

feature_cols = ['lead time', 'average price', 'special requests','day' ,'month', 'market segment type_Online', 'market segment type_Offline']
X_selected = df_encoded[feature_cols]  # Selected Features
y = df['booking status']  # Target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

## training classififer
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier()
clf_RF.fit(X_train, y_train)
predictions = clf_RF.predict(X_test)

##
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


print("Accuracy Score : \n\n" , accuracy_score(y_test, predictions))

print("Confusion Matrix : \n\n" , confusion_matrix(predictions,y_test))

print("Classification Report : \n\n" , classification_report(predictions,y_test),"\n")

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(clf_RF, file)

# Save the scaler using pickle
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler using pickle
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)