import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#Load dataset
data = pd.read_csv("student_data.csv")

#Input features
X = data[["study_hours", "attendance", "previous_marks"]]

#Output
y = data["result"]

#Convert Pass/Fail into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

#Split datadset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create and train model
model = GaussianNB()
model.fit(X_train, y_train)

#Test model
y_pred = model.predict(X_test)

#Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100)

#Save model and encoder
with open("model.pkl", "wb") as file:
    pickle.dump((model, encoder), file)

print("Model saved successfully!")    