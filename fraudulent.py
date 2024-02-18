import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv(r'C:\Users\gopal\Downloads\codeway\archive1\fraudTrain.csv')


le = LabelEncoder()
df.drop(df.columns[0], axis=1, inplace=True)
features = df.columns
for i in features:
  if df[i].dtype=='object':
    df[i] = le.fit_transform(df[i])
    df.fillna(method='ffill',inplace=True)

X = df.drop('is_fraud', axis=1)  
y = df['is_fraud']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))