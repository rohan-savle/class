import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder ,StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report


df = pd.read_csv('real estate.csv')

df.head()
print(df)


print("shape:",df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicates:",df.duplicated().sum())

df = df.drop_duplicates()

X = df.drop("Buys Laptop", axis=1)
y = df["Buys Laptop"]

numeric_features = ["Age", "Credit Score"]
categorical_features = ["Income", "Student", "Location"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))



numeric_pipeline =Pipeline([
    ("imputer",SimpleImputer(strategy="medin")),
    ("scaler",StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown='ignore'))
])
preprocessor =ColumnTransformer([
    ("num",numeric_pipeline, 'numeric_features'),
    ("cat", categorical_pipeline, 'categorical_features')
])

model = Pipeline([
    ("preprocessor",preprocessor),
    ("svm",SVC(kernel="rbf", C=1.0,gamma="scale"))
])
model.fit(X_train,y_train)