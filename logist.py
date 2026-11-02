import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import(GridSearchCV,StratifiedKFold)


from sklearn.metrics import (

    confusion_matrix,

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    classification_report

)
# 2. LOAD THE TITANIC DATASET
df = pd.read_csv('titanic.csv')

print("First five rows:")
print(df.head())
 
print("\nDataset shape:")
print(df.shape)
 
print("\nMissing values:")
print(df.isnull().sum())
 
 
# 3. SELECT FEATURES AND TARGET
X = df[
    ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
]
 
y = df["survived"]
 
numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]
 
 
# 4. DATA PREPROCESSING
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
 
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
 
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])
 
 
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1,
    stratify=y
)
 
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
 
 
# 6. LOGISTIC REGRESSION MODEL
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
 
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
 
print("\nLOGISTIC REGRESSION")
print("First predictions:", y_pred[:10])
print("Actual values:", y_test.to_numpy()[:10])
print("First predicted probabilities:\n", y_prob[:5])
 
 
# 7. LOGISTIC REGRESSION EVALUATION
cm_lr = confusion_matrix(y_test, y_pred)
 
print("\nConfusion Matrix:")
print(cm_lr)
 
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
 
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Did not survive", "Survived"],
    zero_division=0
))
 
ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["Did not survive", "Survived"]
).plot()
 
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()
 
 
# 8. LOGISTIC REGRESSION PREDICTION RESULTS
results_lr = pd.DataFrame({
    "Actual Survival": y_test.to_numpy(),
    "Predicted Survival": y_pred,
    "Survival Probability": y_prob[:, 1]
}, index=y_test.index)
 
print("\nLogistic Regression Results:")
print(results_lr.head(10))
 
 
# 9. PREDICT FOR A NEW PASSENGER
new_passenger = pd.DataFrame([{
    "pclass": 3,
    "sex": "male",
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 5,
    "embarked": "S"
}])
 
prediction_lr = model.predict(new_passenger)[0]
probability_lr = model.predict_proba(new_passenger)[0, 1]
 
print("\nNEW PASSENGER — LOGISTIC REGRESSION")
print("Prediction:", prediction_lr)
print("Survival probability:", round(probability_lr, 4))
 
if prediction_lr == 1:
    print("Predicted outcome: Survived")
else:
    print("Predicted outcome: Did not survive")
 
 
# 10. DECISION TREE CLASSIFIER
tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=1
)
 
model_DT = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", tree)
])
 
model_DT.fit(X_train, y_train)
 
y_pred_DT = model_DT.predict(X_test)
y_prob_DT = model_DT.predict_proba(X_test)
 
 
# 11. DECISION TREE EVALUATION
cm_DT = confusion_matrix(y_test, y_pred_DT)
 
accuracy_DT = accuracy_score(y_test, y_pred_DT)
precision_DT = precision_score(
    y_test, y_pred_DT, zero_division=0
)
recall_DT = recall_score(
    y_test, y_pred_DT, zero_division=0
)
f1_DT = f1_score(
    y_test, y_pred_DT, zero_division=0
)
 
print("\nDECISION TREE")
print("Confusion Matrix:\n", cm_DT)
print("Accuracy:", accuracy_DT)
print("Precision:", precision_DT)
print("Recall:", recall_DT)
print("F1 Score:", f1_DT)
 
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_DT,
    target_names=["Did not survive", "Survived"],
    zero_division=0
))
 
ConfusionMatrixDisplay(
    confusion_matrix=cm_DT,
    display_labels=["Did not survive", "Survived"]
).plot()
 
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.show()
 
 
# 12. DECISION TREE STRUCTURE
feature_names = model_DT.named_steps[
    "preprocessing"
].get_feature_names_out()
 
plt.figure(figsize=(20, 10))
 
plot_tree(
    model_DT.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)
 
plt.title("Decision Tree for Titanic Survival")
plt.tight_layout()
plt.show()
 
 
# 13. FEATURE IMPORTANCE
importance = model_DT.named_steps[
    "classifier"
].feature_importances_
 
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)
 
print("\nFeature Importance:")
print(importance_df)
 
plt.figure(figsize=(10, 6))
 
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)
 
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.show()
 
 
# 14. TRAINING VS TESTING ACCURACY
train_accuracy = model_DT.score(X_train, y_train)
test_accuracy = model_DT.score(X_test, y_test)
 
print("\nDECISION TREE ACCURACY COMPARISON")
print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
 
 
# 15. PREDICT THE SAME NEW PASSENGER USING BOTH MODELS
prediction_lr = model.predict(new_passenger)[0]
prediction_DT = model_DT.predict(new_passenger)[0]
 
probability_lr = model.predict_proba(new_passenger)[0, 1]
probability_DT = model_DT.predict_proba(new_passenger)[0, 1]
 
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Predicted Survival": [prediction_lr, prediction_DT],
    "Survival Probability": [probability_lr, probability_DT]
})
 
print("\nNEW PASSENGER MODEL COMPARISON")
print(comparison)


knn_model = Pipeline([
    ("preprocessor",preprocessor),
    ("knn",
KNeighborsClassifier(n_neighbors=5))])

knn_model.fit(X_train,y_train)
y_pred_knn =knn_model.predict(X_test)
y_pred_knn[:0.5]
y_test[:0.5]

accuracy_score(y_test,y_pred_knn)
classification_report(
    y_test,
    y_pred_knn,
    target_name=["Did not survivde","survivde"]
)

cm_KNN = confusion_matrix(y_test,y_pred_knn)


ConfusionMatrixDisplay(
    confusion_matrix=cm_KNN,
    display_labels=["Did not survived","survived"]

).plot()
plt.title("Knn confunstion matrix")
plt.tight_layout()
plt.show

param_grid ={
    'knn_n_neighbors':[3,5,2],
    'knn__weight':['uniform','distance'],
    'knn_ _metric':['euclidean','manhattan']
}
cv= StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)
grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_
best_model = grid_search.best_estimator_
y_pred_bknn = best_model.predict(X_test)
precision_score(y_test, y_pred_bknn)