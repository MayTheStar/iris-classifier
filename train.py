# train.py
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# 1. تحميل البيانات
data = load_iris()
X = data.data
y = data.target

# 2. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. إعداد pipeline (تحجيم + نموذج)
pipeline = Pipeline([
    ('scaler', StandardScaler()),          # Standardization مهم للنماذج الخطية
    ('clf', RandomForestClassifier(random_state=42))
])


# 4. بحث عن أفضل معلمات 
param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [None, 5, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
best_model = grid.best_estimator_

# 5. تقييم على مجموعة الاختبار
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. حفظ النموذج 
joblib.dump(best_model, "App/iris_pipeline.joblib")
print("Saved model to iris_pipeline.joblib")
