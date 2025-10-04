# predict_test.py
import joblib
import numpy as np

model = joblib.load("iris_pipeline.joblib")
# مثال: توقع لأول 3 عينات من مجموعة الاختبار
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))
print(model.predict_proba([[5.1, 3.5, 1.4, 0.2]]))
