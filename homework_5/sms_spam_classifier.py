import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Загрузка датасета
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# 2. Преобразование меток
df["label_num"] = df.label.map({"ham": 0, "spam": 1})

# 3. Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
   df["message"], df["label_num"], test_size=0.2, random_state=42
)

# 4. Векторизация текста
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Обучение модели
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_vec, y_train)

# 6. Оценка модели
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# 7. Пример предсказания
def predict_spam(text):
   vec = vectorizer.transform([text])
   prediction = model.predict(vec)[0]
   return "СПАМ" if prediction == 1 else "НЕ СПАМ"

# 8. Примеры
print(predict_spam("Congratulations! You won a $1000 Walmart gift card."))
print(predict_spam("Hey, are we still on for dinner tonight?"))

# 9. Сохраняем модель и векторайзер
joblib.dump(model, "spam_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")