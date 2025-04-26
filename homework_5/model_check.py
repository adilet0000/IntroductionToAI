import joblib

# Загружаем модель и векторайзер
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Функция для предсказания
def predict_spam(text):
   vec = vectorizer.transform([text])
   prediction = model.predict(vec)[0]
   return "СПАМ" if prediction == 1 else "НЕ СПАМ"

# Пример использования
print(predict_spam("Win a free vacation now!"))
print(predict_spam("Let's have a meeting tomorrow."))
print(predict_spam("Hey, are we still on for dinner tonight?"))
print(predict_spam("Congratulations! You won a $1000 Walmart gift card."))