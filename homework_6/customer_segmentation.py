import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Загрузка датасета
df = pd.read_csv("Mall_Customers.csv")

# Подготовка признаков
features = df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Обучение KMeans
kmeans = KMeans(n_clusters=8, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# Анализ кластеров
cluster_info = df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1)

# Автоматическое присвоение названий кластерам по логике:
# высокая трата + низкий доход → "Растратчики"
# высокая трата + высокий доход → "Премиум"
# низкая трата + низкий доход → "Экономные"
# низкая трата + высокий доход → "Сдержанные богачи"
# средние значения → "Средний сегмент"

cluster_names = {}
for idx, row in cluster_info.iterrows():
   income = row["Annual Income (k$)"]
   spending = row["Spending Score (1-100)"]
   if income < 40 and spending > 60:
      name = "Растратчики"
   elif income > 70 and spending > 60:
      name = "Премиум"
   elif income < 40 and spending < 40:
      name = "Экономные"
   elif income > 70 and spending < 40:
      name = "Сдержанные богачи"
   else:
      name = "Средний сегмент"
   cluster_names[idx] = name

# Присваиваем названия
df["Тип клиента"] = df["Cluster"].map(cluster_names)

# Центроиды в оригинальном масштабе
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_original, columns=["Annual Income (k$)", "Spending Score (1-100)"])
centroids_df["Тип клиента"] = centroids_df.index.map(cluster_names)

# Визуализация
plt.figure(figsize=(10, 6))
sns.scatterplot(
   data=df,
   x="Annual Income (k$)",
   y="Spending Score (1-100)",
   hue="Тип клиента",  # ← ОШИБКА: такого столбца нет
   palette="Set2",
   s=80
)
# Отметим центроиды
plt.scatter(
   centroids_df["Annual Income (k$)"],
   centroids_df["Spending Score (1-100)"],
   s=40,
   c="black",
   label="Центроиды",
   marker="X"
)
for i, row in centroids_df.iterrows():
   plt.text(row["Annual Income (k$)"] + 1, row["Spending Score (1-100)"] + 1, row["Тип клиента"], fontsize=9)

print(cluster_info)

plt.title("Сегментация клиентов с подписями и центроидами")
plt.xlabel("Годовой доход (в тыс. $)")
plt.ylabel("Оценка расходов (1–100)")
plt.legend(title="Тип клиента", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
