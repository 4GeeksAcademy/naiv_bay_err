from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd  # Para manejar el conjunto de datos
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

url = "https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv"
df = pd.read_csv(url)  # Cargamos el dataset en un DataFrame de pandas

#  Eliminación de la variable 'package_name' porque no influye en la clasificación
df.drop(columns=['package_name'], inplace=True)

#  Procesamiento del texto
df["review"] = df["review"].str.strip().str.lower()  # Eliminamos espacios innecesarios y convertimos el texto a minúsculas

# Paso 4: División del conjunto de datos en entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(df["review"], df["polarity"], test_size=0.2, random_state=42)

#   Transformación del texto usando TF-IDF

tfidf_model = TfidfVectorizer(stop_words="english")

# Aplicamos fit solo en los datos de entrenamiento y luego transformamos ambos conjuntos sin convertir a array directamente
X_train_tfidf = tfidf_model.fit_transform(X_train)
X_test_tfidf = tfidf_model.transform(X_test)

# Paso 6: Entrenamiento del modelo Support Vector Machine (SVM)

svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train_tfidf, y_train)  # Entrenamos el modelo con los datos procesados

 

y_pred_svm = svm_model.predict(X_test_tfidf)  # Realizamos predicciones con el conjunto de prueba
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"Exactitud del modelo SVM con TF-IDF: {svm_accuracy:.4f}")

 

joblib.dump(svm_model, "svm_model.pkl")  # Guardamos el modelo entrenado
joblib.dump(tfidf_model, "tfidf_vectorizer.pkl")  # Guardamos el transformador TF-IDF

#  Explora otras alternativas
# - Redes neuronales recurrentes (RNN) pueden capturar el contexto de cada palabra en la oración
# - Modelos como Gradient Boosting (XGBoost) pueden mejorar la precisión
# - BERT (transformers) podría optimizar la comprensión de los comentarios