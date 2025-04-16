import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)g

X = np.random.randint(0, 2, size=(100, 12))  # 100 примеров, 12 бинарных признаков
Y = np.array([[1, 0] if x.sum() > 6 else [0, 1] for x in X])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# Сохранение данных в файлы
np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y, fmt='%d')

model = keras.Sequential([
    keras.layers.Dense(12, input_shape=(12,), activation='sigmoid'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Оцениваем качество на тестовой выборке
y_pred = np.argmax(model.predict(X_test), axis=1)

# Истинные классы тоже в виде индексов
y_true = np.argmax(y_test, axis=1)
print("Accuracy:", accuracy_score(y_true, y_pred))
# График изменения функции ошибки
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
