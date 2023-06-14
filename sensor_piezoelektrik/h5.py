import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import h5py

# Baca dataset
data = pd.read_csv('arrrghh.csv', delimiter=';')

# Pisahkan fitur dan label
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Konversi label ke tipe data integer
y = y.astype(float).astype(int)

# Split data menjadi training set dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Mengubah label menjadi one-hot encoding
num_classes = np.max(y) + 1
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Membangun model neural network
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Plot grafik accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot grafik loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Menampilkan epochs
print("Epochs:", len(history.history['accuracy']))

# Memasukkan nilai fitur manual
manual_input = np.zeros((1, X_train.shape[1]))
for i in range(X_train.shape[1]):
    manual_input[0, i] = float(input(f"Masukkan nilai fitur ke-{i+1}: "))

# Melakukan penskalaan fitur menggunakan scaler yang sudah didefinisikan sebelumnya
scaled_manual_input = scaler.transform(manual_input)

# Melakukan prediksi menggunakan model neural network
prediction_manual = model.predict(scaled_manual_input)

# Menampilkan hasil prediksi manual
predicted_class_manual = np.argmax(prediction_manual)
if predicted_class_manual == 1:
    print("Prediksi kesehatan janin manual: Normal")
elif predicted_class_manual == 2:
    print("Prediksi kesehatan janin manual: Suspect")
elif predicted_class_manual == 3:
    print("Prediksi kesehatan janin manual: Pathological")
else:
    print("Prediksi tidak valid")

# Simpan model ke dalam file .h5
model.save('model.h5')
