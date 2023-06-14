const tf = require('@tensorflow/tfjs-node');
const tfn = require('@tensorflow/tfjs-node');
const { DataFrame } = require('pandas-js');

// Baca dataset
const data = new DataFrame().read_csv('arrrghh.csv', { delimiter: ';' });

// Pisahkan fitur dan label
const X = data.iloc(null, [0, -2]).values;
const y = data.iloc(null, -1).values;

// Konversi label ke tipe data integer
const yInt = y.astype(float).astype(int);

// Split data menjadi training set dan test set
const { X: X_train, y: y_train, X: X_test, y: y_test } = tfn.util
  .model_selection
  .train_test_split(X, yInt, { test_size: 0.2, random_state: 42 });

// Normalisasi fitur
const scaler = new StandardScaler();
const X_train_scaled = scaler.fit_transform(X_train);
const X_test_scaled = scaler.transform(X_test);

// Mengubah label menjadi one-hot encoding
const numClasses = tf.max(yInt).dataSync()[0] + 1;
const y_train_oneHot = tf.util.to_categorical(y_train, numClasses);
const y_test_oneHot = tf.util.to_categorical(y_test, numClasses);

// Membangun model neural network
const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [X_train_scaled.shape[1]] }));
model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

// Compile model
model.compile({ loss: 'categoricalCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });

// Training model
async function runModel() {
  const history = await model.fit(X_train_scaled, y_train_oneHot, { epochs: 50, batchSize: 32, validationData: [X_test_scaled, y_test_oneHot] });
  // Blok kode lainnya
}

runModel();

// Plot grafik accuracy
const accuracy = history.history.acc;
const valAccuracy = history.history.val_acc;
const epochs = accuracy.length;

console.log(accuracy);
console.log(valAccuracy);
console.log('Epochs:', epochs);

// Memasukkan nilai fitur manual
const manualInput = Array.from({ length: X_train_scaled.shape[1] }, (_, i) => {
  return parseFloat(prompt(`Masukkan nilai fitur ke-${i + 1}: `));
});

// Melakukan penskalaan fitur menggunakan scaler yang sudah didefinisikan sebelumnya
const scaledManualInput = scaler.transform([manualInput]);

// Melakukan prediksi menggunakan model neural network
const predictionManual = model.predict(scaledManualInput);

// Menampilkan hasil prediksi manual
const predictedClassManual = predictionManual.argMax(1).dataSync()[0];
if (predictedClassManual === 0) {
  console.log('Prediksi kesehatan janin manual: Normal');
} else if (predictedClassManual === 1) {
  console.log('Prediksi kesehatan janin manual: Suspect');
} else if (predictedClassManual === 2) {
  console.log('Prediksi kesehatan janin manual: Pathological');
} else {
  console.log('Prediksi tidak valid');
}
