// eval_fixed.js
process.env.TF_CPP_MIN_LOG_LEVEL = '2';
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// -------- Parameters (must match training) --------
const featureSize = 5776; // 19 * 19 * 16
const modelDir = 'test';  // checkpoint directory

// Fixed datasets
const trainX = './test/X_train.bin';
const trainY = './test/Y_train.bin';
const valX   = './test/X_val.bin';
const valY   = './test/Y_val.bin';

// -------- Load entire dataset into memory --------
function loadDataset(xFile, yFile) {
  const Xbuf = fs.readFileSync(xFile);
  const Ybuf = fs.readFileSync(yFile);

  const X = new Float32Array(Xbuf.buffer, Xbuf.byteOffset, Xbuf.byteLength / 4);
  const Y = new Int32Array(Ybuf.buffer, Ybuf.byteOffset, Ybuf.byteLength / 4);

  const numSamples = Y.length;
  const xs = tf.tensor2d(X, [numSamples, featureSize], 'float32');
  const ys = tf.tensor1d(Y, 'int32');

  return { xs, ys, numSamples };
}

// -------- Evaluate model --------
async function evaluate(xFile, yFile, label) {
  const model = await tf.loadLayersModel(`file://${modelDir}/model.json`);
  console.log(`Model loaded from ${modelDir}`);

  const { xs, ys, numSamples } = loadDataset(xFile, yFile);

  const preds = model.predict(xs);
  const predLabels = preds.argMax(-1).dataSync();
  const trueLabels = ys.dataSync();

  let correct = 0;
  for (let i = 0; i < numSamples; i++) {
    if (predLabels[i] === trueLabels[i]) correct++;
  }

  console.log(`${label} Accuracy: ${(100 * correct / numSamples).toFixed(2)}% (${numSamples} moves)`);

  xs.dispose();
  ys.dispose();
  preds.dispose();
}

// -------- Run --------
(async () => {
  await evaluate(trainX, trainY, 'Train');
  await evaluate(valX, valY, 'Validation');
})();
