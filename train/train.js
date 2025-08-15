process.env.TF_CPP_MIN_LOG_LEVEL = '2';
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const path = './dataset';
const featureSize = 5776;    // number of NN input features
const batchSize = 10000;     // number of samples to read from file at once
const trainBatchSize = 128;  // number of samples per training batch

const channels = 96;
const boardSize = 19;        // 19x19 board
const inputChannels = 16;    // as per 19*19*16 = 5776
const inputFeatures = 5776;  // number of flat input features

// Shuffle single batch
function shuffleBatch(Xarr, Yarr, featureSize) {
  const n = Yarr.length;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmpY = Yarr[i];
    Yarr[i] = Yarr[j];
    Yarr[j] = tmpY;
    for (let k = 0; k < featureSize; k++) {
      const tmpX = Xarr[i * featureSize + k];
      Xarr[i * featureSize + k] = Xarr[j * featureSize + k];
      Xarr[j * featureSize + k] = tmpX;
    }
  }
}

// Generator that streams the binary files and yields batches of trainBatchSize
function* dataGenerator() {
  const Xfd = fs.openSync(`${path}/X.bin`, 'r');
  const Yfd = fs.openSync(`${path}/Y.bin`, 'r');
  const XBuffer = Buffer.alloc(batchSize * featureSize * 4);
  const YBuffer = Buffer.alloc(batchSize * 4);
  let bytesReadX, bytesReadY;
  do {
    bytesReadX = fs.readSync(Xfd, XBuffer, 0, XBuffer.length, null);
    bytesReadY = fs.readSync(Yfd, YBuffer, 0, YBuffer.length, null);
    if (bytesReadX === 0 || bytesReadY === 0) break;
    const actualBatchSize = bytesReadX / (featureSize * 4);
    const Xbatch = new Float32Array(XBuffer.buffer, XBuffer.byteOffset, actualBatchSize * featureSize);
    const Ybatch = new Int32Array(YBuffer.buffer, YBuffer.byteOffset, actualBatchSize);
    shuffleBatch(Xbatch, Ybatch, featureSize);
    for (let i = 0; i < actualBatchSize; i += trainBatchSize) {
      const end = Math.min(i + trainBatchSize, actualBatchSize);
      yield {
        xs: tf.tensor2d(Xbatch.slice(i * featureSize, end * featureSize), [end - i, featureSize]),
        ys: tf.tensor1d(Ybatch.slice(i, end), 'int32')
      };
    }
  } while (bytesReadX > 0 && bytesReadY > 0);
  fs.closeSync(Xfd);
  fs.closeSync(Yfd);
}

// Function to create model
function createModel() {
  const input = tf.input({ shape: [inputFeatures] });
  const reshaped = tf.layers.reshape({ targetShape: [boardSize, boardSize, inputChannels] }).apply(input);
  let x = tf.layers.conv2d({
    filters: channels,
    kernelSize: 7,
    padding: 'same',
    activation: 'relu'
  }).apply(reshaped);
  for (let i = 0; i < 4; i++) {
    x = tf.layers.conv2d({
      filters: channels,
      kernelSize: 5,
      padding: 'same',
      activation: 'relu'
    }).apply(x);
  }
  for (let i = 0; i < 6; i++) {
    x = tf.layers.conv2d({
      filters: channels,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu'
    }).apply(x);
  }
  x = tf.layers.flatten().apply(x);
  const output = tf.layers.dense({ units: 361 }).apply(x); // 19x19 output
  return tf.model({ inputs: input, outputs: output });
}

// Create a dataset from the generator
const dataset = tf.data.generator(dataGenerator);

// Create the model
const model = createModel();
model.summary();
(async () => {
  // Take just one batch to inspect
  const oneBatch = await dataset.take(1).toArray();
  const batch = oneBatch[0];

  console.log('xs shape:', batch.xs.shape);
  console.log('ys shape:', batch.ys.shape);

  // Print a portion of the data
  batch.xs.print();
  batch.ys.print();
  console.log(batch.xs.shape)
})();
