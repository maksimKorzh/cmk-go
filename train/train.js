// Packages
process.env.TF_CPP_MIN_LOG_LEVEL = '2';
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Params
const datasetPath = './dataset';  // dataset folder name
const featureSize = 5776;         // number of NN input features
const batchSize = 10000;          // number of samples to read from file at once
const trainBatchSize = 128;       // number of samples per training batch
const channels = 96;              // number of convolutional filters
const boardSize = 19;             // 19x19 board
const inputChannels = 16;         // 19*19*16 = 5776
const inputFeatures = 5776;       // number of flat input features
const totalSamples = 297802;      // "build_dataset.js" prints this number

// Total batches
const totalBatches = Math.ceil(totalSamples / trainBatchSize);

// Generator that streams binary files and yields batches
async function* dataGenerator() {
  const Xfd = fs.openSync(`${datasetPath}/X.bin`, 'r');
  const Yfd = fs.openSync(`${datasetPath}/Y.bin`, 'r');
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
        xs: tf.tensor2d(Xbatch.slice(i * featureSize, end * featureSize), [end - i, featureSize], 'float32'),
        ys: tf.tensor1d(Ybatch.slice(i, end), 'float32')
      };
    }
  } while (bytesReadX > 0 && bytesReadY > 0);
  fs.closeSync(Xfd);
  fs.closeSync(Yfd);
}

// Shuffle a single batch
function shuffleBatch(Xarr, Yarr, featureSize) {
  const n = Yarr.length;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [Yarr[i], Yarr[j]] = [Yarr[j], Yarr[i]];
    for (let k = 0; k < featureSize; k++) {
      [Xarr[i * featureSize + k], Xarr[j * featureSize + k]] =
        [Xarr[j * featureSize + k], Xarr[i * featureSize + k]];
    }
  }
}

// Create model -- katago-like b6c96 policy only (no value head) network --
function createModel() {
  const input = tf.input({ shape: [inputFeatures] });
  const reshaped = tf.layers.reshape({ targetShape: [boardSize, boardSize, inputChannels] }).apply(input);
  let x = tf.layers.conv2d({
    filters: channels,
    kernelSize: 3,
    padding: 'same',
    useBias: false,
    kernelInitializer: tf.initializers.heNormal()
  }).apply(reshaped);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.activation({ activation: 'relu' }).apply(x);
  for (let i = 0; i < 6; i++) {
    const shortcut = x;
    let r = tf.layers.conv2d({
      filters: channels,
      kernelSize: 3,
      padding: 'same',
      useBias: false,
      kernelInitializer: tf.initializers.heNormal()
    }).apply(x);
    r = tf.layers.batchNormalization().apply(r);
    r = tf.layers.activation({ activation: 'relu' }).apply(r);
    r = tf.layers.conv2d({
      filters: channels,
      kernelSize: 3,
      padding: 'same',
      useBias: false,
      kernelInitializer: tf.initializers.heNormal()
    }).apply(r);
    r = tf.layers.batchNormalization().apply(r);
    x = tf.layers.add().apply([shortcut, r]);
    x = tf.layers.activation({ activation: 'relu' }).apply(x);
  }
  let policy = tf.layers.conv2d({
    filters: 2,
    kernelSize: 1,
    padding: 'same',
    useBias: false,
    kernelInitializer: tf.initializers.heNormal()
  }).apply(x);
  policy = tf.layers.batchNormalization().apply(policy);
  policy = tf.layers.activation({ activation: 'relu' }).apply(policy);
  policy = tf.layers.flatten().apply(policy);
  policy = tf.layers.dense({
    units: boardSize * boardSize,
    activation: 'softmax',
    kernelInitializer: tf.initializers.heNormal()
  }).apply(policy);
  return tf.model({ inputs: input, outputs: policy });
}

// Train model with checkpointing
async function trainModel(model, dataset, epochs, learningRate, checkpointDir, checkpointInterval) {
  const optimizer = tf.train.adam(learningRate);
  let startEpoch = 0;
  let batchIndex = 0;
  let countSamples = 0;
  let samplesSinceLastCkpt = 0;
  const checkpointFile = path.join(checkpointDir, 'checkpoint.json');
  if (!fs.existsSync(checkpointDir)) fs.mkdirSync(checkpointDir, { recursive: true });
  if (fs.existsSync(checkpointFile)) {
    const ckptData = JSON.parse(fs.readFileSync(checkpointFile, 'utf8'));
    const modelPath = path.join(checkpointDir, 'model.json');
    if (fs.existsSync(modelPath)) {
      model = await tf.loadLayersModel(`file://${modelPath}`);
      console.log('Model checkpoint loaded.');
    }
    startEpoch = ckptData.epoch || 0;
    samplesSinceLastCkpt = ckptData.samplesSinceLastCkpt || 0;
    batchIndex = ckptData.batchIndex || 0;
    countSamples = samplesSinceLastCkpt;
    console.log(`Resuming from epoch ${startEpoch}, ${samplesSinceLastCkpt} samples since last checkpoint.`);
  } else { console.log('No checkpoint found, starting from scratch.'); }
  model.compile({
    optimizer: optimizer,
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });
  for (let epoch = startEpoch; epoch < epochs; epoch++) {
    let epochLoss = 0;
    let batchCount = 0;
    batchIndex = 0;
    if (countSamples => totalSamples) countSamples = 0;
    const iterator = await dataset.iterator();
    for (let i = 0; i < batchIndex; i++) await iterator.next();
    let result = await iterator.next();
    while (!result.done) {
      const { xs, ys } = result.value;
      const history = await model.fit(xs, ys, { batchSize: xs.shape[0], epochs: 1, verbose: 0 });
      const loss = history.history.loss[0];
      if (!Number.isNaN(loss)) epochLoss += loss;
      batchCount++;
      samplesSinceLastCkpt += xs.shape[0];
      countSamples += xs.shape[0];
      batchIndex++;
      const info = `Epoch ${epoch}/${epochs}, Samples ${countSamples}/${totalSamples}, loss ${loss.toFixed(4)}`;
      console.log(info);
      if (samplesSinceLastCkpt >= checkpointInterval) {
        await model.save(`file://${checkpointDir}`);
        fs.writeFileSync(checkpointFile, JSON.stringify({
          epoch,
          batchIndex,
          samplesSinceLastCkpt: countSamples
        }));
        fs.appendFileSync('log.txt', info + '\n');
        console.log(`Checkpoint saved after ${samplesSinceLastCkpt} samples.`);
        samplesSinceLastCkpt = 0;
      }
      xs.dispose();
      ys.dispose();
      result = await iterator.next();
    }
    const avgLoss = batchCount > 0 ? epochLoss / batchCount : 0;
    const info = `Epoch ${epoch + 1}/${epochs}, Avg Loss: ${avgLoss.toFixed(4)}`;
    console.log(info);
    fs.appendFileSync('log.txt', info + '\n');
    await model.save(`file://${checkpointDir}`);
    fs.writeFileSync(checkpointFile, JSON.stringify({
      epoch: epoch + 1,
      batchIndex: 0,
      samplesSinceLastCkpt
    }));
    samplesSinceLastCkpt = 0;
    console.log(`Checkpoint saved at end of epoch ${epoch + 1}.`);
  }
}

// Start training
(async () => {
  const model = createModel();
  model.summary();
  const dataset = tf.data.generator(dataGenerator);
  await trainModel(
    model,
    dataset,
    10,            // epochs
    0.001,         // learning rate
    'ckpt',        // checkpoint directory
    10000          // save checkpoint every N samples
  );
})();
