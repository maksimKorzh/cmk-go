process.env.TF_CPP_MIN_LOG_LEVEL = '2';
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const path = './dataset';
const featureSize = 7942;
const batchSize = 10000; // samples per batch

const Xstream = fs.createReadStream(`${path}/X.bin`, { highWaterMark: batchSize * featureSize * 4 });
const Ystream = fs.createReadStream(`${path}/Y.bin`, { highWaterMark: batchSize * 4 });

let Xchunks = [];
let Ychunks = [];

Xstream.on('data', chunk => {
  const arr = new Float32Array(chunk.buffer, chunk.byteOffset, chunk.byteLength / 4);
  Xchunks.push(arr);
});

Ystream.on('data', chunk => {
  const arr = new Int32Array(chunk.buffer, chunk.byteOffset, chunk.byteLength / 4);
  Ychunks.push(arr);
});

Xstream.on('end', () => console.log('X fully read in chunks'));
Ystream.on('end', () => console.log('Y fully read in chunks'));

