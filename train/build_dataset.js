process.env.TF_CPP_MIN_LOG_LEVEL = '2';
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const games = require('./games.js')
const goban = require('./goban.js')

let batchSize = 2;

async function saveBatch(fromIndex, toIndex, path) {
  let statesBatch = [];
  let movesBatch = [];
  let gCount = 0;
  let batchGames = games.slice(fromIndex, Math.min(toIndex, games.length));
  for (let game of batchGames) {
    gCount++;
    console.log(`Encoding game ${gCount}`);
    goban.initGoban()
    for (let move of game.split(';')) {
      if (move.length) {
        let stateTensor = goban.inputTensor();
        let moveIndex = goban.loadSgfMove(move);
        statesBatch.push(stateTensor);
        movesBatch.push(moveIndex);
      }
    }
  } console.log(`Processed ${gCount} games`)
  const X = tf.stack(statesBatch);
  const Y = tf.tensor1d(movesBatch, 'int32');
  const Xbuf = Buffer.from(X.dataSync().buffer);
  const Ybuf = Buffer.from(Y.dataSync().buffer);
  if (!fs.existsSync(path)) fs.mkdirSync(path, { recursive: true });
  const XStream = fs.createWriteStream(`${path}/X.bin`, { flags: 'a' });
  const YStream = fs.createWriteStream(`${path}/Y.bin`, { flags: 'a' });
  XStream.write(Xbuf);
  YStream.write(Ybuf);
  XStream.end();
  YStream.end();
  X.dispose();
  Y.dispose();
  console.log(`Saved ${X.shape[0]} total samples to ${path}`);
}

(async () => {
  for (let i = 0; i < 10; i += batchSize) {
    await saveBatch(i, i + batchSize, './dataset');
  }
})();
