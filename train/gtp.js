// Packages
process.env.TF_CPP_MIN_LOG_LEVEL = '2';
const goban = require('./goban.js')


goban.initGoban();
goban.playMove();
