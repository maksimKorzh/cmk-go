// Packages
process.env.TF_CPP_MIN_LOG_LEVEL = '2';
const readline = require('readline');
const goban = require('./goban.js');

// Read line
var gtp = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

// GTP communication
gtp.on('line', function(command){
  if (command == 'quit') process.exit();
  else if (command.includes('name')) console.log('= CMK Go\n');
  else if (command.includes('protocol_version')) console.log('= 2\n');
  else if (command.includes('version')) console.log('= 1.0\n');
  else if (command.includes('list_commands')) console.log('= protocol_version\nclear_board\n');
  else if (command.includes('boardsize')) console.log('=\n');
  else if (command.includes('clear_board')) { goban.initGoban(); console.log('=\n'); }
  else if (command.includes('showboard')) { console.log('= '); goban.printBoard(); }
  else if (command.includes('play')) {
    let color = (command.split(' ')[1].toLowerCase() == 'b') ? goban.BLACK : goban.WHITE;
    let coord = command.split(' ')[2].toLowerCase();
    if (coord == 'pass') goban.passMove();
    else {
      let col = ' abcdefghjklmnopqrst'.indexOf(coord[0]);
      let row = goban.size - parseInt(coord.slice(1))-1;
      let sq = row * goban.size + col;
      goban.setStone(sq, color);
    } console.log('=\n');
  }
  else if (command.includes('genmove')) {
    let moveHistory = goban.getHistory();
    if (moveHistory.length > 1 && moveHistory.slice(-1)[0].move == 0) console.log('= PASS\n');
    else goban.playMove();
  }
  else console.log('=\n');
}); goban.initGoban();
