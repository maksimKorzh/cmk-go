# CMK Go
Play Go/Weiqi/Baduk with a Neural Net in a web browser

# Work in progress...
Currently NN is under training

# Web interface
<a href="https://maksimkorzh.github.io/cmkgo/">PLAY with dummy net</a>

# Training results
I trained two policy nets (no value heads) on Intel Core i5-10400 CPU @ 2.90GHz × 6:
<li>cmkgo-b1c96-s1067091</li>
<li>cmkgo-b6c96-s1067091</li>
<br>
Following table shows training results comparison:
<table>
  <tr>
    <th>Name</th>
    <th>Arch</th>
    <th>Size</th>
    <th>Time</th>
    <th>Samples</th>
    <th>Epochs</th>
    <th>Loss</th>
    <th>Acc. seen</th>
    <th>Acc. pred</th>
  </tr>
  <tr>
    <td>cmkgo-cnn11c96-s1067091</td>
    <td>11 layer CNN with dense output layer, 96 convolutional filters</td>
    <td>~56Mb</td>
    <td>~28hr</td>
    <td>1067091</td>
    <td>5</td>
    <td>2.14</td>
    <td>39.15% (~20k samples)</td>
    <td>32.05% (~17k samples)</td>
  </tr>
  <tr>
    <td>cmkgo-b6c96-s1067091</td>
    <td>6 residual blocks, 96 convolutional filters (katago style)</td>
    <td>~5Mb</td>
  </tr>
</table>
<br>
<strong>cmkgo-b6c96-s1067091</strong> training progress (~8.5hrs/epoch)
<br>
<br>
<table>
 <tr>
   <th>Epoch</th>
   <th>Loss</th>
   <th>Acc. seen</th>
   <th>Acc. pred</th>
   <th>Self-play strength</th>
   <th>Winrate against me (OGS 9 kyu)</th>
   <th>Winrate against GnuGo (6 kyu)</th>
 </tr>
 <tr>
   <td>1</td>
   <td>3.5185</td>
   <td>N/A</td>
   <td>N/A</td>
   <td>N/A</td>
   <td>N/A</td>
   <td>N/A</td>
 </tr>
 </tr>
   <td>2</td>
   <td>2.6923</td>
   <td>34.84%</td>
   <td>36.43%</td>
   <td>~14 kyu</td>
   <td>0%</td>
   <td>0%</td>
 </tr>
 </tr>
   <td>3</td>
   <td>2.5016</td>
   <td>37.40%</td>
   <td>36.65%</td>
   <td>~6 kyu</td>
   <td>90%</td>
   <td>10%</td>
 </tr>
 </tr>
   <td>4</td>
   <td>2.3802</td>
   <td>39.27%</td>
   <td>37.03%</td>
   <td>~6 kyu</td>
   <td>100%</td>
   <td>30%</td>
 </tr>
</table>
    
# How to train your own net
    # Installation
    git clone https://github.com/maksimKorzh/cmkgo
    cd cmkgo/train
    
    # x86_64 systems
    npm install
    
    # For raspberry pi 5 (mostly for running "gtp.js"):
    npm install @tensorflow/tfjs-node@4.8.0
    npm rebuild @tensorflow/tfjs-node --build-from-source

    ./download.sh              # downloads games from https://badukmovies.com
    python extract_games.py    # extract moves from SGFs, write them to "games.js"
    node build_dataset.js      # creates X.bin and Y.bin training data files
    node train.js              # train neural net (you may want to adjust params or model arch)
    node gtp.js                # used to play against the net in GoGUI, make sure "path/to/model" is correct
    ./pack-model.sh            # Make model 75% smaller (optional)
    
    accuracy.js                # used to evaluate NN accuracy in percents,
                               # assumes "model.json", "weights.bin",
                               # "X_train.bin", "Y_train.bin", "X_val.bin", "Y_val.bin"
                               # to be in "./test" folder

    NOTE: you may run out of RAM if processing too many games at once,
          so the suggested way is to extract games year by year (alter
          extract_games.py) and then run "build_dataset.js" to append
          newly encoded games to "./dataset/X.bin" and "./dataset/Y.bin".
          Alternatively you can populate "games.js" with your own games
          all at once or in stages - "build_dataset.js" would append
          encoded moves from listed games to *.bin files. Make sure
          you keep track of eventual number of positions and put this
          number into totalSamples variable in "train.js" to have a
          proper current/total samples rate.
