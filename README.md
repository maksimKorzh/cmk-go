# CMK Go
Play Go/Weiqi/Baduk with a Neural Net in a web browser

# Work in progress...
Currently NN is under training

# Web interface
<a href="https://maksimkorzh.github.io/cmkgo/">PLAY with dummy net</a>

# Training results
    # 11 layer CNN with dense output layer
    Size:        56Mb
    Epochs:      5
    Train time: ~28 hours on Intel Core i5-10400 CPU @ 2.90GHz × 6
    Avg. loss:   2.14
    Accuracy:    39.15% (~20k training samples)
                 32.05% (~17k validation samples)
    
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
    node gtp.js                # used to play agains the net in GoGUI, make sure "path/to/model" is correct
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
