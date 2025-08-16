# CMK Go
Play Go/Weiqi/Baduk with a Neural Net in a web browser

# How to install
    # x86_64 systems
    git clone https://github.com/maksimKorzh/cmkgo
    cd cmkgo
    npm install
    
    # For raspberry pi 5:
    npm install @tensorflow/tfjs-node@4.8.0
    npm rebuild @tensorflow/tfjs-node --build-from-source

# How to train your own net
    ./download.sh              # downloads games from https://badukmovies.com
    python extract_games.py    # extract moves from SGFs, write them to "games.js"
    node build_dataset.js      # creates X.bin and Y.bin training data files
    node train.js              # train neural net (you may want to adjust params or model arch)
    node gtp.js                # used to play agains the net in GoGUI, make sure "path/to/model" is correct
    
# How to make customly trained net smaller (optional)
    pip install tensorflowjs
    tensorflowjs_converter \
      --input_format=tf_saved_model \
        --output_format=tfjs_layers_model \
          --quantization_bytes 1 \
            /path/to/saved_model \
              /path/to/output_model

# Web interface
Should work with dummy net, but not production-ready yet<br>
work in progress...
