# cmk-go
Play Go/Weiqi/Baduk with a Neural Net in a web browser

# Install (RPI 5)
    npm install @tensorflow/tfjs-node@4.8.0
    npm rebuild @tensorflow/tfjs-node --build-from-source
    
    pip install tensorflowjs # to install converter
    # make model smaller
    tensorflowjs_converter \
      --input_format=tf_saved_model \
        --output_format=tfjs_layers_model \
          --quantization_bytes 1 \
            /path/to/saved_model \
              /path/to/output_model
# WORK IN PROGRESS...
