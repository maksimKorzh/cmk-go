#!/bin/bash
tensorflowjs_converter              \
  --input_format=tfjs_layers_model  \
  --output_format=tfjs_layers_model \
  --quantization_bytes 1            \
  ./model/model.json                \
  ./small/
