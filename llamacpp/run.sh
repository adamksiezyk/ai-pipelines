#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 -m llama_cpp.server --config_file config.json
