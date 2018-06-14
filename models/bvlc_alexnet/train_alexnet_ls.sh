#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_ls.prototxt -gpu=all \
    2>&1 | tee models/bvlc_alexnet/logs/alexnet_50e_ls0.2_0.002.log
