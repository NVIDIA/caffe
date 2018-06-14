#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_quick_B256.prototxt -gpu=all \
    2>&1 | tee models/bvlc_alexnet/logs/alexnet_quick_B256.log
