#!/bin/bash
R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/GR1
for m in warm_decode d4096; do $R/g4.sh $R/cfg-gr1san.json $m; done
