#!/bin/bash
R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/BUNDLE1
$R/g4.sh $R/cfg-bundle1.json warm_decode
$R/g4.sh $R/cfg-bundle1.json d4096
echo ABBA-finished $(date)
