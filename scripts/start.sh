#!/bin/bash

cd .. && nohup jupyter notebook --port 9000 &> ./logs/process.out &

echo "Research Lab [ STARTED ]"
