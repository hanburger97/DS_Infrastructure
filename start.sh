#!/bin/bash

nohup jupyter notebook --port 9000 &> ./logs/process.out &
tail -f ./logs/process.out
