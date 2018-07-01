#!/usr/bin/env bash

cd ..

docker build -t xtnc/rlab:latest .

docker push xtnc/rlab:latest