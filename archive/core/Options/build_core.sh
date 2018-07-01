#!/bin/bash

mkdir tmp

while read p; do
   cp $p.py ./tmp/$p.pyx && cd tmp && cythonize -a $p.pyx && cd ..
done < build_index.txt

