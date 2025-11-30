#!/bin/sh

for i in $(seq $1 $2); do
    python3 train_dip.py --checkpoint-prefix dip_$i -i $i
done