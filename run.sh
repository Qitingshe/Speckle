#!/bin/bash
python3 main.py train \
	--train-data-root=data/train_v3 \
	--lr=0.005 \	
	--batch-size=128 \	
	--model='ResNet34' \
	--max-epoch=1 \
	--load-model-path=''

