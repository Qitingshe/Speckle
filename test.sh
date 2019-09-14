#!/bin/bash
python main.py test 	--test-data-root=/home/lab/tmp/16bit500m/test/ 	--batch-size=128 	--model='ResNet34' 	--load-model-path='checkpoints/resnet34_0622_22:23:20.pth'