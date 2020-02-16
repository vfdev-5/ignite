#!/bin/bash

CIFAR10_DATASET_PATH=/home/data0/cifar10

echo "Run benchmark data-echoing, ResNet34 on CIFAR10"

echo "\n\n-- no data-echoing"
python main.py --network="resnet34" --params="data_path=$CIFAR10_DATASET_PATH"


for num_echoes in 3 8 12; do
    echo "\n\n-- example-echoing before dataaug, num_echoes=$num_echoes"
    python main.py --network="resnet34" --params="data_path=$CIFAR10_DATASET_PATH;with_example_echoing=True;num_echoes=$num_echoes"
done

for num_echoes in 3 8 12; do
    echo "\n\n-- example-echoing after dataaug, num_echoes=$num_echoes"
    python main.py --network="resnet34" --params="data_path=$CIFAR10_DATASET_PATH;with_example_echoing=True;echoing_before_dataaug=False;num_echoes=$num_echoes"
done
