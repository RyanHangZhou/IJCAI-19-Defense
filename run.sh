#!/bin/bash



INPUT_DIR=$1
OUTPUT_FILE=$2

python main.py --dataset mine \
               --input_dir "${INPUT_DIR}"

python defense_combine_ensemble.py \
    --input_dir "results/mine/" \
    --output_file "${OUTPUT_FILE}" \
    --checkpoint_1 /checkpoints/mobilenet_v2.pth.tar   \
    --checkpoint_2 /checkpoints/densenet121.pth.tar   \
    --checkpoint_3 /checkpoints/senet154.pth.tar  \
    --checkpoint_4 /checkpoints/resnet18.pth.tar  \
    --checkpoint_5 /checkpoints/inceptionv4.pth.tar  \
    --checkpoint_6 /checkpoints/inceptionresnetv2.pth.tar \
    --checkpoint_7 /checkpoints/resnet34.pth.tar  \
    --checkpoint_8 /checkpoints/resnet152.pth.tar  \
    --checkpoint_9 /checkpoints/se_resnet50.pth.tar  \
    --checkpoint_10 /checkpoints/xception.pth.tar

    
