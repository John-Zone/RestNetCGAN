#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --channel_num_spectral 220 \ 
	--channel_num_spatial 10 \
	--class_num 16 \
	--noise_size 50 \
    --output_dir ./0414/ \
    --summary_dir ./0414/log/ \
    --mode train \
    --is_training True \
    --batch_size 30 \
    --input_data_dir ./data/train.mat \
    --pre_trained_model False \
    --checkpoint None \
	--name_queue_capacity 4096 \
	--image_queue_capacity 4096 \
	--ratio 0.001 \
	--decay_step 1000 \
	--decay_rate 0.1 \
	--stair True \
	--beta 0.9 \
	--max_iter 100000 \
	--queue_thread 10 \
	--learning_rate 0.005 \

