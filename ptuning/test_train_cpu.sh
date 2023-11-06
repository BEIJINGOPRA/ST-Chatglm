PRE_SEQ_LEN=128
LR=2e-2

torchrun --standalone --nnodes=1 test_main.py \
    --do_train \
    --train_file /home/yuxie/test_files/poi_bj_filled.csv \
    --validation_file /home/yuxie/test_files/poi_bj_filled.csv \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /home/yuxie/ChatGLM2_6B/STModel \
    --output_dir output/test-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 100 \
    --per_device_eval_batch_size 10 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    --chatglm_config_file /home/yuxie/ChatGLM2_6B/STModel/chatglm2_6b \
    --blip2_config_file /home/yuxie/ChatGLM2_6B/STModel/blip2 \
    --model_work_dir /home/yuxie/ChatGLM2_6B/STModel \
    --prompt_column mAnnoPlaceTitle \
    --response_column embedding \
    --gpu_num 1 \
    --gradient_checkpointing False \
    --no_cuda True \