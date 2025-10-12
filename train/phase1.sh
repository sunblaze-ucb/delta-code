CLIP_HIGH=0.3
LR=5e-7
SCORE_MODE=pass_rate  # per test rate
ENABLE_FEEDBACK=False
ENABLE_EXPERIENCE_REPLAY=False
SQLITE_LOGGING=True
TRAIN_LIST="manufactoria/has_train 1.0"
EVAL_LIST="manufactoria/has_test 50"
DESCRIPTION="manufactoria_has_per-test-rate"
EXP_NAME=Qwen3_4B_Instruct_manufactoria_has_per-test-pass_rate
BASE_MODEL=Qwen3/Qwen3-4B-Instruct-2507


python mason.py \
    --cluster XXXXX \
    --workspace XXXXX \
    --priority XXXXX \
    --preemptible \
    --num_nodes 2 \
    --resumable \
    --image ai2/cuda12.8-dev-ubuntu22.04-torch2.7.0 \
    --description "${DESCRIPTION}" \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\&  \
    source configs/beaker_configs/manufactoria_api_setup.sh \&\& \
    python open_instruct/grpo_fast_code.py \
    --exp_name $EXP_NAME \
    --beta 0.0 \
    --num_unique_prompts_rollout 48 \
    --num_samples_per_prompt_rollout 16 \
    --kl_estimator kl3 \
    --learning_rate $LR \
    --dataset_mixer_list $TRAIN_LIST \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $EVAL_LIST \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path $BASE_MODEL \
    --apply_verifiable_reward true \
    --manufactoria_api_url \$MANUFACTORIA_API_URL/test_solution \
    --manufactoria_scoring_mode $SCORE_MODE \
    --feedback_max_iterations 1 \
    --enable_feedback_generation $ENABLE_FEEDBACK \
    --enable_experience_replay $ENABLE_EXPERIENCE_REPLAY \
    --enable_sqlite_logging $SQLITE_LOGGING \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --chat_template_name qwen3 \
    --total_episodes 1000000 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 8 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --clip_higher $CLIP_HIGH \
    --vllm_num_engines 8 \
    --lr_scheduler_type constant \
    --seed 1 \
    --num_evals 100 \
    --save_freq 40 \
    --gradient_checkpointing \
    --checkpoint_state_freq 100 \
    --sqlite_db_path output/dbs/$EXP_NAME.db \
    --with_tracking 
    