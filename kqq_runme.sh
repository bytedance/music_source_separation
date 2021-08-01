CONFIG_YAML="kqq_scripts/musdb18/configs/train/resnet143_vocals_ismir2021b.yaml"
CHECKPOINT_PATH="/home/tiger/my_code_2019.12-/python/music_source_separation/workspaces/music_source_separation/checkpoints/musdb18/train/config=resnet143_vocals_ismir2021b,gpus=2/step=200000.pth"

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch music_source_separation/inference.py \
    --config_yaml=$CONFIG_YAML \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path='resources/vocals_accompaniment_10s.mp3' \
    --output_path='sep_results/vocals_accompaniment_10s.mp3'