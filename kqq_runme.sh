CONFIG_YAML="kqq_scripts/musdb18/configs/train/resnet143_vocals_ismir2021.yaml"

CHECKPOINT_PATH="/home/tiger/my_code_2019.12-/python/music_source_separation/workspaces/music_source_separation/checkpoints/musdb18/train/config=resnet143_vocals_ismir2021,gpus=2/step=300000.pth"

CUDA_VISIBLE_DEVICES=3 python3 music_source_separation/inference.py \
    --config_yaml=$CONFIG_YAML \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path='resources/vocals_accompaniment_10s.mp3' \
    --output_path='sep_results/vocals_accompaniment_10s.mp3'



CUDA_VISIBLE_DEVICES=4 python3 bytesep/inference.py \
    --config_yaml="kqq_scripts/train/violin_piano/configs/02_violin.yaml" \
    --checkpoint_path="/home/tiger/my_code_2019.12-/python/music_source_separation/workspaces/bytesep/checkpoints/violin-piano/train/config=02_violin,gpus=1/step=20000.pth" \
    --audio_path='resources/brahms_violin.mp3' \
    --output_path='sep_results/brahms_violin.mp3'


CUDA_VISIBLE_DEVICES=4 python3 bytesep/inference.py \
    --config_yaml="kqq_scripts/train/piano_symphony/configs/01_piano.yaml" \
    --checkpoint_path="workspaces/bytesep/checkpoints/piano-symphony/train/config=01_piano,gpus=1/step=10000.pth" \
    --audio_path='resources/rach_no2_30s.mp3' \
    --output_path='sep_results/rach_no2_30s.mp3'

CUDA_VISIBLE_DEVICES=4 python3 bytesep/inference_many.py \
    --config_yaml="/home/tiger/my_code_2019.12-/python/music_source_separation/kqq_scripts/train/voicebank-musdb18/configs/01.yaml" \
    --checkpoint_path="/home/tiger/my_code_2019.12-/python/music_source_separation/workspaces/bytesep/checkpoints/voicebank-musdb18/train/config=01,gpus=1/step=30000.pth" \
    --audios_dir='resources/音频物料 - 人声+bgm的内容_mp3s' \
    --output_dir='sep_results/音频物料 - 人声+bgm的内容_mp3s_b'
