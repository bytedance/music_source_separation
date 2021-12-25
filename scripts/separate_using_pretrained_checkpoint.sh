#!/bin/bash
python3 -m bytesep separate \
    --source_type="vocals" \
    --audio_path="./resources/vocals_accompaniment_10s.mp3" \
    --output_path="separated_results/output.mp3"
