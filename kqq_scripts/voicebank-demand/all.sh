
```bash
./scripts/voicebank-demand/pack_audios_to_hdf5s.sh "/home/tiger/datasets/voicebank-demand"
```

## 3. Create indexes for training

```bash
./scripts/voicebank-demand/create_indexes.sh
```

## 4. Train & evaluate & save checkpoints
```bash
./scripts/voicebank-demand/train.sh
```
