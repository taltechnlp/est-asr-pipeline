Weakly supervised training for speaker ID

Train with e.g.:

    python local/audio_trainer/main.py \
      --train-datadir data/train_segmented --dev-datadir data/päevakaja.dev_segmented \
      --gpus 1  --accumulate_grad_batches 16  --learning-rate 0.01 --optimizer-name sgd --val_check_interval 1.0 \
      --freeze-backbone-steps 5000 --segment-length 5

Test, e.g.:
    
    python ./local/audio_trainer/main.py \
      --load-checkpoint lightning_logs/version_283639/checkpoints/last.ckpt \
      --test-datadir data/arvamusfestival.fixed_segmented \
      --gpu 1 --segments-per-spk 0 --segment-length 5 --test-threshold 0.8
