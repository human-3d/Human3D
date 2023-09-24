#!/bin/bash

python main.py \
general.experiment_name="Mask3D_eval" \
general.project_name="mask3d_humanseg" \
data/datasets=egobody \
general.num_targets=16 \
data.num_labels=16 \
model=mask3d \
loss=set_criterion \
model.num_queries=5 \
trainer.check_val_every_n_epoch=1 \
general.topk_per_image=-1 \
model.non_parametric_queries=false \
trainer.max_epochs=36 \
data.batch_size=4 \
data.num_workers=10 \
general.reps_per_epoch=1 \
general.save_visualizations=false \
model.config.backbone._target_=models.Res16UNet18B \
data.part2human=true \
loss.num_classes=2 \
model.num_classes=2 \
callbacks=callbacks_instance_segmentation_human \
general.checkpoint="checkpoints/mask3d.ckpt" \
general.train_mode=false \
general.save_visualizations=false
