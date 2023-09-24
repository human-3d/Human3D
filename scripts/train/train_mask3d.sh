#!/bin/bash

### 1) FIRST TRAIN THE MODEL ON SYNTHETIC DATA
python main.py \
general.experiment_name="Mask3D_on_synthetic_data" \
general.project_name="mask3d_humanseg" \
data/datasets=synthetic_humans \
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
general.train_mode=true


### 2) THEN FINETUNE WITH EGOBODY DATA
python main.py \
general.experiment_name="Mask3D_finetuned_on_egobody_data" \
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
general.checkpoint='saved/Mask3D_on_synthetic_data/last.ckpt' \
general.train_mode=true
