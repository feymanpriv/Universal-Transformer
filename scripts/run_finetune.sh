python -m torch.distributed.launch --nproc_per_node 8 --master_port 24373  main.py \
--cfg ./configs/vit/vit_large_patch14_336.yaml \
--data-path ../data/universal_train_data_v1 \
--batch-size 8 \
--dataset universal \
--amp-opt-level O1 \
--pretrained ./pretrained/ViT-L-14-336px.pt \
--tag universal_train_vit_large_14_336_ft_test \
--output ./output
