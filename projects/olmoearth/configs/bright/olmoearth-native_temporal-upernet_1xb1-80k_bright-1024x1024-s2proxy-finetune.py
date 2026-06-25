_base_ = "./olmoearth-native_temporal-upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen.py"

# Token-level early-fusion fine-tuning: train the selected OLMoEarth-v1 encoder
# parameters from iteration zero.
custom_hooks = []

work_dir = "./work_dirs/olmoearth-native_temporal-upernet_1xb1-80k_bright-1024x1024-s2proxy-finetune"
