_base_ = "./olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen.py"

# Official OLMoEarth-v1: both BRIGHT dates use the Sentinel-2 L2A proxy path.
# Do not install the freeze hook: all encoder parameters used by this backbone
# are trainable from the first iteration.
custom_hooks = []

work_dir = "./work_dirs/olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-finetune"
