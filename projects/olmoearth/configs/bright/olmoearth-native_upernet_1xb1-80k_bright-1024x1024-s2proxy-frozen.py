_base_ = "./olmoearth-base_upernet_1xb1-80k_bright-1024x1024-rgbproxy.py"

# Official OLMoEarth-v1: both BRIGHT dates use the Sentinel-2 L2A proxy path.
# Keep the complete downstream backbone frozen for all 80k iterations.
custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=None)]

work_dir = "./work_dirs/olmoearth-native_upernet_1xb1-80k_bright-1024x1024-s2proxy-frozen"
