import tensorflow as tf
from tensorflow.python.client import device_lib

def check_cuda():
    devices = device_lib.list_local_devices()
    return any("GPU" in str(device.device_type) for device in devices)

print("CUDA Enabled:" , check_cuda())
