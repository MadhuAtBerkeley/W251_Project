import torch
import torch.onnx
from models.mtcnn import PNet, RNet, ONet
import tensorrt
import subprocess
USE_FP16 = 1

def gen_trt_engine(net, onnx_file, trt_file, dummy_input):
   
    torch.onnx.export(net, dummy_input, onnx_file, verbose=False)

    # step out of Python for a moment to convert the ONNX model to a TRT engine using trtexec
    if USE_FP16:
        cmd = ['/usr/src/tensorrt/bin/trtexec', '--onnx='+str(onnx_file), '--saveEngine='+str(trt_file),  '--explicitBatch', '--inputIOFormats=fp16:chw', '--outputIOFormats=fp16:chw', '--fp16']
    else:
        cmd = ['/usr/src/tensorrt/bin/trtexec', '--onnx='+str(onnx_file), '--saveEngine='+str(trt_file),  '--explicitBatch']
    subprocess.call(cmd)
    return
    
def main():
    
    # Generate pnet
    onnx_file = 'pnet1.onnx'
    trt_file = './models/trt_engines/pnet_engine1.trt'
    pnet = PNet()
    H = 700
    W = int(1280*12/40)
    dummy_input=torch.randn(1, 3, H, W)
    gen_trt_engine(pnet, onnx_file, trt_file, dummy_input)
    
    
    # Generate rnet
    onnx_file = 'rnet1.onnx'
    trt_file = './models/trt_engines/rnet_engine1.trt'
    rnet = RNet()
    dummy_input=torch.randn(64, 3, 24, 24)
    gen_trt_engine(rnet, onnx_file, trt_file, dummy_input)
    
    # Generate onet
    onnx_file = 'onet1.onnx'
    trt_file = './models/trt_engines/onet_engine1.trt'
    onet = ONet()
    dummy_input=torch.randn(64, 3, 48, 48)
    gen_trt_engine(onet, onnx_file, trt_file, dummy_input)

if __name__ == '__main__':
    main()