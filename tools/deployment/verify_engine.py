"""Compare a built engine against ONNX Runtime using normalized input tensors."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--onnx', required=True)
    p.add_argument('--engine', required=True)
    p.add_argument('--inputs', required=True, help='NPZ: float32 [1,3,640,640] preprocessed tensors')
    p.add_argument('--output', required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--conf', type=float, default=.5)
    args = p.parse_args()
    import numpy as np
    import onnxruntime as ort
    import torch
    import tensorrt as trt
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(args.conf) or not 0 <= args.conf <= 1:
        raise ValueError('conf must be in [0,1]')
    torch.cuda.set_device(args.device)
    session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(args.engine).read_bytes())
    if engine is None:
        raise RuntimeError('Engine deserialization failed')
    context = engine.create_execution_context()
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_dtype(name) != trt.float32 or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            raise ValueError('Expected linear FP32 I/O')
        buffers[name] = torch.empty(tuple(engine.get_tensor_shape(name)), dtype=torch.float32, device='cuda')
        context.set_tensor_address(name, buffers[name].data_ptr())
    if set(buffers) != {'images','pred_logits','pred_boxes'} or tuple(buffers['images'].shape) != (1,3,640,640):
        raise ValueError('Unexpected engine I/O contract')

    def rank(values):
        logits, boxes = values['pred_logits'], values['pred_boxes']
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError('Nonfinite output')
        classes = logits.shape[-1]
        scores = 1 / (1 + np.exp(-np.clip(logits.reshape(-1),-80,80)))
        idx = np.argsort(-scores, kind='stable')[:300]
        boxes = boxes.reshape(-1,4)[idx//classes]
        valid = (scores[idx] >= args.conf) & (boxes[:,2] > 0) & (boxes[:,3] > 0)
        return {'classes': sorted(set((idx[valid]%classes).tolist())), 'scores': scores[idx], 'boxes': boxes}

    rows = []
    with np.load(args.inputs, allow_pickle=False) as samples, torch.inference_mode():
        if not samples.files:
            raise ValueError('No input tensors')
        for name in samples.files:
            x = samples[name]
            if x.dtype != np.float32 or x.shape != (1,3,640,640) or not np.isfinite(x).all():
                raise ValueError(f'Invalid input: {name}')
            ref = dict(zip([o.name for o in session.get_outputs()],session.run(None,{'images':x})))
            buffers['images'].copy_(torch.from_numpy(x))
            for _ in range(20):
                if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                    raise RuntimeError('TRT execution failed')
            torch.cuda.synchronize()
            start = time.perf_counter()
            if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError('TRT execution failed')
            torch.cuda.synchronize()
            ms = (time.perf_counter()-start)*1000
            got = {n:buffers[n].cpu().numpy().copy() for n in ('pred_logits','pred_boxes')}
            a,b = rank(ref),rank(got)
            rows.append({'id':name, 'onnx_classes':a['classes'], 'trt_classes':b['classes'],
                         'class_set_equal':a['classes']==b['classes'],
                         'top10_score_max_abs':float(np.max(abs(a['scores'][:10]-b['scores'][:10]))),
                         'forward_ms':ms})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'samples':rows,'conf':args.conf,
        'class_set_differences':sum(not r['class_set_equal'] for r in rows),
        'scope':'Input tensors only; no GT recall, localization accuracy or pipeline timing'},indent=2),encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
