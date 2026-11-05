"""DEIMv2 fixed-640 export, layout repair and TensorRT mixed-precision build."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def new_file(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_names(value):
    """Validate an ordered list of class labels from CLI or model metadata."""
    labels = [name.strip() for name in value.split(',')] if isinstance(value, str) else value
    if (not isinstance(labels, list) or not labels
            or not all(isinstance(name, str) and name.strip() for name in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('Class names must be nonempty and unique')
    return list(labels)


def resolve_names(explicit=None, stored=None):
    requested = parse_names(explicit) if explicit is not None else None
    saved = parse_names(stored) if stored is not None else None
    if requested is not None and saved is not None and requested != saved:
        raise ValueError('Explicit class names/order conflict with saved model metadata')
    return requested if requested is not None else saved


def export_options(data=None, names=None):
    """Configure class count without opening a dataset when names are supplied."""
    if data is not None and names is not None:
        raise ValueError('Use either --data or --names, not both')
    if names is not None:
        labels = parse_names(names)
        return {'yolo_data': None, 'num_classes': len(labels),
                'remap_mscoco_category': False}
    return {'yolo_data': str(Path(data).resolve())} if data else {}


def export(args):
    import torch
    from engine.core import YAMLConfig

    output = new_file(args.output)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    names = resolve_names(args.names, state.get('names'))
    if args.data:
        from engine.data.dataset.yolo_dataset import read_yolo_metadata
        _, data_names, _ = read_yolo_metadata(args.data)
        names = resolve_names(data_names, names)
    options = export_options(names=names) if names is not None else export_options(data=args.data)
    if state.get('num_image_classes'):
        # Multi-task checkpoint: size the image-classification head like the trained one.
        options['num_image_classes'] = int(state['num_image_classes'])
    cfg = YAMLConfig(str(Path(args.config).resolve()), **options)
    if names is None and cfg.yaml_cfg.get('yolo_data'):
        from engine.data.dataset.yolo_dataset import read_yolo_metadata
        _, names, _ = read_yolo_metadata(cfg.yaml_cfg['yolo_data'])
        names = parse_names(names)
    if names is not None and 'num_classes' in state and state['num_classes'] != len(names):
        raise ValueError('Checkpoint num_classes does not match names')
    cfg.yaml_cfg['DINOv3STAs']['weights_path'] = None
    model = cfg.model
    weights = state['ema']['module'] if 'ema' in state else state['model']
    model.load_state_dict(weights, strict=True)
    model.to(args.device).eval()

    class Export(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.m = module

        def forward(self, images):
            values = self.m(images)
            return values['pred_logits'], values['pred_boxes']

    with torch.inference_mode():
        torch.onnx.export(Export(model).eval(), torch.zeros(1,3,640,640,device=args.device), str(output),
                          input_names=['images'], output_names=['pred_logits','pred_boxes'],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    import onnx
    graph = onnx.load(output)
    properties = {}
    if names is not None:
        properties.update(names=json.dumps(names, ensure_ascii=False), num_classes=str(len(names)))
    if state.get('model_name') is not None:
        if not isinstance(state['model_name'], str) or not state['model_name'].strip():
            raise ValueError('Checkpoint model_name must be a nonempty string')
        properties['model_name'] = state['model_name']
    onnx.helper.set_model_props(graph, properties)
    onnx.checker.check_model(graph)
    onnx.save(graph, output)
    print(output)


def patch(args):
    import onnx
    import numpy as np
    import onnxruntime as ort
    from layout import rewrite_layout

    output = new_file(args.output)
    original = onnx.load(args.onnx)
    fixed = rewrite_layout(original)
    # Validate unchanged computation, with exactly the same input and query order.
    options = ort.SessionOptions()
    before = ort.InferenceSession(original.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    after = ort.InferenceSession(fixed.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as archive:
            inputs = [(name, archive[name]) for name in archive.files]
    else:
        inputs = [('seed42_random', np.random.default_rng(42).normal(size=(1,3,640,640)).astype(np.float32))]
    if not inputs:
        raise ValueError('Empty input archive')
    errors = {}
    for name, x in inputs:
        if x.shape != (1,3,640,640) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f'Invalid normalized input tensor: {name}')
        a, b = before.run(None, {'images': x}), after.run(None, {'images': x})
        for av, bv in zip(a, b):
            np.testing.assert_allclose(av, bv, rtol=1e-5, atol=1e-5)
        errors[name] = [float(np.max(np.abs(av-bv))) for av,bv in zip(a,b)]
    del before, after, original
    import gc
    gc.collect()
    onnx.save(fixed, output)
    output.with_suffix('.verification.json').write_text(json.dumps({
        'source_sha256': digest(args.onnx), 'output_sha256': digest(output),
        'raw_output_max_abs_errors': errors, 'business_accuracy_validated': False,
    }, indent=2), encoding='utf-8')
    print(output)


def build(args):
    import torch
    import tensorrt as trt
    import onnx
    from layout import SLICES

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new package directory')
    model = onnx.load(args.onnx)
    nodes = {n.name: n for n in model.graph.node}
    if any(n in nodes for n in SLICES) or any(f'layout_fix_{i}_SliceBeforeTranspose' not in nodes for i in range(3)):
        raise ValueError('Run patch first; original or unknown layout rejected')
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if dims != [1,3,640,640] or {o.name for o in model.graph.output} != {'pred_logits','pred_boxes'}:
        raise ValueError('Unsupported engine contract')
    props = {item.key: item.value for item in model.metadata_props}
    stored = json.loads(props['names']) if 'names' in props else None
    names = resolve_names(args.names, stored)
    if names is None:
        raise ValueError('ONNX has no class names; supply --names in training class order')
    logits = next(o for o in model.graph.output if o.name == 'pred_logits')
    classes = logits.type.tensor_type.shape.dim[-1].dim_value
    if classes != len(names):
        raise ValueError(f'Class count mismatch: ONNX={classes}, names={len(names)}')
    torch.cuda.set_device(args.device)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(Path(args.onnx).resolve())):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.clear_flag(trt.BuilderFlag.INT8)
    protected = []
    if args.precision == 'fp16mix':
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.TF32)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.REDUCE,trt.LayerType.NORMALIZATION,trt.LayerType.SOFTMAX) or (
                layer.type == trt.LayerType.ELEMENTWISE and '/Pow' in layer.name
            ):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    if layer.get_output(j).dtype in (trt.float32,trt.float16):
                        layer.set_output_type(j,trt.float32)
                protected.append(layer.name)
    else:
        config.clear_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.TF32)
    start = time.perf_counter()
    blob = builder.build_serialized_network(network,config)
    if blob is None:
        raise RuntimeError('TensorRT build failed')
    output.mkdir(parents=True, exist_ok=False)
    engine = output / 'model.engine'
    engine.write_bytes(bytes(blob))
    metadata = {'backend':'deim_trt', 'names':names, 'aliases':{}, 'input_shape':[1,3,640,640],
                'engine':engine.name, 'sha256':digest(engine), 'onnx_sha256':digest(args.onnx),
                'gpu':torch.cuda.get_device_name(args.device), 'tensorrt_version':trt.__version__,
                'precision':args.precision, 'tf32':config.get_flag(trt.BuilderFlag.TF32),
                'build_seconds':time.perf_counter()-start, 'validation':'pending_on_target_machine'}
    if props.get('model_name'):
        metadata['model_name'] = props['model_name']
    (output/'model.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (output/'fp32_protected_layers.json').write_text(json.dumps(protected,indent=2),encoding='utf-8')
    print(output)


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('export')
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    classes = p.add_mutually_exclusive_group()
    classes.add_argument('--data', help='YOLO dataset YAML used to determine class count')
    classes.add_argument('--names', help='Ordered comma-separated classes; bypasses dataset YAML')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',required=True)
    p.set_defaults(run=export)
    p = commands.add_parser('patch')
    p.add_argument('--onnx',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--inputs',help='NPZ of preprocessed float32 [1,3,640,640] tensors')
    p.set_defaults(run=patch)
    p = commands.add_parser('build')
    p.add_argument('--onnx',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--precision',choices=['fp32','fp16mix'],default='fp16mix')
    p.add_argument('--names',help='Ordered classes; defaults to ONNX metadata')
    p.add_argument('--device',type=int,default=0)
    p.set_defaults(run=build)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
