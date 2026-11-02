"""Equivalent fixed-640 DEIMv2 value-layout rewrite, without changing weights."""
import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper


SLICES = ['/m/decoder/decoder/Slice', '/m/decoder/decoder/Slice_1', '/m/decoder/decoder/Slice_2']
MEMORY = '/m/decoder/Concat_3_output_0'


def rewrite_layout(model):
    """Return a new graph; reject exports outside the verified X/640 contract."""
    model = copy.deepcopy(model)
    if len(model.graph.input) != 1:
        raise ValueError('Expected one fixed input')
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if model.graph.input[0].name != 'images' or shape != [1, 3, 640, 640]:
        raise ValueError('Only images=[1,3,640,640] is supported')
    if {o.name for o in model.graph.output} != {'pred_logits', 'pred_boxes'}:
        raise ValueError('Expected raw pred_logits and pred_boxes outputs')
    nodes = {n.name: n for n in model.graph.node}
    if any(name not in nodes or nodes[name].op_type != 'Slice' for name in SLICES):
        raise ValueError('Expected three original value Slice nodes; already patched or incompatible graph')
    producers = {out: node for node in model.graph.node for out in node.output}
    # The exporter leaves shape arithmetic dynamic. Verify it by CPU execution
    # on the fixed input instead of guessing dimensions from node names.
    import onnxruntime as ort
    probes = [MEMORY, '/m/decoder/decoder/Reshape_output_0']
    for name in SLICES:
        probes.extend(nodes[name].input[1:])
    probes = list(dict.fromkeys(probes))
    probe = copy.deepcopy(model)
    del probe.graph.output[:]
    for name in probes:
        value = onnx.ValueInfoProto()
        value.name = name
        probe.graph.output.append(value)
    session = ort.InferenceSession(probe.SerializeToString(), providers=['CPUExecutionProvider'])
    measured = dict(zip(probes, session.run(probes, {'images': np.zeros(shape, np.float32)})))

    def evaluate(name):
        return measured[name]

    for i, name in enumerate(SLICES):
        node = nodes[name]
        if node.input[0] != '/m/decoder/decoder/Transpose_output_0':
            raise ValueError('Unexpected Slice source')
        expected = ([0, 6400, 8000][i], [6400, 8000, 8400][i], 3)
        actual = tuple(int(evaluate(v).item()) for v in node.input[1:4])
        actual = (actual[0], actual[1], actual[2] % 4)
        if actual != expected or (len(node.input) > 4 and int(evaluate(node.input[4]).item()) != 1):
            raise ValueError(f'Unexpected Slice range: {actual}')
    transpose = producers['/m/decoder/decoder/Transpose_output_0']
    perm = next((list(a.ints) for a in transpose.attribute if a.name == 'perm'), [])
    reshape = producers[transpose.input[0]]
    # Infer the memory/reshape dimensions using ONNX shape inference where available.
    if perm != [0, 2, 3, 1] or reshape.op_type != 'Reshape' or reshape.input[0] != MEMORY:
        raise ValueError('Unexpected value reshape/transpose path')
    for name, expected_shape in [(MEMORY, [1,8400,256]), (reshape.output[0], [1,8400,8,32])]:
        dims = list(measured[name].shape)
        if dims != expected_shape:
            raise ValueError(f'Unsupported shape for {name}: {dims}')
    if any(v.name.startswith('layout_fix_') for v in model.graph.initializer):
        raise ValueError('Layout rewrite name collision')
    rebuilt = []
    for node in model.graph.node:
        if node.name not in SLICES:
            rebuilt.append(node)
            continue
        i = SLICES.index(node.name)
        start, end = [0,6400,8000][i], [6400,8000,8400][i]
        prefix = f'layout_fix_{i}'
        for suffix, data in [('start',[start]), ('end',[end]), ('axis',[1]), ('shape',[1,end-start,8,32])]:
            model.graph.initializer.append(numpy_helper.from_array(np.array(data, dtype=np.int64), prefix+'_'+suffix))
        rebuilt.extend([
            helper.make_node('Slice', [MEMORY,prefix+'_start',prefix+'_end',prefix+'_axis'], [prefix+'_sliced'], name=prefix+'_SliceBeforeTranspose'),
            helper.make_node('Reshape', [prefix+'_sliced',prefix+'_shape'], [prefix+'_reshaped'], name=prefix+'_Reshape'),
            helper.make_node('Transpose', [prefix+'_reshaped'], list(node.output), perm=[0,2,3,1], name=prefix+'_Transpose'),
        ])
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    onnx.checker.check_model(model)
    return model
