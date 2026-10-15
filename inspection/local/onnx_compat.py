"""Load the opset-17 models on an ONNX Runtime that stops at opset 16.

The Jetson Nano's GPU build of ONNX Runtime (1.11, JetPack 4.6) reads opset 16 at
most. Opset 17 added only new operators, and of those the models use one,
LayerNormalization (SegFormer). Here each LayerNormalization is written out as
the arithmetic it is defined by, the form PyTorch itself exports below opset 17,
and the opset is set to 16. The files on disk are not changed.
"""
MAX_OPSET_BY_RUNTIME = ((1, 12), 17)   # ONNX Runtime 1.12 was the first with opset 17


def runtime_max_opset(version):
    """Highest ai.onnx opset an ONNX Runtime version loads (16 for anything before 1.12)."""
    parts = tuple(int(p) for p in version.split('+')[0].split('.')[:2])
    return 17 if parts >= MAX_OPSET_BY_RUNTIME[0] else 16


def runtime_needs_lowering():
    """True on an ONNX Runtime too old for the opset-17 models (the Jetson Nano's 1.11)."""
    import onnxruntime as ort
    return runtime_max_opset(ort.__version__) < 17


def model_opset(model):
    return next((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), 0)


def _attribute(node, name, default):
    from onnx import helper
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def lower_layer_norm(model):
    """Replace every LayerNormalization in place; returns how many were replaced."""
    from onnx import helper, TensorProto
    graph = model.graph
    nodes, replaced = [], 0
    for node in graph.node:
        if node.op_type != 'LayerNormalization' or node.domain not in ('', 'ai.onnx'):
            nodes.append(node)
            continue
        axis = _attribute(node, 'axis', -1)
        epsilon = float(_attribute(node, 'epsilon', 1e-5))
        if axis >= 0:
            raise ValueError(f'LayerNormalization {node.name}: only a negative axis is supported')
        if len(node.output) > 1 and any(node.output[1:]):
            raise ValueError(f'LayerNormalization {node.name}: mean/inv-std outputs are not supported')
        axes = list(range(axis, 0))
        x, scale = node.input[0], node.input[1]
        bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
        p = f'{node.name or "layernorm"}_{replaced}_'
        out = node.output[0]
        nodes += [
            helper.make_node('ReduceMean', [x], [p + 'mean'], axes=axes, keepdims=1),
            helper.make_node('Sub', [x, p + 'mean'], [p + 'centred']),
            helper.make_node('Mul', [p + 'centred', p + 'centred'], [p + 'squared']),
            helper.make_node('ReduceMean', [p + 'squared'], [p + 'variance'], axes=axes, keepdims=1),
            helper.make_node('Constant', [], [p + 'epsilon'],
                             value=helper.make_tensor(p + 'epsilon_value', TensorProto.FLOAT, [], [epsilon])),
            helper.make_node('Add', [p + 'variance', p + 'epsilon'], [p + 'variance_eps']),
            helper.make_node('Sqrt', [p + 'variance_eps'], [p + 'std']),
            helper.make_node('Div', [p + 'centred', p + 'std'], [p + 'normalised']),
            helper.make_node('Mul', [p + 'normalised', scale], [p + 'scaled'] if bias else [out]),
        ]
        if bias:
            nodes.append(helper.make_node('Add', [p + 'scaled', bias], [out]))
        replaced += 1
    del graph.node[:]
    graph.node.extend(nodes)
    return replaced


def compatible_model(path, runtime_version):
    """Bytes of the model rewritten for this runtime, or None when it loads the file as is."""
    import onnx
    model = onnx.load(str(path))
    limit = runtime_max_opset(runtime_version)
    if model_opset(model) <= limit:
        return None
    lower_layer_norm(model)
    left = sorted({n.op_type for n in model.graph.node if n.op_type == 'LayerNormalization'})
    if left:
        raise ValueError(f'{path}: cannot lower {left} to opset {limit}')
    for opset in model.opset_import:
        if opset.domain in ('', 'ai.onnx'):
            opset.version = limit
    onnx.checker.check_model(model)
    return model.SerializeToString()
