"""Convert Inkling's GGUF projector to the native Strata vision shard layout."""
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import uuid


def tensor_name(name):
    fixed = {
        'v.class_embd': 'backbone.embeddings.class_embedding',
        'v.class_position_embd.weight': 'backbone.embeddings.cls_position_embedding',
        'v.patch_embd.weight': 'backbone.embeddings.patch_embedding.weight',
        'v.position_embd.weight': 'backbone.embeddings.position_embedding',
        'v.pre_ln.weight': 'backbone.ln_pre.weight',
        'v.pre_ln.bias': 'backbone.ln_pre.bias',
        'mm.1.weight': 'proj_in.weight', 'mm.2.weight': 'proj_out.weight',
    }
    if name in fixed:
        return 'model.visual.' + fixed[name]
    match = re.fullmatch(r'v.blk.(\d+).(.+)\.(weight|bias)', name)
    if not match:
        raise ValueError('未支持的 Inkling 视觉张量：' + name)
    layer, kind, suffix = match.groups()
    mapping = {'ln1': 'ln_1', 'ln2': 'ln_2', 'ls1': 'ls_1', 'ls2': 'ls_2',
               'ffn_up': 'mlp.c_fc', 'ffn_down': 'mlp.c_proj',
               'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj',
               'attn_v': 'self_attn.v_proj', 'attn_out': 'self_attn.out_proj'}
    if kind not in mapping:
        raise ValueError('未支持的 Inkling 视觉层：' + kind)
    if kind in ('ls1', 'ls2'):
        suffix = 'gamma'
    return f'model.visual.backbone.layers.{layer}.{mapping[kind]}.{suffix}'


def convert_projector(source, folder):
    import numpy as np
    try:
        import zstandard as zstd
        import gguf
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'gguf==0.19.0', 'zstandard==0.25.0'], check=True)
        import zstandard as zstd
    from gguf import GGUFReader
    from gguf.quants import dequantize

    source, folder = Path(source), Path(folder)
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    folder = folder / 'vision-conversions' / digest
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'Inkling-Vision.strata'
    if target.is_file() and (folder / 'complete.json').is_file():
        try:
            record = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
            with target.open('rb') as stream:
                if record.get('output_sha256') == hashlib.file_digest(stream, 'sha256').hexdigest():
                    return target
        except (OSError, ValueError):
            pass
    reader = GGUFReader(str(source))
    def field(key):
        value = reader.fields.get('clip.vision.' + key)
        if value is None:
            raise ValueError('视觉模型缺少元数据：' + key)
        return value.contents()
    if field('projector_type') != 'inkling2v':
        raise ValueError('此转换器只支持 Inkling 2 视觉投影模型。')
    config = dict(hidden_size=int(field('embedding_length')),
                  num_hidden_layers=int(field('block_count')),
                  num_attention_heads=int(field('attention.head_count')),
                  native_patch_size=int(field('patch_size')),
                  adapter_merge_size=int(field('spatial_merge_size')),
                  text_hidden_size=int(field('projection_dim')),
                  projection_hidden_dim=int(field('feed_forward_length')),
                  posemb_grid_size=int(field('image_size')) // int(field('patch_size')),
                  layer_norm_eps=float(field('attention.layer_norm_epsilon')),
                  mlp_ratio=int(field('feed_forward_length')) / int(field('embedding_length')))
    partial = target.with_suffix('.' + uuid.uuid4().hex + '.partial')
    tensors, entries = [], {}
    with partial.open('wb') as output:
        output.write(bytes(4096))
        section_start = output.tell()
        section_hash = hashlib.sha256()
        for tensor in reader.tensors:
            name = tensor_name(tensor.name)
            shape = list(reversed(list(map(int, tensor.shape))))
            if tensor.name == 'v.patch_embd.weight':
                # GGUF convolution is [out, channels, height, width]. The native
                # runner consumes flattened [out, height, width, channels].
                values = dequantize(tensor.data, tensor.tensor_type).reshape(shape)
                values = np.ascontiguousarray(values.transpose(0, 2, 3, 1)).reshape(shape[0], -1)
                shape = list(values.shape)
            else:
                values = dequantize(tensor.data, tensor.tensor_type).reshape(shape)
            raw = np.ascontiguousarray(values, dtype='<f4').tobytes()
            tensors.append(dict(name=name, dtype='float32', shape=shape,
                                offset=output.tell() - section_start, bytes=len(raw)))
            output.write(raw)
            section_hash.update(raw)
        entries[1] = (section_start, output.tell() - section_start, section_hash.digest())
        output.write(bytes((-output.tell()) % 4096))
        manifest = dict(format='strata-v1.1-vision', arch='inkling2_vision',
                        vision_config=config, tensors=tensors)
        raw = zstd.ZstdCompressor(level=3).compress(json.dumps(manifest).encode())
        entries[0] = (output.tell(), len(raw), hashlib.sha256(raw).digest())
        output.write(raw)
        output.write(bytes((-output.tell()) % 4096))
        header = bytearray(4096)
        header[:8] = b'STRATA\x00\x01'
        struct.pack_into('<IHH', header, 8, 1, 3, 6)
        for index in range(6):
            offset, size, sha = entries.get(index, (0, 0, hashlib.sha256(b'').digest()))
            struct.pack_into('<QQ32sIIQ', header, 16 + index * 64, offset, size, sha, index, 0, 0)
        output.seek(0)
        output.write(header)
    os.replace(partial, target)
    with target.open('rb') as stream:
        output_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    record = folder / (uuid.uuid4().hex + '.json.tmp')
    record.write_text(json.dumps({'source_sha256': digest, 'output_sha256': output_digest, 'format': 1}), encoding='utf-8')
    os.replace(record, folder / 'complete.json')
    return target
