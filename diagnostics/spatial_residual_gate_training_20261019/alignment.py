"""Bounded gate-only initialization alignment. No seeding, data I/O or steps."""
import hashlib
import pickle
import random
import numpy as np
import torch

GATE_KEYS = {'spatial_residual_gate_conv.weight', 'spatial_residual_gate_conv.bias'}
METHOD = 'isolated_gate_off_reference_from_same_preconstruction_rng_restore_reference_poststate_v1'

def optimizer_membership(model, optimizer):
    params = [p for group in optimizer.param_groups for p in group['params']]
    ids = [id(p) for p in params]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise AssertionError('Optimizer must contain every parameter exactly once')
    if any(group['lr'] != 3e-4 or group['weight_decay'] != 0.005 for group in optimizer.param_groups):
        raise AssertionError('Unapproved parameter-group hyperparameters')
    names = dict(model.named_parameters())
    if any(ids.count(id(names[k])) != 1 for k in GATE_KEYS):
        raise AssertionError('Each gate parameter must enter optimizer exactly once')
    return {'all_parameters_exactly_once': True, 'gate_parameters_exactly_once': True,
            'parameter_tensors': len(ids), 'groups': len(optimizer.param_groups),
            'learning_rate': 3e-4, 'weight_decay': 0.005, 'optimizer_step_called': False}

def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state().clone(),
            'cuda': [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None}

def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    elif torch.cuda.is_initialized():
        raise RuntimeError('Unexpected CUDA initialization inside bounded construction')

def rng_summary(state):
    sha = lambda b: hashlib.sha256(b).hexdigest()
    return {'python': sha(pickle.dumps(state['python'])), 'numpy': sha(pickle.dumps(state['numpy'])),
            'cpu': sha(state['cpu'].numpy().tobytes()),
            'cuda': [sha(s.cpu().numpy().tobytes()) for s in state['cuda']] if state['cuda'] is not None else None}

def rng_equal(left, right):
    if left['python'] != right['python']:
        return False
    a, b = left['numpy'], right['numpy']
    if a[0] != b[0] or not np.array_equal(a[1], b[1]) or a[2:] != b[2:]:
        return False
    if not torch.equal(left['cpu'], right['cpu']):
        return False
    if (left['cuda'] is None) != (right['cuda'] is None):
        return False
    return left['cuda'] is None or (len(left['cuda']) == len(right['cuda']) and all(torch.equal(a, b) for a, b in zip(left['cuda'], right['cuda'])))

def tensor_summary(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str((str(value.dtype), tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {'tensor_count': len(state), 'numel': sum(t.numel() for t in state.values()), 'sha256': h.hexdigest()}

def build_aligned(factory, model_name, in_channels, num_classes, patch_size, *, generator=None, **kwargs):
    if model_name != 'gscvit_dssr_spatial' or kwargs.get('spatial_residual_gate') is not True:
        raise ValueError('Alignment is exclusively an explicit serial gate=True operation')
    pre = capture_rng()
    generator_before = generator.get_state().clone() if generator is not None else None
    try:
        reference = factory(model_name, in_channels, num_classes, patch_size, **dict(kwargs, spatial_residual_gate=False))
        reference_post = capture_rng()
        reference_state = reference.state_dict()
        restore_rng(pre)
        model = factory(model_name, in_channels, num_classes, patch_size, **kwargs)
        raw_gate_post = capture_rng()
        state = model.state_dict()
        if set(state) - set(reference_state) != GATE_KEYS or set(reference_state) - set(state):
            raise AssertionError('Unexpected shared/new state_dict keys')
        for name, value in reference_state.items():
            other = state[name]
            if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(value, other):
                raise AssertionError('Shared initialization differs: ' + name)
        if sum(state[k].numel() for k in GATE_KEYS) != 130 or any(torch.count_nonzero(state[k]).item() for k in GATE_KEYS):
            raise AssertionError('Gate must have exactly 130 zero-initialized parameters')
        for name in ('python', 'numpy', 'cuda'):
            if rng_summary(pre)[name] != rng_summary(reference_post)[name] or rng_summary(pre)[name] != rng_summary(raw_gate_post)[name]:
                raise AssertionError('Unexpected non-CPU RNG use during CPU construction: ' + name)
        if generator is not None and not torch.equal(generator_before, generator.get_state()):
            raise AssertionError('DataLoader generator was consumed during initialization')
        # Restore the actual reference post-state, never an arbitrary seed.
        restore_rng(reference_post)
        after = capture_rng()
        if not rng_equal(after, reference_post):
            raise AssertionError('Postconstruction RNG restoration failed')
        report = {'method': METHOD, 'shared_state_exact': True,
                  'shared_initial_state': tensor_summary(reference_state),
                  'gate': {k: {'shape': list(state[k].shape), 'numel': state[k].numel(), 'all_zero': True} for k in sorted(GATE_KEYS)},
                  'rng_before': rng_summary(pre), 'reference_rng_after': rng_summary(reference_post),
                  'raw_gate_rng_after': rng_summary(raw_gate_post), 'aligned_rng_after': rng_summary(after),
                  'cpu_rng_exact': True, 'initialized_cuda_rng_exact': True if pre['cuda'] is not None else None,
                  'python_numpy_unchanged': True, 'loader_generator_unchanged': True if generator is not None else None,
                  'trained_checkpoint_loaded': False, 'reference_forward_executed': False}
        del reference, reference_state
        return model, report
    except BaseException:
        restore_rng(pre)
        if generator is not None:
            generator.set_state(generator_before)
        raise
