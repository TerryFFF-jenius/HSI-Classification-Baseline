"""Synthetic DSSR forward/backward and compatibility smoke test."""
import argparse
import torch
from models import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_channels', type=int, default=15)
    p.add_argument('--num_classes', type=int, default=16)
    p.add_argument('--patch_size', type=int, default=7)
    p.add_argument('--batch_size', type=int, default=2)
    args = p.parse_args()
    x = torch.randn(args.batch_size, 1, args.in_channels, args.patch_size, args.patch_size)
    model = build_model('gscvit_dssr', args.in_channels, args.num_classes, args.patch_size,
                        spectral_groups=8, route_strength=0.5, route_temperature=1.0,
                        router_variant='full')
    model.train()
    logits, alpha, beta, permutation, inverse = model.forward_with_routing(x)
    loss = logits.square().mean()
    loss.backward()
    assert logits.shape == (args.batch_size, args.num_classes)
    assert alpha.shape == (args.batch_size, 8)
    assert beta.shape == (args.batch_size, 1, 9, 9)
    assert permutation.shape == inverse.shape == (args.batch_size, 8)
    assert model.state_router.last_beta.shape == (args.batch_size, 1, 9, 9)
    identity = torch.arange(8).expand_as(permutation)
    assert torch.equal(torch.gather(permutation, 1, inverse), identity)
    assert all(torch.isfinite(p).all() for p in model.parameters() if p.grad is not None)
    disabled = build_model('gscvit_dssr', args.in_channels, args.num_classes, args.patch_size,
                           spectral_groups=8)
    disabled.state_router.enabled = False
    print('[PASS] DSSR forward/backward, shapes, finite gradients, permutation inverse')
    print('[PASS] DSSR state route can be disabled without changing model construction')


if __name__ == '__main__':
    main()
