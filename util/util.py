import torch
from torch.optim import SGD, Adam, AdamW

def make_optimizer(params, args, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW
    }[args.optim]
    if args.optim == 'sgd':
        optimizer_spec = {
            'args': dict(lr=args.lr_start, weight_decay=args.weight_decay),
            'sd': None
        }
    elif args.optim == 'adam':
        optimizer_spec = {
            'args': dict(lr=args.lr_start, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay),
            'sd': None
        }
    elif args.optim == 'adamw':
        optimizer_spec = {
            'args': dict(lr=args.lr_start, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay),
            'sd': None
        }
    optimizer = Optimizer(params, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

def prepare_training(args, model):
    optimizer = make_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    if args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch_cycle, eta_min=args.lr_min)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    elif args.lr_scheduler == 'poly':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epo: args.lr_decay ** ((epo - 1) / 10 + 1))
    elif args.lr_scheduler == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay) 
    elif args.lr_scheduler == 'reduce': 
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay) 
    elif args.lr_scheduler == 'cosinewarm':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.lr_min)
    elif args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
    return optimizer, lr_scheduler
