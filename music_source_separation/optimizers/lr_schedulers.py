def get_lr_lambda(step, warm_up_steps, reduce_lr_steps):
    r"""Get lr_lambda for LambdaLR. E.g.,

    .. code-block: python
        lr_lambda = lambda step: get_lr_lambda(step, warm_up_steps=1000, reduce_lr_steps=10000)

        from torch.optim.lr_scheduler import LambdaLR
        LambdaLR(optimizer, lr_lambda)
    """
    if step <= warm_up_steps:
        return step / warm_up_steps
    else:
        return 0.9 ** (step // reduce_lr_steps)