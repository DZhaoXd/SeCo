from datasets import transforms


def build_transform(cfg):
    args = cfg.copy()
    func_name = args.pop('type')
    if not func_name:
        return None
    return transforms.__dict__[func_name](**args)
