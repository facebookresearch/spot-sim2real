from yacs.config import CfgNode as CN


def construct_config(file_path, opts=None):
    if opts is None:
        opts = []
    config = CN()
    config.set_new_allowed(True)
    config.merge_from_file(file_path)
    config.merge_from_list(opts)

    return config
