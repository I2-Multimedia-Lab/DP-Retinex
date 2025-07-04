import importlib
from os import path as osp
from pdb import set_trace as stx

from basicsr.utils import scandir

# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('_arch.py')
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f'basicsr.models.archs.{file_name}')
    for file_name in arch_filenames
]

# stx()

def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class: Instantiated class.
    """

    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)


def define_network(opt):
    """
    根据传入的参数定义网络结构。

    参数:
    opt (dict): 包含网络类型和其他相关配置的字典。

    返回:
    net: 根据opt中的配置动态实例化后的网络对象。
    """
    # 从opt字典中弹出网络类型，这个键值对将不再需要
    network_type = opt.pop('type')
    # 使用dynamic_instantiation函数根据网络类型和剩余的配置选项动态实例化网络
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    # 返回实例化后的网络对象
    return net
