from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]
# 显式导入每个模块
for module_name in __all__:
    __import__(f"{__name__}.{module_name}")