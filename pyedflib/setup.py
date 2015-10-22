#!/usr/bin/env python
from __future__ import division, print_function, absolute_import


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy as np

    config = Configuration('pyedflib', parent_package, top_path)

    config.add_data_dir('tests')

    sources = ["_pyedflib", "edflib"]
    headers = ["edflib"]

    # add main PyEDFlib module
    config.add_extension(
        '_pyedflib',
        sources=["src/{0}.c".format(s) for s in sources],
        depends=["src/{0}.h".format(s) for s in headers],
        include_dirs=["src", np.get_include()],
        define_macros=[("PY_EXTENSION", None)],
    )

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
