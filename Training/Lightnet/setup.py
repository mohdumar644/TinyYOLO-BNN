import setuptools as setup
from pkg_resources import get_distribution, DistributionNotFound


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

def find_packages():
    return ['lightnet'] + ['lightnet.'+p for p in setup.find_packages('lightnet')]

def get_version():
    with open('VERSION', 'r') as f:
        version = f.read().splitlines()[0]
    with open('lightnet/version.py', 'w') as f:
        f.write('#\n')
        f.write('#   Lightnet version: Automatically generated version file\n')
        f.write('#\n\n')
        f.write(f'__version__ = "{version}"\n')

    return version


requirements = [
    'numpy',
    'torchvision',
    'brambox',
]
pillow_req = 'pillow-simd' if get_dist('pillow-simd') is not None else 'pillow'
requirements.append(pillow_req)

setup.setup(
    name='lightnet',
    version=get_version(),
    author='EAVISE',
    description='Building blocks for recreating darknet networks in pytorch',
    long_description=open('README.md').read(),
    packages=find_packages(),
    test_suite='test',
    install_requires=requirements,
    extras_require={
        'visual': ['visdom']
    },
)
