import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

python_versions = '>=3.6, <3.9'  # probably other versions work as well

requirements_default = [
    'numpy',  # for all datastructures
    'matplotlib',  # vis
    'numba',  # speedup for numpy-quaternion
    'numpy-quaternion',  # numpy integration for quaternions
    'tqdm',  # progress bars
    'pybullet',  # for the simulation module
    'attrdict',
    'collections',
    'joblib',
    'gc'
]

setuptools.setup(
    name='GPNet-simulator',
    version='0.1',
    python_requires=python_versions,
    install_requires=requirements_default,
    packages=setuptools.find_packages(),
    url='',
    license='',
    author='GPNet authors: Chaozheng Wu, Jian Chen, Qiaoyu Cao, Jianchi Zhang, Yunxin Tai, Lin Sun, and Kui Jia;' +
           'package author: Martin Rudorfer',
    author_email='m.rudorfer@bham.ac.uk',
    description='GPNet simulator for robotic grapsing',
    long_description=long_description
)
