from setuptools import find_packages, setup

setup(
    name='bytesep',
    version='0.1.1',
    description='Music source separation',
    author='ByteDance',
    url="https://github.com/bytedance/music_source_separation",
    license='Apache 2.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch==1.7.1',
        'librosa==0.8.0',
        'museval==0.4.0',
        'h5py==2.10.0',
        'pytorch_lightning==1.2.1',
        'numpy==1.18.5',
        'torchlibrosa==0.0.9',
        'matplotlib==3.3.4',
        'musdb==0.4.0',
        'museval==0.4.0',
        'samplerate==0.1.0'
    ],
    zip_safe=False
)
