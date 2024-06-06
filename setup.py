# coding: utf-8
from setuptools import setup

setup(name='heinsen_attention',
    version='1.0.0',
    description='Reference implementation of "Softmax Attention with Constant Cost per Token" (Heinsen, 2024).',
    url='https://github.com/glassroom/heinsen_attention',
    author='Franz A. Heinsen',
    author_email='franz@glassroom.com',
    license='MIT',
    packages=['heinsen_attention'],
    install_requires='torch',
    zip_safe=False)
