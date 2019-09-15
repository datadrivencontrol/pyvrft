from setuptools import setup, Extension

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name='pyvrft',
    version='1.1',
    description='Virtual Reference Feedback Tuning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['vrft'],
    install_requires=['numpy','scipy','matplotlib'],
    author='Diego Eckhard',
    author_email='diego@eckhard.com.br',
    url='http://github.com/datadrivencontrol/pyvrft',
    license='MIT',
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
