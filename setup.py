from setuptools import setup, find_packages

with open('README.md','r') as f:
    description = f.read()

setup(
    name='fusemap',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'scanpy==1.9.3',
        'torch==2.0.1',
        'dgl==1.1.1',
        'sparse==0.14.0',
        'leidenalg==0.10.1',
        'dglgo==0.0.2'
    ],
    entry_points={
        "console_scripts":[
            "fusemap = fusemap:spatial_integrate"
        ]
    },
    long_description=description,
    long_description_content_type='text/markdown',

)