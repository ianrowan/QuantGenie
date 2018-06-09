from setuptools import setup

setup(
    name='QuantGenie',
    version='0.9',
    packages=['Quant'],
    license='MIT',
    install_requires = ['tkinter','numpy', 'sklearn', 'tensorflow-gpu', 'pandas', 'tqdm','matplotlib', 'pandas_datareader']
)

