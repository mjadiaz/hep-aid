from setuptools import setup

setup(
    name='hepaid',
    version='2.0',
    description='Modules for pythonic interaction with hep tools',
    author='Mauricio A. Diaz',
    author_email='mauricio.jadiaz@gmail.com',
    packages=['hepaid'],
    install_requires=['numpy'],
    include_package_data=True
)
