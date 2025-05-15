from setuptools import setup

#

setup(
   name='CFApi',
   version='0.2',
   description='Generate Counterfactuals',
   author='Finn Schwall',
   author_email='finn.schwall@iosb.fraunhofer.de',
   packages=['CFApi'],
   install_requires=['numpy', 'pandas', 'scipy', "scikit-learn"]
)