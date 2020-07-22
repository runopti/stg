from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='stg',
      version='0.1.2',
      description='feature selection using stochastic gates',
      url='https://github.com/runopti/stg',
      author='Yutaro Yamada',
      author_email='yutaro.yamada@yale.edu',
      long_description=long_description,
      long_description_content_type="text/markdown",
      license='MIT',
      packages=['stg'],
      install_requires=[
          'torch',
          'h5py',
          'six',
          'lifelines'
      ],
      zip_safe=False)