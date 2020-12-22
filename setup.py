# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Install fedjax."""

from setuptools import find_namespace_packages
from setuptools import setup

__version__ = None

with open('fedjax/version.py') as f:
  exec(f.read(), globals())

setup(
    name='fedjax',
    version=__version__,
    url='https://github.com/google/fedjax',
    license='Apache 2.0',
    author='FedJAX Team',
    author_email='no-reply@google.com',
    description=('Federated learning with JAX.'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords='federated python machine learning',
    packages=find_namespace_packages(exclude=['*_test.py']),
    install_requires=[
        'dm-haiku',
        'frozendict',
        'jax',
        'jaxlib',
        'optax',
        'tensorflow-federated',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
