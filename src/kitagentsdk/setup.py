# kitagentsdk/setup.py
from setuptools import setup, find_packages

setup(
    name='kitagentsdk',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'requests',
        'stable-baselines3',
    ],
    entry_points={
        'console_scripts': [
            'kitagentcli=kitagentsdk.cli:main',
        ],
    },
)