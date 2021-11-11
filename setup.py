from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="audioseg",
    version="0.1.0",
    author="Krishanthan",
    description="Python Audio Segmentation ",
    long_description=readme,
    url="https://github.com/KrishRN/audioseg",
    packages=find_packages(exclude=("tests",)),
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.8',
    install_requires=[
        'wheel',
        'pandas', 
        'scipy', 
        'tables',
        'python_speech_features',
        'tensorflow',
    ],
)