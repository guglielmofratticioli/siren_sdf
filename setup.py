from setuptools import setup, find_packages

setup(
    name='siren_sdf',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'Pillow',
        'scikit-image',
        'trimesh',
        'pyrender',
        'torchvision',
        'mesh-to-sdf'
    ],
    entry_points={
        'console_scripts': [
            'siren_sdf=siren_sdf:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A description of your project',
    url='https://github.com/guglielmofratticioli/siren_sdf.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)