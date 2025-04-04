from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='CellSegmentationEvaluator',
    version='1.5.15',    
    description='Functions for reference-free evaluation of the quality of cell segmentations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/murphylab/CellSegmentationEvaluator/',
    author='Haoran Chen and Ce Zhang and Robert F. Murphy',
    author_email='murphy@cmu.edu',
    license='MIT',
    packages=['CellSegmentationEvaluator'],
    install_requires=['numpy', 'xmltodict', 'pandas', 'scipy', 'scikit-image', 'scikit-learn', 'tifffile', 'aicsimageio'],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
)
