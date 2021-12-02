from setuptools import find_packages, setup

	
print("""
*******************************************************************
MIT License

Copyright (c) 2021 Haoran Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************
""")

setup(
	name="CellSegmentationEvaluator",
	version='1.0',
	description='Evaluation of cell segmentation methods without reference segmentations',
	url='https://github.com/murphygroup/CellSegmentationEvaluator',
	author='Haoran Chen, Robert F. Murphy',
	author_email='hrchen@cmu.edu, murphy@andrew.cmu.edu',
	classifiers=[
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 3 :: Only',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8'
	],
	license='MIT',
	packages=find_packages(),
	python_requires=">=3.7",
	install_requires=[
		"matplotlib",
		"numpy",
		"pandas",
		"scikit-image==0.16.2",
		"scikit-learn",
		"tifffile",
		"opencv-python",
	],
	zip_safe=False,
)
