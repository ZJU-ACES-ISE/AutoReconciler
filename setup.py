from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'gplearn',
        'sympy'
    ],
    entry_points={
        'console_scripts': [
            'my_script=my_package.my_module:main'
        ]
    },
    author='Your Name',
    author_email='your@email.com',
    description='My package description',
    keywords='my package keywords',
    url='https://github.com/yourusername/my_package',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ]
)