from setuptools import setup, find_packages

# To read the requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='spelltest',
    version='0.0.1',
    packages=find_packages(),
    install_requires=required,  # Using the read requirements
    entry_points={
        'console_scripts': [
            'spelltest = spelltest.cli:main',
        ],
    },
    classifiers=[
        # Classifiers to indicate the status and intended audience
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='testing, large language models, synthetic users',
    python_requires='==3.10.*',
    project_urls={
        'Bug Reports': 'https://github.com/artas728/spelltest/issues',
        'Source': 'https://github.com/artas728/spelltest/',
    },
)