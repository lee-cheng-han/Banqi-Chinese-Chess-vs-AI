#!/usr/bin/env python3
"""
Setup script for Banqi AI Game
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name='banqi-ai',
    version='1.0.0',
    description='A Python implementation of Banqi (Chinese Dark Chess) with AI opponent',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/banqi-ai',
    py_modules=['banqi'],
    python_requires='>=3.7',
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Games/Entertainment :: Board Games',
        'Topic :: Games/Entertainment :: Turn Based Strategy',
        'Operating System :: OS Independent',
    ],
    keywords='banqi chinese chess dark chess board game ai minimax strategy',
    entry_points={
        'console_scripts': [
            'banqi=banqi:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/banqi-ai/issues',
        'Source': 'https://github.com/yourusername/banqi-ai',
        'Documentation': 'https://github.com/yourusername/banqi-ai#readme',
    },
)
