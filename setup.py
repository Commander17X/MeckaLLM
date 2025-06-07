from setuptools import setup, find_packages
import os

# Read README.md safely with UTF-8 encoding
def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read()

setup(
    name="meckallm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rich>=10.0.0",
        "psutil>=5.9.0",
        "typing-extensions>=4.0.0",
    ],
    python_requires=">=3.8",
    author="Commander17X",
    author_email="",
    description="MeckaLLM - Advanced Learning and Monitoring System",
    long_description=read_file("README.md") if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/Commander17X/MeckaLLM",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 