from setuptools import setup, find_packages

setup(
    name="meckallm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "docker>=7.0.0",
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
        "rich>=13.7.0",
        "python-dotenv>=1.0.0",
        "pyautogui>=0.9.53",
        "keyboard>=0.13.5",
        "selenium>=4.15.2",
        "yt-dlp>=2023.11.16",
        "psutil>=5.9.6",
        "pywin32>=306",
        "schedule>=1.2.1"
    ],
    python_requires=">=3.8",
    author="MeckaLLM Team",
    description="AI-powered voice control system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
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