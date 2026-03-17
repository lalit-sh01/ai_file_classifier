from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="file-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered file organization using Claude API or Ollama",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/file-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: System :: Filesystems",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anthropic>=0.20.0",
        "requests>=2.28.0",
        "PyMuPDF>=1.23.0",
        "python-docx>=0.8.11",
        "openpyxl>=3.1.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "file-classifier=file_classifier.cli:main",
            "fclass=file_classifier.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
