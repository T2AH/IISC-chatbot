"""
Setup script for the IISc Research Chatbot package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="iisc-research-chatbot",
    version="1.0.0",
    author="IISc Research Team",
    author_email="research@iisc.ac.in",
    description="AI-powered research and academic chatbot for IISc with RAG capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/iisc-research-chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "scrapy>=2.11.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "requests>=2.31.0",
        "spacy>=3.7.0",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "keybert>=0.8.0",
        "sentence-transformers>=2.2.0",
        "nltk>=3.8.0",
        "neo4j>=5.14.0",
        "chromadb>=0.4.18",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-openai>=0.0.5",
        "openai>=1.6.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "pyyaml>=6.0.0",
        "loguru>=0.7.0",
        "tqdm>=4.66.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iisc-chatbot=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
