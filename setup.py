"""
setup.py for Interview Trainer Agent
Allows 'pip install -e .' for local development in IBM Watson Studio.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="interview-trainer-agent",
    version="1.0.0",
    description="RAG-powered Interview Trainer Agent using IBM Granite on IBM Cloud",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Interview Trainer Agent",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["interview", "RAG", "IBM Granite", "watsonx", "AI", "NLP"],
)
