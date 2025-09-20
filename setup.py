from setuptools import setup, find_packages

setup(
    name="miladlearn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0",
        "numpy>=1.20"
    ],
    author="Milad",
    description="Custom ML algorithms with feature ratio support (MiladClassifier)",
    url="https://github.com/YourUsername/miladlearn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
