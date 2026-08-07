from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantummultiassetportfolio",
    version="0.1.0",
    author="Prasad Shetty",
    author_email="prasadshetty99@gmail.com",
    description="Quantum-Classical Hybrid Portfolio Optimization Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prasadshetty99/quantummultiassetportfolio",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "qiskit>=0.43.0",
        "qiskit-aer>=0.12.0",
        "qiskit-optimization>=0.6.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scipy>=1.11.3",
        "cvxpy>=1.3.2",
        "yfinance>=0.2.32",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.17.0",
        "scikit-learn>=1.3.1",
        "statsmodels>=0.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.6.1",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.7",
            "notebook>=7.0.4",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
