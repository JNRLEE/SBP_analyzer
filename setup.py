"""
為SBP_analyzer項目設置安裝配置
"""

from setuptools import setup, find_packages

setup(
    name="sbp_analyzer",
    version="0.1.0",
    description="胸腔聲音分析工具套件",
    author="Jenner Lee",
    author_email="jenner.lee.com@gmail.com",
    packages=find_packages(where=".", exclude=["tests*", "sbp_analyzer*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "librosa",
        "torch",
        "scikit-learn",
        "pytest",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
