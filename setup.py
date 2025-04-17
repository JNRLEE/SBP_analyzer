"""
# 專案安裝腳本
此文件定義了項目的安裝配置，包括依賴、版本號及其他相關信息。
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
    # 移除注釋和空行
    requirements = [req for req in requirements if req and not req.startswith("#")]

setup(
    name="sbp_analyzer",
    version="0.1.0",
    author="MicforDysphagia Team",
    author_email="contact@example.com",
    description="SBP Analyzer for analyzing Subglottal Pressure signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sbp_analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.1.0",
            "pylint>=2.13.0",
            "black>=22.3.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
)
