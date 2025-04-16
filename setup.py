from setuptools import setup, find_packages

setup(
    name='sbp_analyzer',
    version='0.1.0',
    description='Analysis and monitoring tools for model training pipelines.',
    author='Your Name', # 請替換成你的名字
    author_email='your.email@example.com', # 請替換成你的 Email
    packages=find_packages(),
    install_requires=[
        # 從 requirements.txt 讀取或直接列出
        'numpy',
        'tensorboard',
        # 'matplotlib',
        # 'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # 或選擇其他授權
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8', # 指定 Python 版本要求
)
