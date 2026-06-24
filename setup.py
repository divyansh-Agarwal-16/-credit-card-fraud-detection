"""
setup.py — makes fraud-ml-pipeline installable as a package.
Install in dev mode:  pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fraud-ml-pipeline",
    version="2.0.0",
    author="Divyansh Agarwal",
    author_email="divyanshagg296@gmail.com",
    description="Production-grade fraud detection pipeline: SVM · XGBoost · LightGBM · SHAP · FastAPI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/divyansh-Agarwal-16/fraud-ml-pipeline",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fraud-train=scripts.train:main",
            "fraud-predict=scripts.predict:main",
            "fraud-serve=scripts.serve:main",
            "fraud-evaluate=scripts.evaluate:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
