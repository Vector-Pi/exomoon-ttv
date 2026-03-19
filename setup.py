from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="exomoon-ttv",
    version="1.0.0",
    author="Om Arora",
    description="Population-level exomoon upper limits via TESS transit timing variations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vector-Pi/exomoon-ttv",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
