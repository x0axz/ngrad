import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ngrad",
    version="1.0.0",
    description="An Autograd engine and a neural network library that handle an N-dimensional array.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nooh",
    author_email="x0axz@protonmail.com",
    url="https://github.com/x0axz/ngrad",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
