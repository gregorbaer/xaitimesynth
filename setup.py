from setuptools import find_packages, setup

setup(
    name="xaitimesynth",
    version="0.1",
    description="",
    author="Gregor Baer",
    author_email="g.baer@tue.nl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "lets_plot",
        "ipykernel",
        "ipywidgets",
        "ruff",
    ],
)
