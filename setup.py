from setuptools import setup, find_packages

setup(
    name="german-power-grid",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Lukas Malms",
    author_email="lukas.malms@gmail.com",
    description="Energy Demand Forecasting for the German Power Grid",
    # url=
    install_requires=[
        "beautifulsoup4~=4.9.3",
        "numpy~=1.21.1",
        "pandas~=1.3.1",
        "requests~=2.26.0",
        "regex~=2021.8.3",
        "scikit-learn~=0.24.2",
        "scipy~=1.7.1",
        "tqdm~=4.62.0"
    ]
)