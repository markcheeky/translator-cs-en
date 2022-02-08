from setuptools import setup, find_packages

setup(
    name="translator-cs-en",
    author="Marek Kadlcik",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": ["translator=translator.main:main"],
    },
)
