from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mangroves_classification",
    author="Davide Lomeo",
    author_email="davide.lomeo20@imperial.co.uk",
    description="mangroves_classification is a workflow that runs entirely on google cloud platforms and allows the user to classify mangroves using CNNs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3",
    license="MIT",
    packages=find_packages()
)
