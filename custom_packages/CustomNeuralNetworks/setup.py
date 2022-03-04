from setuptools import find_packages, setup

setup(
    name='CustomNeuralNetworks',
    packages=find_packages(include=['CustomNeuralNetworks']),
    version='0.1.0',
    description='Python package that implements custom Keras neural newtoks',
    author='Davide Lomeo',
    author_email='davide.lomeo20@imperial.ac.uk',
    url='https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3/tree/main/custom_packages/CustomNeuralNetworks',
    license='MIT',
    install_requires=['tensorflow'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='test_CustomNeuralNetworks',
)
