from setuptools import find_packages, setup

setup(
    name='eeCustomTools',
    packages=find_packages(include=['eeCustomTools']),
    version='0.1.0',
    description='''Python package that supports classification tasks
                   through computation of spectral indices, image
                   segmetation and cloud masking.''',
    author='Davide Lomeo',
    author_email='davide.lomeo20@imperial.ac.uk',
    url='https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3/tree/main/custom_packages/eeCustomTools/eeCustomTools/',
    license='MIT',
    install_requires=['earthengine-api'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='test_eeCustomTools',
)
