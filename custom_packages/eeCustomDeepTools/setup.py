from setuptools import find_packages, setup

setup(
    name='eeCustomDeepTools',
    packages=find_packages(include=['eeCustomDeepTools']),
    version='0.1.0',
    description='''Python package that supports the conversion of
                   TFRecords exported from Earth Engine into batches
                   ready to be fed into Keras Deep models.''',
    author='Davide Lomeo',
    author_email='davide.lomeo20@imperial.ac.uk',
    url='https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3/tree/main/custom_packages/eeCustomDeepTools',
    license='MIT',
    install_requires=['tensorflow'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='test_eeCustomDeepTools',
)
