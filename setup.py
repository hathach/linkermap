from distutils.core import setup

setup(
    name='linkermap',
    version='0.3.0',
    author='Ha Thach',
    author_email='thach@tinyusb.org',
    packages=['linkermap'],
    license='MIT',
    description='Analyze GNU ld’s linker map.',
    install_requires=[
        'Click',
    ],
    entry_points={
    'console_scripts': [
            'linkermap = linkermap:main',
            ],
    },
)

