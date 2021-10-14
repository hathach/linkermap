from distutils.core import setup
from linkermap.__version__ import version_str

setup(
    name='linkermap',
    version=version_str,
    author='Ha Thach',
    author_email='thach@tinyusb.org',
    packages=['linkermap'],
    license='MIT',
    description='Analyze GNU ld’s linker map.',
    install_requires=[
        'click'
    ],
    entry_points={
    'console_scripts': [
            'linkermap = linkermap:main',
            ],
    },
)

