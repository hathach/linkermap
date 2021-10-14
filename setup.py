from distutils.core import setup
import linkermap.__version__

setup(
    name='linkermap',
    version=linkermap.__version__.version_str,
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

