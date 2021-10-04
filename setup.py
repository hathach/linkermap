from distutils.core import setup

setup(
    name='linkermap',
    version='0.2.0',
    author='Ha Thach',
    author_email='thach@tinyusb.org',
    packages=['linkermap'],
    license='LICENSE',
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

