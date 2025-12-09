from distutils.core import setup

setup(
    name='linkermap',
    version='0.7.0',
    author='Ha Thach',
    author_email='thach@tinyusb.org',
    packages=[],
    py_modules=['linkermap'],
    license='MIT',
    description='Analyze GNU ld’s linker map.',
    install_requires=[
        'importlib_metadata; python_version<"3.8"',
    ],
    entry_points={
        'console_scripts': [
            'linkermap = linkermap:main',
        ],
    },
)
