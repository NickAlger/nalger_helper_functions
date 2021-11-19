from distutils.core import setup

setup(
    name='nalger_helper_functions',
    version='0.1dev',
    packages=['nalger_helper_functions',],
    license='MIT',
    author='Nick Alger',
    # long_description=open('README.txt').read(),
    include_package_data=True,
    package_data={'nalger_helper_functions': ['nalger_helper_functions_cpp.so']},
)
