from distutils.core import setup
import subprocess

PYSUFFIX = subprocess.run(["python3-config", "--extension-suffix"], capture_output=True, text=True).stdout.strip("\n")
# PYSUFFIX = '.cpython-38-x86_64-linux-gnu.so'

setup(
    name='nalger_helper_functions',
    version='0.1dev',
    packages=['nalger_helper_functions',],
    license='MIT',
    author='Nick Alger',
    # long_description=open('README.txt').read(),
    include_package_data=True,
    package_data={'nalger_helper_functions': ['nalger_helper_functions_cpp'+PYSUFFIX]},
)
