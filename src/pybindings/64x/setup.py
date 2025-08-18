from setuptools import setup, Extension, find_packages
import pybind11

ext_modules = [
    Extension(
        'neural_web',
        sources=['neural_web.cpp'],  
        include_dirs=[
            pybind11.get_include()
        ],
        language='c++',
        extra_compile_args=['-std=c++17'],  
    ),
]

setup(
    name='neural_web',
    version='1.0',
    author='Okerew',
    author_email='okerewgroup@proton.me',
    description='Neural Web',
    packages=find_packages(),
    package_data={'': ['*.txt', '*.json']},
    include_package_data=True,
    ext_modules=ext_modules,
    install_requires=['pybind11'],
)
