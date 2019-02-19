# -*- coding:utf-8 -*-
import os
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.file_util import copy_file

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        #self.build_temp = ext.sourcedir+'/build'
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir], cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'], cwd=self.build_temp)
        # copy to lib
        if not os.path.exists(self.build_lib):
            os.makedirs(self.build_lib)
        for f in self.get_outputs():
            _, name = os.path.split(f)
            dst = f
            src = os.path.join(self.build_temp, name)
            copy_file(src, dst)
            #print(dst)
        
setup(
    name='PyLandmark',
    version='0.1',
    author='zeka',
    author_email='zekang.tian@gmail.com',
    description='A facial landmark detector for python',
    long_description='',
    #packages = find_packages(),
    ext_modules=[CMakeExtension('PyLandmark')], 
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=True,
    install_requires=[
        'numpy',
    ],
    include_package_data=True,    # use MANIFEST.in
    exclude_package_data={'':['.gitignore']}
)