#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: update.py 
@time: 2018/03/22 
"""

import pip
from subprocess import call
for package in pip.get_installed_distributions():
   call('python3 -m pip install --upgrade ' + package.project_name)
