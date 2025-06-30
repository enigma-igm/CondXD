# Installation setup for XDhighzQSO
from setuptools import setup

setup(name='CondXD',
      version='1.0',
      description='XD classification high-z QSOs',
      author='Yi Kang',
      author_email='yi_kang@physics.ucsb.edu,',
      #license='GPL',
      url='https://github.com/enigma-igm/CondXD',
      packages=['condxd'],
      requires=['numpy', 'matplotlib','scipy','torch'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.6',
          'Topic :: Documentation :: Sphinx',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Software Development :: User Interfaces'
      ],
     )
