import os

from IPython import embed

# get $USER 
user = os.environ['USER']
__all__ = ['parentpath', 'outpath', 'datpath']

if user == 'topol': # Daming Yang
    parentpath = '/Users/topol/workspace/euclid/euclid_qsos/'

    outpath = os.path.join(parentpath, 'output')
    datpath = os.path.join(parentpath, 'data')
    figpath = os.path.join(parentpath, 'figures')

    os.makedirs(outpath, exist_ok=True)
    os.makedirs(datpath, exist_ok=True)
    os.makedirs(figpath, exist_ok=True)

if user == 'dyang': # on igm server
    parentpath = '/data1/dyang/euclid/euclid_qsos/'

    outpath = os.path.join(parentpath, 'output')
    datpath = os.path.join(parentpath, 'data')
    figpath = os.path.join(parentpath, 'figures')

    os.makedirs(outpath, exist_ok=True)
    os.makedirs(datpath, exist_ok=True)
    os.makedirs(figpath, exist_ok=True)