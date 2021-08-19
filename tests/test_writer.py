#!/usr/bin/env python3
# coding: utf-8
"""
@file: test_writer.py
@description: 
@author: Yiran Wu
@email: wuyiran@genomics.cn
@last modified by: Yiran Wu

change log:
    2021/8/13 create file.
"""

from stereo.io.writer import write
from stereo.io.writer import write_h5ad
from stereo.io.reader import read

def check_write(data,out):
    myout = out + "/check_write/"
    check_out(myout)
    write(data,output=myout+"/F5.bin200.h5ad")
    print("finished in "+myout)

def check_write_h5ad(data,out):
    myout = out + "/check_write_h5ad/"
    check_out(myout)
    data.output=myout+"/F5.bin200.h5ad"
    write_h5ad(data)
    print("finished in " + myout)

def check_out(out):
    from pathlib import Path
    myout = Path(out)
    if not myout.exists():
        myout.mkdir()

if __name__ == '__main__':
    import sys
    out = sys.argv[1]
    print("working dir is "+out)
    data = read(file_path=out+"/F5.gem.txt",file_format='txt',bin_size=200)
    check_write(data,out)
    check_write_h5ad(data,out)