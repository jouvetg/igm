#!/usr/bin/env python3
 
import glob,os,sys

L=[]
for m in ['preproc','process','postproc']:
    
    M = glob.glob('../igm/modules/'+m+'/*.py')
    M = [l.split('/')[-1].split('.')[0] for l in M]
    M.remove('__init__')
    L=L+M
    
L.remove('optimize_v1')
L.remove('iceflow_v1')
L.remove('particles_v1')
L.remove('flow_dt_thk')

print(L)
    
with open('igm-run.py', "w") as f:

    print("import sys", file=f)
    print("sys.path.append('../')", file=f)
    print("import argparse", file=f)
    print("import igm", file=f)
    print("parser = igm.params_core()", file=f)
    for l in L:
        print("igm.params_"+l+"(parser)", file=f)
    print("params = parser.parse_args()", file=f)
            
 
os.system("argmark -f igm-run.py")
os.system("echo ' ' > argmark1.md")
os.system("echo '# Parameters: ' >> argmark1.md")
os.system("sed -z 's/.*Arguments//' argmark.md >> argmark1.md")
os.system("mv argmark.md all.md")
os.system("rm argmark1.md igm-run.py")


