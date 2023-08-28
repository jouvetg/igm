#!/usr/bin/env python3
 
import glob,os,sys


for m in ['preproc','process','postproc']:
    
    L = glob.glob('../igm/modules/'+m+'/*.py')
    L = [l.split('/')[-1].split('.')[0] for l in L]
    L.remove('__init__')
    
    for l in L:

        print(m,l)

        with open('igm-run.py', "w") as f:

            print("import sys", file=f)
            print("sys.path.append('../')", file=f)
            print("import argparse", file=f)
            print("import igm", file=f)
            print("parser = argparse.ArgumentParser()", file=f)
            print("igm.params_"+l+"(parser)", file=f)
            print("params = parser.parse_args()", file=f)
            
         
        os.system("argmark -f igm-run.py")
        os.system("echo ' ' > argmark1.md")
        os.system("echo '# Parameters: ' >> argmark1.md")
        os.system("sed -z 's/.*Arguments//' argmark.md >> argmark1.md")

        os.system('cat ../igm/modules/'+m+'/'+l+'.md argmark1.md > ' + l + '.md')

os.system('rm igm-run.py argmark.md argmark1.md')

######################################################

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
    print("parser = argparse.ArgumentParser()", file=f)
    for l in L:
        print("igm.params_"+l+"(parser)", file=f)
    print("params = parser.parse_args()", file=f)
            
 
os.system("argmark -f igm-run.py")
os.system("echo ' ' > argmark1.md")
os.system("echo '# Parameters: ' >> argmark1.md")
os.system("sed -z 's/.*Arguments//' argmark.md >> argmark1.md")
os.system("mv argmark.md all.md")
os.system("rm argmark1.md igm-run.py")


