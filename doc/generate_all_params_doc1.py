#!/usr/bin/env python3
 
import glob,os,sys


for m in ['preproc','process','postproc']:
    
    L = glob.glob('../igm/modules/'+m+'/*/*.py')
    L = [l.split('/')[-1].split('.')[0] for l in L]
    while '__init__' in L:
        L.remove('__init__')
     
    for l in L:

        print(m,l)

        with open('igm-run.py', "w") as f:

            print("import sys", file=f)
            print("sys.path.append('../')", file=f)
            print("import igm", file=f)            
            print("parser = igm.params_core()", file=f)
            print("params, __ = parser.parse_known_args()", file=f)
            print('params.modules_'+m+'=["'+l+'"]', file=f)
            print("imported_modules = igm.load_modules(params)", file=f)
            print("igm.modules."+m+"."+l+".params(parser)", file=f)
            print("params = parser.parse_args()", file=f)

        os.system("argmark -f igm-run.py")
        os.system("echo ' ' > argmark1.md")
        os.system("echo '# Parameters: ' >> argmark1.md")
        os.system("sed -z 's/.*Arguments//' argmark.md >> argmark1.md")

        os.system('cat ../igm/modules/'+m+'/'+l+'/'+l+'.md argmark1.md > ' + l + '.md')

os.system('rm igm-run.py argmark.md argmark1.md')
 
