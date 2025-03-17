import json
 
translate = {
  "time": "time",
  "thk": "thk",
  "avalanche": "aval",
  "load_ncdf": "lncd",
  "load_tif": "ltif",
  "plot2d": "plt2d",
  "clim_glacialindex": "clim",
  "clim_oggm": "clim_oggm",
  "smb_accpdd": "smb_accpdd",
  "smb_accmelt": "smb_accmelt",
  "smb_oggm": "smb_oggm",
  "smb_simple": "simple",
  "vert_flow": "vflo",
  "enthalpy": "enth",
  "write_ncdf": "wncd",
  "read_output": "rncd",
  "oggm_shop": "oggm",
  "particles": "part"
}

translate_iceflow = {
  "iceflow": "iflo",
  "optimize": "opti"
}

def remove_comments(json_str) -> str:
    lines = json_str.split("\n")
    cleaned_lines = [
        line for line in lines if not line.strip().startswith(("//", "#"))
    ]  # ! TODO: Add blocks comments...
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text

 
with open('params.json', 'r') as f:
    json_text = f.read()

json_text = remove_comments(json_text)

params_dict = json.loads(json_text)

list_modules = params_dict["modules_preproc"] + params_dict["modules_process"] + params_dict["modules_postproc"]


# print_info = ("print_info" in params_dict["modules_postproc"])
# print_comp = ("print_comp" in params_dict["modules_postproc"])
 
params_dict["modules_postproc"] = [ f for f in params_dict["modules_postproc"] if f not in ["print_info","print_comp"] ]
   

imodules = params_dict["modules_preproc"]
modules  = params_dict["modules_process"] 
omodules = params_dict["modules_postproc"]

del params_dict["modules_preproc"] 
del params_dict["modules_process"] 
del params_dict["modules_postproc"]

def print_in_file(modules,file):
    for key in modules:
        print("  "+key+":", file=file)

        if key=="iceflow":
            param_iceflow = [s for s in params_dict if s.startswith('iflo')]
            if len(param_iceflow)>0:
                param_iceflow = sorted(param_iceflow)
                print("    iceflow:", file=file)
                for s in param_iceflow:
                    print("      ",s.split('_',1)[-1],":",params_dict[s], file=file)

            param_optimize = [s for s in params_dict if s.startswith('opti')]
            if len(param_optimize)>0:
                param_optimize = sorted(param_optimize)
                print("    optimize:", file=file)
                for s in param_optimize:
                    print("      ",s.split('_',1)[-1],":",params_dict[s], file=file)
        else:
            if key in translate:
                param_module = [s for s in params_dict if s.startswith(translate[key])]
                param_module = sorted(param_module)
                for s in param_module:
                    print("    ",s.split('_',1)[-1],":",params_dict[s], file=file)

with open('params.yaml', 'w') as file:

    print("# @package _global_", file=file)
    print("", file=file)

#    print("hydra:", file=file)
#    print("  job:", file=file)
#    print("    chdir: True", file=file)

    print("", file=file)
    print("core:", file=file)
    if "url_data" in params_dict: 
        print("  url_data: "+params_dict["url_data"], file=file) 

    print("", file=file)
    print("defaults:", file=file) 
    print("  - override /inputs: [" + ", ".join(imodules) + "]", file=file)
    print("  - override /processes: [" + ", ".join(modules) + "]", file=file)
    print("  - override /outputs: [" + ", ".join(omodules) + "]", file=file)

    print("", file=file)
    print("inputs:", file=file)
    print_in_file(imodules,file)

    print("", file=file)
    print("processes:", file=file)
    print_in_file(modules,file)

    print("", file=file)
    print("outputs:", file=file)
    print_in_file(omodules,file)


