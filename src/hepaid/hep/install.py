
import os
import subprocess
import tarfile
import requests
import shutil
import yaml
from pathlib import Path

def download_file(url, filename):
  """Downloads a file from the specified URL and saves it with the given filename."""

  response = requests.get(url)

  if response.status_code == 200:  
    with open(filename, "wb") as file:
      file.write(response.content)
    print(f"Downloaded {filename} successfully!")
  else:
    print(f"Download failed with status code: {response.status_code}")

def delete_file_or_directory(path):
    """Deletes the specified file or directory."""

    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError(f"Invalid path: '{path}' is not a file or directory.")

    print(f"Deleted '{path}' successfully.")

def extract_targz(targz_file, extract_path="."):
    """
    Extracts a .tar.gz archive to the specified path.

    Parameters:
        targz_file (str): Path to the .tar.gz file to extract.
        extract_path (str, optional): Path where the files should be extracted. Defaults to the current working directory (".").
    """
    try:
        with tarfile.open(targz_file, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Successfully extracted {targz_file} to {extract_path}")
    except (FileNotFoundError, tarfile.ReadError) as e:
        print(f"Error extracting {targz_file}: {e}")

def run_make_command(command, cwd):
    try:
        process = subprocess.Popen(     
            command, 
            cwd=cwd, 
            stdout=subprocess.PIPE,    
            stderr=subprocess.STDOUT,   
            text=True                  
        )

        for line in process.stdout:    
            print(line, end="")        
            if "Error" in line:         
                process.kill()          
                raise subprocess.CalledProcessError(
                    returncode=process.returncode, 
                    cmd=process.args, 
                    output=line           
                )

        returncode = process.wait()     
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, process.args)  

        print("Successfully installed!")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e.output}")


def install_spheno(
    url="https://spheno.hepforge.org/downloads?f=SPheno-4.0.4.tar.gz", 
    compiler='gfortran',
    on_mac =True,
    model_dir="BLSSM_SPheno",
    model_name="BLSSM",
    ):

    filename = url.split("/")[-1]
    filepath = Path(filename.replace('.tar.gz', "").replace("downloads?f=", ""))
    download_file(url, filename)
    extract_targz(filename, extract_path=".")
    
    tool_dir = filepath

    # 4. Modify the Makefile 
    if compiler != "ifort":
        with open(tool_dir / "Makefile", "r+") as makefile:
            content = makefile.read()
            content = content.replace("F90 = ifort", f"F90 = {compiler}")
            makefile.seek(0)
            makefile.write(content)
            makefile.truncate()
    # To build the SM SPheno
    if on_mac:
        with open(tool_dir / "src/Makefile", "r+") as makefile:
            content = makefile.read()
            content = content.replace("-U", "")
            makefile.seek(0)
            makefile.write(content)
            makefile.truncate()
    make_command = ["make"]
    run_make_command(make_command, tool_dir )
    
    if model_dir is not None:
        new_model_dir = tool_dir / model_name
        shutil.copytree(model_dir, new_model_dir)
        if on_mac:
            with open( new_model_dir/ "Makefile", "r+") as makefile:
                content = makefile.read()
                content = content.replace("-U", "")
                makefile.seek(0)
                makefile.write(content)
                makefile.truncate()

    make_command = ["make", f"Model={model_name}"]
    run_make_command(make_command, tool_dir )
    delete_file_or_directory(filename)
    return tool_dir
    

def install_higgsbounds(
    url='https://gitlab.com/higgsbounds/higgsbounds/-/archive/master/higgsbounds-master.tar.gz',
    ):
    filename= 'higgsbounds.tar.gz'
    filepath = 'higgsbounds-master'
    download_file(url, filename)

    extract_targz(filename, extract_path=".")

    
    source_dir = Path(filepath)
    build_dir = source_dir /"build"

    
    build_dir.mkdir(exist_ok=True)  

    run_make_command(["cmake", ".."], build_dir)

    
    run_make_command(["make"], cwd=build_dir)
    delete_file_or_directory(filename)
    return build_dir 
    


def install_higgssignals(
    url='https://gitlab.com/higgsbounds/higgssignals/-/archive/master/higgssignals-master.tar.gz'
    ):

    filename= 'higgssignals.tar.gz'
    filepath = 'higgssignals-master'
    download_file(url, filename)

    extract_targz(filename, extract_path=".")

    
    source_dir = Path(filepath)
    build_dir = source_dir /"build"

    
    build_dir.mkdir(exist_ok=True)  

    
    run_make_command(["cmake", ".."], build_dir)

    
    run_make_command(["make"], cwd=build_dir)
    delete_file_or_directory(filename)
    return build_dir

def install_madgraph(
        url = "https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5_aMC_v3.5.4.tar.gz",
        model_dir = 'BLSSM_UFO',
        model_name = 'BLSSM'
):

    filename=url.split("/")[-1]
    download_file(url, filename)
    extract_targz(filename)

    filepath = Path(filename.replace('.tar.gz', "").replace(".", "_"))

    model_dir = Path(model_dir)
    new_model_dir = filepath / 'models' / model_name
    shutil.copytree(model_dir, new_model_dir)

    delete_file_or_directory(filename)
    return filepath

def install_hepstack(
    spheno_url="https://spheno.hepforge.org/downloads?f=SPheno-4.0.4.tar.gz", 
    spheno_compiler='gfortran',
    spheno_on_mac =True,
    spheno_model_dir="BLSSM_SPheno",
    spheno_model_name="BLSSM",
    higgsbounds_url='https://gitlab.com/higgsbounds/higgsbounds/-/archive/master/higgsbounds-master.tar.gz',
    higgssignals_url='https://gitlab.com/higgsbounds/higgssignals/-/archive/master/higgssignals-master.tar.gz',
    madgraph_url = "https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5_aMC_v3.5.4.tar.gz",
    madgraph_model_dir = 'BLSSM_UFO',
    madgraph_model_name = 'BLSSM'
    ):
    spheno_dir = install_spheno(
        spheno_url,
        spheno_compiler,
        spheno_on_mac,
        spheno_model_dir,
        spheno_model_name,
        )
    hb_dir = install_higgsbounds(
        higgsbounds_url,
    )
    hs_dir = install_higgssignals(
        higgssignals_url
    )
    mg_dir = install_madgraph(
        madgraph_url,
        madgraph_model_dir,
        madgraph_model_name 
    )
    write_config_template(spheno_dir, hb_dir, hs_dir, mg_dir)

def write_config_template(spheno_dir, hb_dir, hs_dir, mg_dir):
    """Write a configuration file template with blank values."""
    config_template = {
        'model': {
            'name': 'BLSSM',
            'input': {
                'm0': {
                    'block_index': 1,
                    'block_name': 'MINPAR'
                },
                'm12': {
                    'block_index': 2,
                    'block_name': 'MINPAR'
                },
                'TanBeta': {
                    'block_index': 3,
                    'block_name': 'MINPAR'
                },
                'Azero': {
                    'block_index': 5,
                    'block_name': 'MINPAR'
                },
                'MuInput': {
                    'block_index': 11,
                    'block_name': 'EXTPAR'
                },
                'MuPInput': {
                    'block_index': 12,
                    'block_name': 'EXTPAR'
                },
                'BMuInput': {
                    'block_index': 13,
                    'block_name': 'EXTPAR'
                },
                'BMuPInput': {
                    'block_index': 14,
                    'block_name': 'EXTPAR'
                }
            }
        },
        'spheno': {
            'model': 'BLSSM',
            'reference_slha': 'configs/hep_files/diphoton_paper_v2',
            'directory': spheno_dir
        },
        'higgsbounds': {
            'neutral_higgs': 6,
            'charged_higgs': 1,
            'directory': hb_dir
        },
        'higgssignals': {
            'neutral_higgs': 6,
            'charged_higgs': 1,
            'directory': hs_dir
        },
        'madgraph': {
            'directory': mg_dir,
            'scripts': {
                'gghaa': 'configs/hep_files/mg5/blssm_pphaa_LHC13.txt'
            }
        },
        'hep_stack': {
            'name': 'SPhenoHBHSMG5',
            'scan_dir': 'datasets/scans',
            'final_dataset': 'datasets',
            'delete_on_exit': True
        }
    }
    
    
    # Write the configuration template to a YAML file
    with open('hepstack_config.yaml', 'w') as file:
        yaml.dump(config_template, file, default_flow_style=False)
    
    print(f"Configuration template written to {'hepstack_config.yaml'}")


