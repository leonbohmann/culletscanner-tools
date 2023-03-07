import os
import re
import shutil
from itertools import groupby
import argparse

def get_img_id(file):
    return os.path.basename(file)[:4]

def get_img_type(file):
    name = os.path.basename(file)
    match = re.search(r"\[(.*)\]", name)
    return match.group(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-sc", "--start-count", help="start id", type=int, default=1)    
    parser.add_argument("-d", "--thickness", help="thickness of glass", type=int, default=4)    
    parser.add_argument("-rs", "--residual-stress", help="residual stress in glass", type=int, default=70)

    args = parser.parse_args()
    
    # cullet scanner debug output files
    extensions = [".bmp", ".zip"]
    
    # arguments to build output file names
    thickness = args.thickness
    residual_stress = args.residual_stress
    specimen_id = args.start_count
    
    # create output dir
    if not os.path.exists("out"):
        os.makedirs("out")  
    
    
    files = [f for f in os.listdir('.') if os.path.isfile(f) and any(f.endswith(ext) for ext in extensions)]

    sorted_files = sorted(files, key=get_img_id)
    grouped_files = {k: list(g) for k, g in groupby(sorted_files, key=lambda x: x[:4])}
    
    fileKeyToId = {}
    
    for key, group in grouped_files.items():        
        # get file index
        id = key
        
        specimen_name = f"{thickness}.{residual_stress}.{specimen_id}"
        print(f"Copy specimen: {specimen_name}")
        
        for file in group:            
            _, ext = os.path.splitext(file)
            cpy_path = ""
            
            # image files have "green", "blue" or "transmission"
            if( ext == ".bmp" ):
                type = get_img_type(file)
                cpy_path = f"out/{specimen_name}_p-{type.lower()}{ext}"
            else:
                cpy_path = f"out/{specimen_name}_d{ext}"

            
            shutil.copy2(file, cpy_path)
            
        # increment specimen id
        specimen_id += 1
        