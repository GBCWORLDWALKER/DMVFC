import yaml
import os
import shutil

def load_config(args):
    if args.config:
        # Load YAML configuration
        yaml_config = yaml.load(open(args.config), Loader=yaml.FullLoader)
        # Update the command-line arguments with YAML config
        for key, value in yaml_config.items():
            setattr(args, key, value)
    return args

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))
    
    if scripts_to_save is not None:
        scripts_path = os.path.join(path, 'scripts')
        if not os.path.exists(scripts_path):
            os.mkdir(scripts_path)
        for script in scripts_to_save:
            dst_file = os.path.join(scripts_path, os.path.basename(script))
            shutil.copyfile(script, dst_file) 