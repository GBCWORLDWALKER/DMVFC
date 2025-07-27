#!/usr/bin/env python3
"""
run_matlab.py

This script runs the MATLAB script 'test.m' on a Linux server and captures its output.
"""

import subprocess
import os

def run_matlab_script(matlab_script_path):
    """
    Executes a MATLAB script from the command line and displays output in real-time.

    Args:
        matlab_script_path (str): Path to the MATLAB script (.m file).

    Returns:
        int: Return code of the MATLAB process.
    """
    # Ensure the MATLAB script exists
    if not os.path.isfile(matlab_script_path):
        raise FileNotFoundError(f"MATLAB script not found: {matlab_script_path}")

    # Construct the MATLAB command
    matlab_command = [
        'matlab',
        '-nodisplay',
        '-nosplash',
        '-nodesktop',
        '-r',
        f"try, run('{matlab_script_path}'); catch ME, disp(ME.message); end; exit;"
    ]

    try:
        # Execute the MATLAB command
        process = subprocess.Popen(
            matlab_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Print output in real-time
        for line in process.stdout:
            print(line, end='')  # end='' to avoid double line breaks

        # Wait for the process to complete and get the return code
        return_code = process.wait()
        return return_code

    except subprocess.CalledProcessError as e:
        # Handle errors in MATLAB execution
        print("An error occurred while running the MATLAB script.")
        print("Return Code:", e.returncode)
        print("Error:", e.stderr)
        raise

def main():
    # Path to the MATLAB script
    matlab_script = 'src/code/distanceBOLD_bundle.m'

    print(f"Running MATLAB script: {matlab_script}")

    # Run the MATLAB script
    return_code = run_matlab_script(matlab_script)

    if return_code != 0:
        print(f"MATLAB script exited with non-zero return code: {return_code}")
    else:
        print("MATLAB script completed successfully.")

    # Verify that the output file was created
    output_file = 'output.txt'
    if os.path.isfile(output_file):
        print(f"'{output_file}' has been created successfully.")
        # Optionally, display its contents
        with open(output_file, 'r') as f:
            content = f.read()
        print("Contents of 'output.txt':")
        print(content)
    else:
        print(f"Failed to create '{output_file}'.")

if __name__ == "__main__":
    main()
