#!/usr/bin/env python3
"""
run_matlab.py

This script runs the MATLAB script 'test.m' on a Linux server and captures its output.
"""

import subprocess
import os

def run_matlab_script(matlab_script_path):
    """
    Executes a MATLAB script from the command line.

    Args:
        matlab_script_path (str): Path to the MATLAB script (.m file).

    Returns:
        tuple: (stdout, stderr) from the MATLAB execution.
    """
    # Ensure the MATLAB script exists
    if not os.path.isfile(matlab_script_path):
        raise FileNotFoundError(f"MATLAB script not found: {matlab_script_path}")

    # Construct the MATLAB command
    # -batch runs the script and exits MATLAB after completion
    # Alternatively, you can use -nodisplay -nosplash -r "run('test.m'); exit;"
    matlab_command = [
        'matlab',
        '-batch',
        f"run('{matlab_script_path}');"
    ]

    try:
        # Execute the MATLAB command
        result = subprocess.run(
            matlab_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Decode output as string
            check=True  # Raise an error if the command fails
        )
        return (result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        # Handle errors in MATLAB execution
        print("An error occurred while running the MATLAB script.")
        print("Return Code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)
        raise

def main():
    # Path to the MATLAB script
    matlab_script = 'test.m'  # Ensure this script is in the current directory or provide full path

    # Optionally, change to the directory containing the MATLAB script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)

    print(f"Running MATLAB script: {matlab_script}")

    # Run the MATLAB script
    stdout, stderr = run_matlab_script(matlab_script)

    # Display MATLAB output
    print("MATLAB Output:")
    print(stdout)

    if stderr:
        print("MATLAB Errors:")
        print(stderr)

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
