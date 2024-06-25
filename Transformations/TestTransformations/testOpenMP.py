import re
import subprocess
from io import open
import os

# Remove the unnecessary import statement

# Assuming the source file contains the functions as provided
source_file_path = 'functions.c'
main_code_template = """
int main() {
    // Test code here, e.g., initialize arrays x and y, call saxpy_parallel, and print results
    return 0;
}
"""
expected_output = "expected_output.txt"
test_file_prefix = "Test"
test_file_counter = 1

# Function to parse and return functions from the source file
def parse_functions(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    # Regex to match C function definitions
    functions = re.findall(r'void\s+saxpy_parallel.*?\{.*?^\}', content, flags=re.DOTALL | re.MULTILINE)
    return functions

# Function to compile, execute, and verify test file
def compile_execute_verify(file_path):
    compile_command = 'gcc -fopenmp ' + file_path + ' -o ' + file_path[:-2]
    compile_result = subprocess.run(compile_command, shell=True, capture_output=True)
    if compile_result.returncode != 0:
        return False, 'Compilation Failed'
    # Execute the compiled program
    run_command = './' + file_path[:-2]
    run_result = subprocess.run(run_command, shell=True, capture_output=True, text=True)
    output = run_result.stdout
    # Compare output
    with open(expected_output, 'r') as expected_file:
        expected = expected_file.read().strip()
    if output.strip() == expected:
        return True, 'Output Correct'
    else:
        return True, 'Output Incorrect'

# Main process
functions = parse_functions(source_file_path)
for function in functions:
    test_file_name = test_file_prefix + '{:03}'.format(test_file_counter) + '.c'
    with open(test_file_name, 'w') as test_file:
        test_file.write(function + '\n' + main_code_template)
    compiled, result = compile_execute_verify(test_file_name)
    print('{test_file_name}: Compiled={compiled}, Result={result}'.format(test_file_name=test_file_name, compiled=compiled, result=result))
    test_file_counter += 1