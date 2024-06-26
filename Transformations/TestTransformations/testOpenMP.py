import re
import subprocess
from io import open
import os

# Assuming the source file contains the functions as provided
source_file_path = 'functions.cpp'
libraries_needed = """
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
"""
axpy_code_template_before_call = """

int main() {
    const int n = 10;
    float x[n], y[n];
    float a = 2.0;

    // Initialize arrays x and y
    for (int i = 0; i < n; ++i) {
        x[i] = 3.0;
        y[i] = 0.0;
    }

    // Call the saxpy_parallel function
"""

axpy_code_template_after_call = """(n, a, x, y);
// Print the results
    for (int i = 0; i < n; ++i) {
        std::cout << "y(" << i + 1 << ") = " << y[i] << std::endl;
    }
    return 0;
}
"""
gemv_code_template_before_call = """



"""

expected_output = "expected_output.txt"
test_file_prefix = "Test"
test_file_counter = 1



# Declare function_names as a global variable
function_names = []

# Function to parse and return functions from the source file
def parse_functions(file_path):
    global function_names  # Declare the use of the global variable within this function
    with open(file_path, 'r') as file:
        content = file.read()
    content_filtered = re.sub(r'^.*Code Block\s*$', '', content, flags=re.MULTILINE)
    
    # Use a capturing group to capture function names
    pattern_for_names = r'void\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(?:[^)]*)\)\s*'
    function_names = re.findall(pattern_for_names, content_filtered)  # This will update the global variable
    
    # Use the original pattern to capture the entire function definition
    pattern_for_functions = r'void\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*(?:[^)]*)\)\s*\{.*?^\}'
    functions = re.findall(pattern_for_functions, content_filtered, flags=re.DOTALL | re.MULTILINE)
    
    return functions

# Function to compile, execute, and verify test file
def compile_execute_verify(file_path):
    compile_command = 'g++ -fopenmp ' + file_path + ' -o ' + file_path[:-2]
    # Use subprocess.Popen to compile
    process = subprocess.Popen(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        return False, 'Compilation Failed'
    
    # Execute the compiled program
    run_command = './' + file_path[:-2]
    process = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
    output = stdout 
       
    # Compare output
    with open(expected_output, 'r') as expected_file:
        expected = expected_file.read().strip()
        # print('output: ' + output.strip() + ' expected: ' + expected) 
    if output.strip() == expected:
        return True, 'Output Correct'
    else:
        return True, 'Output Incorrect'
    
def print_test_file(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the contents of the file
        content = file.read()
    
# Initialize lists to keep track of function test results
functions_passed = []
functions_failed = []

# Main process
functions = parse_functions(source_file_path)
print(source_file_path)
totalCorrect = 0
test_file_counter = 0  # Ensure test_file_counter is initialized
for function in functions:
    test_file_name = test_file_prefix + '{:03}'.format(test_file_counter) + '.cpp'
    with open(test_file_name, 'w') as test_file:
        test_file.write(libraries_needed + '\n' + function + '\n' + axpy_code_template_before_call + function_names[test_file_counter] + axpy_code_template_after_call)
    compiled, result = compile_execute_verify(test_file_name)
    print('{test_file_name}: Compiled={compiled}, Result={result}'.format(test_file_name=test_file_name, compiled=compiled, result=result))
    if compiled:
        totalCorrect += 1
        # Append both the function and its corresponding test file number
        functions_passed.append((function, test_file_counter))
    else:
        functions_failed.append((function, test_file_counter))
    test_file_counter += 1

# Write the test results to a file
with open('test_results.cpp', 'w') as results_file:
    results_file.write("Functions that passed:\n")
    for function, file_number in functions_passed:
        results_file.write(f"(Test File: {file_number})\n {function}\n")
    results_file.write("\nFunctions that failed:\n")
    for function, file_number in functions_failed:
        results_file.write(f"(Test File: {file_number})\n {function} \n")
    results_file.write(f"\nTotal Correct: {len(functions_passed)}/{len(functions_passed) + len(functions_failed)}\n")

print("Test results written to test_results.cpp")

