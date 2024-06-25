import re
import subprocess
from io import open
import os

# Assuming the source file contains the functions as provided
source_file_path = '/home/jvalglz/gptFortranLara/Transformations/TestTransformations/functions.c'
libraries_needed = """
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
"""
main_code_template = """

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
    saxpy_parallel(n, a, x, y);

    // Print the results
    for (int i = 0; i < n; ++i) {
        std::cout << "y(" << i + 1 << ") = " << y[i] << std::endl;
    }
    return 0;
}
"""
expected_output = "/home/jvalglz/gptFortranLara/Transformations/TestTransformations/expected_output.txt"
test_file_prefix = "Test"
test_file_counter = 1

# Function to parse and return functions from the source file
def parse_functions(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    print(content)
    
    # Remove lines containing "Code Block #: "
    content_filtered = re.sub(r'^.*Code Block\s*$', '', content, flags=re.MULTILINE)
    
    # Updated Regex to use non-capturing groups
    functions = re.findall(
        r'void\s+saxpy_parallel\s*\(\s*(?:const\s+)?int\s+n,\s*(?:const\s+)?float\s+a,\s*(?:const\s+)?float\s*\*x,\s*float\s*\*y\s*\)\s*\{.*?^\}',
        content_filtered, flags=re.DOTALL | re.MULTILINE)
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
    
    # Print the contents
    print(content)

# Main process
functions = parse_functions(source_file_path)
print(source_file_path)
totalCorrect = 0
count = 0
for function in functions:
    print(str(count))
    test_file_name = test_file_prefix + '{:03}'.format(test_file_counter) + '.cpp'
    with open(test_file_name, 'w') as test_file:
        #print(function)
        test_file.write(libraries_needed+ '\n' + function + '\n' + main_code_template)
    compiled, result = compile_execute_verify(test_file_name)
    # print_test_file(test_file_name)
    print('{test_file_name}: Compiled={compiled}, Result={result}'.format(test_file_name=test_file_name, compiled=compiled, result=result))
    test_file_counter += 1
    count+= 1
    if compiled:
        totalCorrect += 1

print('Total Correct: ' + str(totalCorrect))



