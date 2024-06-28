import re
import subprocess
from io import open
import os

# Assuming the source file contains the functions as provided
# source_file_path = 'functions.cpp'
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

int main() {
    const int n = 15;
    float A[n][n], x[n], y[n];

    // Initialize the matrix and vectors
    for (int i = 0; i < n; ++i) {
        x[i] = 5.0; // Initialize x to 5.0
        y[i] = 0.0; // Initialize y to 0.0
        for (int j = 0; j < n; ++j) {
            A[i][j] = 3.0; // Initialize A to 3.0
        }
    }

    // Call the gemv_parallel function

"""

gemv_code_template_after_call = """(n, (const float *)A, x, y);

    // Print the results
    for (int i = 0; i < n; ++i) {
        std::cout << "y(" << i + 1 << ") = " << y[i] << std::endl;
    }

    return 0;
}

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

error_per_test = []

# Function to compile, execute, and verify test file
def compile_execute_verify(file_path):
    compile_command = 'g++ -fopenmp ' + file_path + ' -o ' + file_path[:-2]
    # Use subprocess.Popen to compile
    process = subprocess.Popen(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        error_message = stderr.decode('utf-8')  # Decode stderr to a string
        error_per_test.append(error_message)  # Append error message to error_per_test
        return False, 'Compilation Failed: ' + error_message
    error_per_test.append(stdout.decode('utf-8'))  # Decode and append stdout to error_per_test
    # print('error_per_test: ' + str(error_per_test))
    
    # Execute the compiled program
    run_command = './' + file_path[:-2]
    process = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
    output = stdout
    
    # Compare output
    with open(expected_output, 'r') as expected_file:
        expected = expected_file.read().strip()
    if output.strip() == expected:
        return True, 'Output Correct'
    else:
        return True, 'Output Incorrect'
    
def print_test_file(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the contents of the file
        content = file.read()
    
def write_test_file(test_file_name, function, test_file_counter):
    content = libraries_needed + '\n' + function + '\n' + gemv_code_template_before_call + function_names[test_file_counter] + gemv_code_template_after_call
    with open(test_file_name, 'w') as test_file:
        test_file.write(content)

def write_results_file(functions_passed, functions_failed, error_per_test):
    with open('test_results.cpp', 'w') as results_file:
        results_file.write("Functions that passed:\n")
        for function, file_number in functions_passed:
            results_file.write(f"(Test File: {file_number})\n {function}\n")
        results_file.write("\nFunctions that failed:\n")
        for function, file_number in functions_failed:
            results_file.write(f"(Test File: {file_number})\n {function} \nError: \n")
    



def main_process(source_file_path):
    functions_passed = []
    functions_failed = []
    error_per_test = []  # Assuming this is populated within compile_execute_verify

    functions = parse_functions(source_file_path)
    print(source_file_path)
    totalCorrect = 0
    test_file_counter = 0
    for function in functions:
        test_file_name = test_file_prefix + '{:03}'.format(test_file_counter) + '.cpp'
        write_test_file(test_file_name, function, test_file_counter)
        compiled, result = compile_execute_verify(test_file_name)
        print(f'{test_file_name}: Compiled={compiled}, Result={result}')
        if compiled:
            totalCorrect += 1
            functions_passed.append((function, test_file_counter))
        else:
            functions_failed.append((function, test_file_counter))
        test_file_counter += 1

    write_results_file(functions_passed, functions_failed, error_per_test)

# Assuming the rest of the necessary variables and functions (like parse_functions, compile_execute_verify, etc.) are defined elsewhere
if __name__ == "__main__":
    source_file_path = "functions.cpp"
    main_process(source_file_path)
    print('Results printed to test_results.cpp')