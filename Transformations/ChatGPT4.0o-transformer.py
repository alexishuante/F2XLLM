
import re
from openai import OpenAI

client = OpenAI()  # REPLACE API KEY

# Improved function to extract __global__ functions, considering nested braces
def extract_global_functions(code):
    pattern = re.compile(r'__global__\s+\w+\s+\w+\s*\([^)]*\)\s*{')
    matches = pattern.finditer(code)
    functions = []

    for match in matches:
        start = match.start()
        brace_count = 0
        for i in range(start, len(code)):
            if code[i] == '{':
                brace_count += 1
            elif code[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    functions.append(code[start:i+1])
                    break

    return functions


# Improved function to extract regular functions, including those starting with extern "C" and excluding __global__ and int main
def extract_regular_functions(code):
    pattern = re.compile(r'(?<!__global__\s)(extern\s+"C"\s+)?\b(?!int\s+main\b)\w+\s+\w+\s*\([^)]*\)\s*{')
    matches = pattern.finditer(code)
    functions = []

    for match in matches:
        start = match.start()
        brace_count = 0
        for i in range(start, len(code)):
            if code[i] == '{':
                brace_count += 1
            elif code[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    functions.append(code[start:i+1])
                    break

    return functions

messages = []
message = ''
with open('prompts.txt', 'r') as f:
    for line in f:
        if line.strip() == '':
            if message:
                messages.append(message.strip())  # Add message to list
                message = ''  # Restart message again from 0.
        else:
            message += line
    if message:
        messages.append(message.strip())  # Add last message

num_iter = 10

for i, each_message in enumerate(messages):
    prompt_number = str(i + 1).zfill(3)
    print(f'----------- PROMPT {prompt_number} -----------\n')
    # Open a file to store the responses
    with open(f'Prompt{prompt_number}_responses_gpt-4o.txt', 'w') as response_file, \
            open(f'Prompt{prompt_number}_code_blocks_gpt-4o.txt', 'w') as code_file:
        # Loop through and generate responses
        for j in range(num_iter):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": each_message}]
            )
            response_content = response.choices[0].message.content
            response_file.write(f'Output {j + 1}:\n{response_content}\n\n')  # Write the response to the file

            # Extract and save the first non-main function
            regular_matches = extract_regular_functions(response_content)
            global_matches = extract_global_functions(response_content)

            code_file.write(f"Code Block {j + 1}:\n")
            if regular_matches:
                code_file.write(f"{regular_matches[0].strip()}\n\n")  # Write the first regular function
            if global_matches:
                code_file.write(f"{global_matches[0].strip()}\n\n")  # Write the first __global__ function

            print(f'Iteration {j + 1}: {response_content}')

        print('\n')
        print(f'All responses saved to Prompt{prompt_number}_responses_gpt-4o.txt')
        print(f'All code blocks saved to Prompt{prompt_number}_code_blocks_gpt-4o.txt')
        print('\n')

print('-------------------WORK DONE---------------------')
