'''
Step 1: prompts.txt must contain multiple prompts each one separated by a white space.
Step 2: The code then proceeds to prompt ChatGPT for one prompt included in prompts.txt 10 times.
Step 3: After ChatGPT responds 10 times, we store the responses on a txt file, and the code block from each response in another txt file.
Step 4: Steps 2 and 3 are repeated for every prompt.
'''

import re
from openai import OpenAI

client = OpenAI(api_key='')  # REPLACE API KEY

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

# Improved function to extract regular functions, considering __global__ on the previous line
def extract_regular_functions(code):
    pattern = re.compile(r'\b\w+\s+\w+\s*\([^)]*\)\s*{')
    matches = pattern.finditer(code)
    functions = []

    for match in matches:
        start = match.start()
        # Ensure it is not a __global__ function
        prev_line = code[:start].rsplit('\n', 1)[-1].strip()
        if prev_line == '__global__':
            continue
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
    prompt_index = f'{i+1:03}'
    print(f'----------- PROMPT {prompt_index} -----------\n')
    # Open a file to store the responses
    with open(f'Prompt{prompt_index}_responses_gpt3.5turbo.txt', 'w') as response_file, \
         open(f'Prompt{prompt_index}_code_blocks_gpt3.5turbo.txt', 'w') as code_file:
        # Loop through and generate responses
        for j in range(num_iter):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": each_message}]
            )
            response_content = response.choices[0].message.content
            response_file.write(f'Output {j+1}:\n{response_content}\n\n')  # Write the response to the file

            # Extract and save the first function and the global function
            regular_matches = extract_regular_functions(response_content)
            global_matches = extract_global_functions(response_content)
            code_file.write(f"Code Block {j+1}:\n")
            if regular_matches:
                code_file.write(f"{regular_matches[0].strip()}\n\n")  # Write the first regular function
            if global_matches:
                code_file.write(f"{global_matches[0].strip()}\n\n")  # Write the first __global__ function
            print(f'Iteration {j+1}: {response_content}')

        print('\n')
        print(f'All responses saved to Prompt{prompt_index}_responses_gpt3.5turbo.txt')
        print(f'All code blocks saved to Prompt{prompt_index}_code_blocks_gpt3.5turbo.txt')
        print('\n')

print('-------------------WORK DONE---------------------')