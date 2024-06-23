'''
Step 1: prompts.txt must contain multiple prompts each one separated by a white space.
Step 2: The code then proceeds to prompt Llama-3 for one prompt included in prompts.txt 10 times.
Step 3: After Llama-3 responds 10 times, we store the responses on a txt file, and the code block from each response in another txt file.
Step 4: Steps 2 and 3 are repeated for every prompt.
'''

import re
from meta_ai_api import MetaAI

# Explanation of this code pattern:
#\b\w+: Matches a word boundary followed by one or more word characters, representing the return type (like int, float, void, etc.).
# \s+: Matches one or more whitespace characters.
# \w+: Matches the function name, which is one or more word characters.
# \s*\([^)]*\): Matches zero or more whitespaces followed by parentheses that might include function parameters ([^)]* matches any character except a closing parenthesis, repeated any number of times).
# \s*{[^}]*}: Matches zero or more whitespaces followed by a curly brace containing the function body ([^}]* matches any character except a closing brace, repeated any number of times).
code_block_pattern = r"\b\w+\s+\w+\s*\([^)]*\)\s*{[^}]*}"

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

ai = MetaAI()  # No need to Reset conversation context (in this code, llama-3 does not remember previous conversations)
for i, each_message in enumerate(messages):
    print('----------- PROMPT', i + 1, '-----------\n')
    # Open a file to store the responses
    with open('Prompt' + str(i+1) + '_responses_llama3.txt', 'w') as response_file, \
         open('Prompt' + str(i+1) + '_code_blocks_llama3.txt', 'w') as code_file:
        # Loop through and generate responses
        for j in range(num_iter):
            response = ai.prompt(message=each_message)  # Prompt and get a response
            response_message = response['message']  # Get message only instead of message, sources, and media
            response_file.write(f'Output {j+1}:\n{response_message}\n\n')  # Write the response to the file

            # Extract and save the first code block
            matches = re.findall(code_block_pattern, response_message) #Find the code pattern in the response_message
            if matches:
                code_only = matches[0].strip()  # Extract and clean the first code #Sometimes the LLM may give multiple codes, just extract the first one.
                code_file.write(f'Code Block {j+1}:\n{code_only}\n\n') # Write the code in the txt file
            print(f'Iteration {j+1}: {response_message}')

        print('\n')
        print('All responses saved to Prompt' + str(i+1) + '_responses_llama3.txt')
        print('All code blocks saved to Prompt' + str(i+1) + '_code_blocks_llama3.txt')
        print('\n')

print('-------------------WORK DONE---------------------')
