from meta_ai_api import MetaAI

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
    prompt_number = str(i + 1).zfill(3)
    print(f'----------- PROMPT {prompt_number} -----------\n')
    # Open a file to store the responses
    with open(f'Prompt{prompt_number}_responses_llama3.txt', 'w') as response_file:
        # Loop through and generate responses
        for j in range(num_iter):
            response = ai.prompt(message=each_message)  # Prompt and get a response
            response_message = response['message']  # Get message only instead of message, sources, and media
            response_file.write(f'--------------- Output {j+1} ---------------\n{response_message}\n')  # Write the response to the file

            print(f'\n--------------- Iteration {j+1} ---------------\n\n {response_message}')

        print('\n')
        print(f'All responses saved to Prompt{prompt_number}_responses_llama3.txt')
        print('\n')

print('-------------------WORK DONE---------------------')
