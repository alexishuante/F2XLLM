'''
Step 1: prompts.txt must contain multiple prompts each one separated by a white space.
Step 2: The code then proceeds to prompt Llama-3 for one prompt included in prompts.txt 10 times.
Step 3: After Llama-3 responds 10 times, we store the responses on a txt file. 
Step 4: Steps 2 and 3 are repeated for every prompt.
'''



from meta_ai_api import MetaAI


messages = []
message = ''
with open('prompts.txt', 'r') as f:
    for line in f:
        if line.strip() == '':
            if message:
                messages.append(message.strip())  # Add message to list
                message = '' #restart message again from 0.
        else:
            message += line
    if message:
        messages.append(message.strip())  # Add last message


num_iter = 10
    

for i, each_message in enumerate(messages):
    # Open a file to store the responses
    with open('responsesPrompt' + str(i+1) + '.txt', 'w') as f:
    #Loop through and generate responses
        for j in range(num_iter):
            ai = MetaAI() # Reset conversation context
            response = ai.prompt(message=each_message) # Prompt and get a response
            response_message = response['message'] #get message only instead of message, sources, and media
            f.write(f'Output {j+1}:\n{response_message}\n\n')  # Write the response to the file
            print(f'Iteration {j+1}: {response_message}')

        print('All responses saved to responsesPrompt' + str(i+1) + '.txt')

print('Work Done')