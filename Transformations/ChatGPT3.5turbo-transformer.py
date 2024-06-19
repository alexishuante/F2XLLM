'''
Step 1: prompts.txt must contain multiple prompts each one separated by a white space.
Step 2: The code then proceeds to prompt ChatGPT for one prompt included in prompts.txt 10 times.
Step 3: After ChatGPT responds 10 times, we store the responses on a txt file.
Step 4: Steps 2 and 3 are repeated for every prompt.
'''

from openai import OpenAI

client = OpenAI(api_key = ' REPLACE API KEY')  #REPLACE API KEY

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
    with open('Prompt' + str(i+1) + '_responses_gpt3.5turbo.txt', 'w') as f:
    #Loop through and generate responses
        for j in range(num_iter):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": each_message}]
            )
            response_content = response.choices[0].message.content
            f.write(f'Output {j+1}:\n{response_content}\n\n')  # Write the response to the file
            print(f'Iteration {j+1}: {response_content}')

        print('\n')
        print('All responses saved to Prompt' + str(i+1) + '_responses_gpt3.5turbo.txt')
        print('\n')

print('-------------------WORK DONE---------------------')
