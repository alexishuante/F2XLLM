from openai import OpenAI

client = OpenAI(api_key= 'REPLACE API KEY HERE')  # REPLACE API KEY

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
    with open(f'Prompt{prompt_number}_responses_gpt3.5turbo.txt', 'w') as response_file:
        # Loop through and generate responses
        for j in range(num_iter):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": each_message}]
            )
            response_content = response.choices[0].message.content
            response_file.write(f'--------------- Output {j + 1} ---------------\n{response_content}\n\n')  # Write the response to the file

            print(f'\n--------------- Iteration {j + 1} ---------------\n\n {response_content}')

        print('\n')
        print(f'All responses saved to Prompt{prompt_number}_responses_gpt3.5turbo.txt')
        print('\n')

print('-------------------WORK DONE---------------------')
