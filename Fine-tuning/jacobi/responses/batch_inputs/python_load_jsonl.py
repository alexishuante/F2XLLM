import json
import pandas as pd


lines = []
with open(r'/home/hally/ornl/openai-training/test/batch_results/file_output_fortran_openACC_to_C_openMP.jsonl') as f:
    lines = f.read().splitlines()

line_dicts = [json.loads(line) for line in lines]
df_final = pd.DataFrame(line_dicts)

print(df_final)
print(df[:2])