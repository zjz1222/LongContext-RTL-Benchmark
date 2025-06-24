from model import ChatModel
import os

annotation_repo = 'systolic_array'

ref_module_path = f"./benchmark/{annotation_repo}/ref"
all_modules = os.listdir(ref_module_path)
annotation_modules = [entry for entry in all_modules if os.path.isfile(os.path.join(ref_module_path, entry))]


llm = ChatModel(model_name = 'gpt-4.1-mini', temperature = 0.0)
annotation_system_prompt = """
You will be given a Verilog Module, please refer to the Example to generate a natural language description of the Verilog Module for the given Verilog Module: it consists of the following 5 parts
1. Module name: the name of the given design.
2. Functional Description: a general abstraction of the function of the design.
3. Input ports: description of the input signals.
4. Output ports: description of the output signals.
5. Implementation: implementation details of the Verilog design, the more detailed the better.
"""

if __name__ == "__main__":

    for annotation_module in annotation_modules:
        print(annotation_module)

        design_path = f"./benchmark/{annotation_repo}/ref/{annotation_module}"
        description_path = f"./benchmark/{annotation_repo}/description/{annotation_module}"
        description_path = description_path.replace(".v", ".txt")

        example_name_list = ['alu', 'instr_reg']   
        
        example_prompt = ""
        for index, example_name in enumerate(example_name_list):
            with open(f"./annotation/examples/{example_name}.txt", encoding = 'utf-8', errors = 'ignore') as file: 
                example_description = file.read()
            with open(f"./annotation/examples/{example_name}.v", encoding = 'utf-8', errors = 'ignore') as file:
                example_design = file.read()
            
            example_prompt += f"### Example {index + 1} ###\n\n"
            example_prompt += f"[Input]:\n```verilog\n{example_design}```\n\n"
            example_prompt += f"[Output]:\n{example_description}\n\n"
            example_prompt += f"### Example End ###\n\n"

        with open(design_path, "r", encoding = 'utf-8', errors = 'ignore') as file: design = file.read()

        system_prompt = annotation_system_prompt + f"\n\n{example_prompt}"
        user_prompt = f"[Input]:\n```verilog\n{design}```\n\n[Output]:\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        description = llm.generate(messages = messages)
        with open(description_path, 'w') as file: file.write(description)
