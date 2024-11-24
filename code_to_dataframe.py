import os
import pandas as pd
directory='CareerPred_Scraping'
def get_python_code(directory):
    code_list = []
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    code_list.append(code)
                    file_paths.append(file_path)
    return code_list, file_paths

# Step 3: Get file paths.  CodeBERT embedding generation is removed.
code_list, file_paths = get_python_code(directory)

# Step 4: Create DataFrame with file paths and plagiarism labels.
df = pd.DataFrame({'code':code_list, 'label': 'plagiarised'}) # All files are labeled as 'Plagiarised'

# Save DataFrame to Excel file.
df.to_excel('code_labels.xlsx', index=False)

print("File paths and plagiarism labels saved to 'file_paths_plagiarism_labels.xlsx'")
print(f"DataFrame size: {df.shape}")
