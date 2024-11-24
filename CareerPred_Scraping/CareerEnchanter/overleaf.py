# Add this code snippet after generating the resume in main.py

import webbrowser

def redirect_to_overleaf(resume):
    # Assuming the Overleaf URL for templates is "https://www.overleaf.com/templates"
    overleaf_url = "https://www.overleaf.com/project/new/template/34599?id=533668126&latexEngine=pdflatex&mainFile=main.tex&templateName=Northeastern+University+COS+Faculty+CV+Template&texImage=texlive-full%3A2023.1"
    # Save the generated resume to a file
    # with open("a.tex", "w") as file:
    #     file.write(resume)
    # Redirect to Overleaf with the generated resume
    webbrowser.open_new_tab(overleaf_url)
    # webbrowser.open_new_tab("generated_resume.tex")

# Call this function after generating the resume
redirect_to_overleaf("generated_resume")