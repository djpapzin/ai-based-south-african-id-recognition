import json
import re

def convert_to_notebook(py_file, ipynb_file):
    """Convert a Python file with cell markers to a Jupyter notebook."""
    
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Initialize notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Split content into cells
    cell_pattern = r'# %% \[(.*?)\](.*?)(?=# %% |$)'
    matches = re.finditer(cell_pattern, content, re.DOTALL)
    
    # Process each cell
    for match in matches:
        cell_type = match.group(1)
        cell_content = match.group(2).strip()
        
        if cell_type == 'markdown':
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [cell_content]
            }
        elif cell_type == 'code':
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [cell_content]
            }
        
        notebook["cells"].append(cell)
    
    # Save as notebook
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    convert_to_notebook(
        'sa_id_book_training_notebook.py',
        'SA_ID_Book_Training_Detectron2_V2.ipynb'
    ) 