import tokenize
import io

def check_indentation(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if line.strip() and line[0] != ' ' and line[0] != '\t' and not line.startswith('#'):
            print(f"Line {i} not indented: {line}")
            continue
            
        # Check for indentation inconsistencies in a simple way
        if i > 1 and lines[i-2].strip() and lines[i-2].endswith(':'):
            prev_indent = len(lines[i-2]) - len(lines[i-2].lstrip())
            curr_indent = len(line) - len(line.lstrip())
            if curr_indent <= prev_indent and line.strip():
                print(f"Line {i} has indentation issue after colon on line {i-1}:")
                print(f"  Line {i-1}: {lines[i-2]}")
                print(f"  Line {i}: {line}")

if __name__ == "__main__":
    check_indentation("main.py") 