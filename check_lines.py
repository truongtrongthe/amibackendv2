def check_specific_lines():
    with open("main.py", "r") as f:
        lines = f.readlines()
    
    # Check line 303 and surrounding lines
    for i in range(300, 310):
        print(f"Line {i + 1}: '{lines[i].rstrip()}'")
    
    print("\n" + "="*40 + "\n")
    
    # Check line 403 and surrounding lines
    for i in range(400, 410):
        print(f"Line {i + 1}: '{lines[i].rstrip()}'")

if __name__ == "__main__":
    check_specific_lines() 