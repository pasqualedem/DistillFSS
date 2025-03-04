import re

def remove_fields(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        bib_content = file.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as file:
        skip = False
        for line in bib_content:
            if re.match(r'\s*(abstract|file)\s*=\s*\{', line, re.IGNORECASE):
                skip = True
            if not skip:
                file.write(line)
            if skip and line.strip().endswith('},'):
                skip = False

if __name__ == "__main__":
    input_bib = "input.bib"  # Change this to your input bib file
    output_bib = "output.bib"  # Change this to your desired output file
    remove_fields(input_bib, output_bib)
    print(f"Processed file saved as {output_bib}")