import re
import json

file_path = '/mnt/c/Users/Utente/Desktop/vol_gen.tagged.quality_rdx_fib.aug'

# label: at the start of the line
# then any amount of spaces (\s*)
# then a sequence of uppercase letters or underscores ([A-Z_]+)
# then the number in parentheses ((\d+))
pattern = re.compile(r"^label:\s*([A-Za-z_]+)\s*\((\d+)\)")

atria_dict = {}

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        match = pattern.match(line)
        if match:
            label = match.group(1)
            idx = int(match.group(2))
            atria_dict[idx] = label


with open("atria_dict_original.json", "w") as f:
    json.dump(atria_dict, f)