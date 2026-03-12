with open("mpx/sdf_pretrain/data/dataset_generator.py", "r") as f:
    lines = f.readlines()
# keep removing lines from back if they contain """ or are empty
while len(lines) > 0 and ('\"\"\"' in lines[-1] or lines[-1].strip() == ""):
    lines.pop()
with open("mpx/sdf_pretrain/data/dataset_generator.py", "w") as f:
    f.writelines(lines)
