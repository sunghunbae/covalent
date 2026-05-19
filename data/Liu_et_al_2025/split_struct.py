import re
import json

input_file = "Liu_et_al_2025_struct.txt"

atom_line = re.compile(r'^[A-Z][a-z]?\s+[0-9.-]+\s+[0-9.-]+\s+[0-9.-]+$')
frequency_line = re.compile(r'[01] imaginary freq.+')
energy_line = re.compile(r'(?P<key>(SP|U|H|G))\s+=\s+(?P<value>[\d.-]+)')

with open(input_file) as f:
    lines = f.readlines()

energies = {}
blocks = []
current_name = None
current_atoms = []
current_energy = {}


for line in lines:
    stripped = line.strip()
    if not stripped:
        continue

    # detect new block (name line)
    if not (atom_line.match(stripped) or energy_line.match(stripped) or "imaginary" in stripped):
        if current_name and current_atoms and current_energy:
            blocks.append((current_name, current_atoms, current_energy))
            energies[current_name] = current_energy
        # set up for new block
        current_name = stripped.replace(" ", "_").replace("-", "_")
        current_atoms = []
        current_energy = {}
        continue

    if atom_line.match(stripped):
        current_atoms.append(stripped)

    if energy_line.match(stripped):
        try:
            match = energy_line.match(stripped)
            k = match.group('key')
            v = match.group('value')
            current_energy[k] = float(v)
        except (ValueError, AttributeError):
            print(stripped, "is not a valid metadata line")


# add lastls
if current_name and current_atoms:
    blocks.append((current_name, current_atoms, current_energy))


# write .json file with energies
with open("energies.json", "w") as f:
    json.dump(energies, f, indent=2)


# write xyz files
for i, (name, atoms, metadata) in enumerate(blocks, 1):
    filename = f"{name}.xyz"
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        metadata_str = ",".join([f"{k}={v}" for k,v in metadata.items()])
        f.write(f"{name} ({metadata_str})\n")
        f.write("\n".join(atoms))