python
from pymol import cmd

# reference object name
ref = "hypothetical_protein__YP_004934016__Human_papillomavirus_126__1055684"

# C-terminal selector (last 50 residues)
cterm = "resi -50:"

for obj in cmd.get_object_list():
    if obj != ref:
        num_residues = cmd.count_atoms(f"{obj} and name CA")
        begin=max(1, num_residues - 50)
        cmd.super(
            f"{obj} and resi {begin}-{num_residues}",
            f"{ref} and resi 115-170"
        )
        cmd.disable("all")
        cmd.enable(obj)
        cmd.show("cartoon", obj)
        cmd.ray(1200, 1200)
        cmd.png(f"E4_figures/{obj}.png", dpi=350)
python end