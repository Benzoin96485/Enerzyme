import requests
from typing import Dict, Any, Tuple
from .data import SIMPLE_PERIODIC_TABLE


def simple_xyz_supplier(xyz_file: str) -> Dict[str, Any]:
    with open(xyz_file, "r") as f:
        lines = f.readlines()
    N_atoms = int(lines[0])
    Ra = []
    Za = []
    for i, line in enumerate(lines[2:2+N_atoms]):
        atom_type, x, y, z = line.split()
        Ra.append([float(x), float(y), float(z)])
        Za.append(SIMPLE_PERIODIC_TABLE[atom_type])
    return {
        "Za": Za,
        "Ra": Ra,
        "N": N_atoms,
    }


def parse_orca_input(orca_input_file: str) -> Tuple[Dict[str, Any], str]:
    if not orca_input_file.endswith(".extinp.tmp"):
        raise ValueError(f"Unsupported format: {orca_input_file}")
    basename = orca_input_file.replace(".extinp.tmp", "")
    with open(orca_input_file, "r") as f:
        lines = f.readlines()
    xyz_file = lines[0].strip()
    charge = int(lines[1].strip())
    multiplicity = int(lines[2].strip())
    features = simple_xyz_supplier(xyz_file)
    features["Q"] = charge
    features["S"] = multiplicity - 1
    return features, basename


def write_orca_output(results: Dict[str, Any], basename: str):
    with open(f"{basename}_EXT.engrad", "w") as f:
        f.write(f'''
#
# Number of atoms: must match the XYZ
#
{len(results["outputs"]["Fa"])}
#
# The current total energy in Eh
#
{results["outputs"]["E"] / results["units"]["Hartree_in_E"]}
#
# The current gradient in Eh/bohr: Atom1X, Atom1Y, Atom1Z, Atom2X, etc.
#
''')
        for atom_Fa in results["outputs"]["Fa"]:
            for component in atom_Fa:
                f.write(f"{component / results["units"]["Hartree_in_E"] * results["units"]["Bohr_in_R"]}\n")


def FFRequest(
    url: str,
    format: str,
    input_file: str,
    model_key: str=''
):
    if format == "ORCA":
        features, basename = parse_orca_input(input_file)
        info = {"features": features}
        info["input_file"] = input_file
        if model_key:
            info["model_key"] = model_key
        response = requests.post('http://' + url + "/calculate", json=info)
        response.raise_for_status()
        result = response.json()
        write_orca_output(result, basename)
    else:
        raise ValueError(f"Unsupported format: {format}")
