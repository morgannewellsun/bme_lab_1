
from typing import *
from collections import namedtuple

import numpy as np


Ion = namedtuple("Ion", "z P Ci Co")

def nernst(ion: Ion):
    """
    Calculates reversal membrane potential for an ion using the Nernst equation
    :param ion: Ion to calculate reversal potential for
    :return: Reversal potential in mV
    """
    return (25 / ion.z) * np.log(ion.Co / ion.Ci)

def ghk(ions: List[Ion]):
    """
    Calculates equilibrium membrane potential with multiple ions
    :param ions: A list of relevant ions
    :return: Equilibrium potential in mV
    """
    num = 0
    den = 0
    for ion in ions:
        num += ion.P * (ion.Co if ion.z > 0 else ion.Ci)
        den += ion.P * (ion.Ci if ion.z > 0 else ion.Co)
    return 25 * np.log(num/den)


def main():

    ion_K = Ion(z=1, P=1.0, Ci=155, Co=4)
    ion_Na = Ion(z=1, P=0.04, Ci=12, Co=145)
    ion_Ca = Ion(z=2, P=None, Ci=1e-4, Co=1.5)
    ion_Cl = Ion(z=-1, P=0.45, Ci=4, Co=120)

    # 1.1. Calculate the reversal potential of each ion
    E_K = nernst(ion_K)
    E_Na = nernst(ion_Na)
    E_Ca = nernst(ion_Ca)
    E_Cl = nernst(ion_Cl)
    print(f"E_K : {E_K:.3}")
    print(f"E_Na: {E_Na:.3}")
    print(f"E_Ca: {E_Ca:.3}")
    print(f"E_Cl: {E_Cl:.3}")

    # 1.2. Calculate the equilibrium potential with multiple ions
    V_m = ghk([ion_K, ion_Na, ion_Cl])
    print(f"V_m: {V_m:.3}")


if __name__ == "__main__":
    main()
