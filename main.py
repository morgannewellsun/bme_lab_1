
import argparse

import numpy as np
from matplotlib import pyplot as plt


def pprint(name, value, decimals=4):
    print(f"{name:8} = {value:8.{decimals}f}")


def find_local_maxima(a):
    a = np.array(a)
    signs_of_deltas = (a[1:] - a[:-1]) > 0
    is_local_maximum = np.logical_and(
        np.concatenate(([True], signs_of_deltas)),
        np.concatenate((np.logical_not(signs_of_deltas), [True])))
    return list(np.where(is_local_maximum)[0])


class Ion:

    def __init__(self, charge, permeability, concentration_inner, concentration_outer, g_bar):
        self.charge = charge
        self.permeability = permeability
        self.concentration_inner = concentration_inner
        self.concentration_outer = concentration_outer
        self.g_bar = g_bar
        self.nernst_potential = self.nernst()

    def nernst(self):
        return (25 / self.charge) * np.log(self.concentration_outer / self.concentration_inner)

    @staticmethod
    def ghk(ions):
        num = 0
        den = 0
        for ion in ions:
            num += ion.permeability * (ion.concentration_outer if ion.charge > 0 else ion.concentration_inner)
            den += ion.permeability * (ion.concentration_inner if ion.charge > 0 else ion.concentration_outer)
        return 25 * np.log(num/den)


class Channel:

    def __init__(self, alpha_A, alpha_V_half, alpha_k, beta_A, beta_V_half, beta_k):
        self.alpha_A = alpha_A
        self.alpha_V_half = alpha_V_half
        self.alpha_k = alpha_k
        self.beta_A = beta_A
        self.beta_V_half = beta_V_half
        self.beta_k = beta_k

    def alpha(self, membrane_potential):
        return (self.alpha_A * (membrane_potential - self.alpha_V_half)) / (1 - np.exp(-1 * (membrane_potential - self.alpha_V_half) / self.alpha_k))

    def beta(self, membrane_potential):
        return (-1 * self.beta_A * (membrane_potential - self.beta_V_half)) / (1 - np.exp((membrane_potential - self.beta_V_half) / self.beta_k))

    def resting_point(self, resting_membrane_potential):
        alpha = self.alpha(resting_membrane_potential)
        beta = self.beta(resting_membrane_potential)
        return alpha / (alpha + beta)

    def rate_of_change(self, membrane_potential, channel_value):
        return (self.alpha(membrane_potential) * (1 - channel_value)) - (self.beta(membrane_potential) * channel_value)


class Neuron:

    def __init__(self):
        self.ion_K = Ion(charge=1, permeability=1.0, concentration_inner=155, concentration_outer=4, g_bar=50)
        self.ion_Na = Ion(charge=1, permeability=0.04, concentration_inner=12, concentration_outer=145, g_bar=100)
        self.ion_Ca = Ion(charge=2, permeability=None, concentration_inner=1e-4, concentration_outer=1.5, g_bar=None)
        self.ion_Cl = Ion(charge=-1, permeability=0.45, concentration_inner=4, concentration_outer=120, g_bar=None)
        self.channel_m = Channel(alpha_A=0.182, alpha_V_half=-35, alpha_k=9, beta_A=0.124, beta_V_half=-35, beta_k=9)
        self.channel_h = Channel(alpha_A=0.024, alpha_V_half=-50, alpha_k=5, beta_A=0.0091, beta_V_half=-75, beta_k=5)
        self.channel_n = Channel(alpha_A=0.02, alpha_V_half=20, alpha_k=9, beta_A=0.002, beta_V_half=20, beta_k=9)
        self.leak_g = 0.5
        self.leak_nernst_potential = -72.5
        self.membrane_capacitance = 1.0
        self.init_membrane_potential = Ion.ghk([self.ion_K, self.ion_Na, self.ion_Cl])
        self.init_m = self.channel_m.resting_point(self.init_membrane_potential)
        self.init_h = self.channel_h.resting_point(self.init_membrane_potential)
        self.init_n = self.channel_n.resting_point(self.init_membrane_potential)

    def run(self, stim_magnitude, stim_time, warmup_time, run_time, delta_time=0.01):

        final_t = warmup_time + stim_time + run_time
        init_current_stim = (
                [0] * int(np.round(warmup_time / delta_time))
                + [stim_magnitude] * int(np.round(stim_time / delta_time)))

        values_t = [0]
        values_membrane_potential = [self.init_membrane_potential]
        values_m = [self.init_m]
        values_h = [self.init_h]
        values_n = [self.init_n]
        values_current_stim = [0]
        while True:
            current_Na = (
                    self.ion_Na.g_bar * np.power(values_m[-1], 3) * (1 - values_h[-1])
                    * (values_membrane_potential[-1] - self.ion_Na.nernst_potential))
            current_K = (
                    self.ion_K.g_bar * np.power(values_n[-1], 4)
                    * (values_membrane_potential[-1] - self.ion_K.nernst_potential))
            current_leak = self.leak_g * (values_membrane_potential[-1] - self.leak_nernst_potential)
            try:
                current_stim = init_current_stim.pop(0)
            except IndexError:
                current_stim = 0
            delta_membrane_potential = (
                current_stim - (current_Na + current_K + current_leak)) * (delta_time / self.membrane_capacitance)
            delta_m = self.channel_m.rate_of_change(values_membrane_potential[-1], values_m[-1]) * delta_time
            delta_h = self.channel_h.rate_of_change(values_membrane_potential[-1], values_h[-1]) * delta_time
            delta_n = self.channel_n.rate_of_change(values_membrane_potential[-1], values_n[-1]) * delta_time
            values_t.append(values_t[-1] + delta_time)
            values_membrane_potential.append(values_membrane_potential[-1] + delta_membrane_potential)
            values_m.append(values_m[-1] + delta_m)
            values_h.append(values_h[-1] + delta_h)
            values_n.append(values_n[-1] + delta_n)
            values_current_stim.append(current_stim)
            if values_t[-1] >= final_t:
                return {
                    "t": values_t,
                    "I_stim": values_current_stim,
                    "V_m": values_membrane_potential,
                    "m": values_m,
                    "h": values_h,
                    "n": values_n,
                }

    @staticmethod
    def plot(t, I_stim, V_m, m, h, n, show=True, output_filepath=None):
        plt.rcParams["figure.figsize"] = [6, 6]
        plt.rcParams["figure.autolayout"] = True
        fig, axes = plt.subplots(3, 1, sharex="all")
        axes[0].set_ylabel(r"$I_{stim}$  $(\mu A/cm^2)$")
        axes[0].set_ylim(-10, 310)
        axes[0].plot(t, I_stim, color="tab:blue")
        axes[1].set_ylabel(r"$V_{m}$  $(mV)$")
        axes[1].set_ylim(-90, 90)
        axes[1].plot(t, V_m, color="tab:orange")
        axes[2].set_xlabel("time (ms)")
        axes[2].set_ylabel("")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].plot(t, m, label="m", color="tab:green")
        axes[2].plot(t, h, label="h", color="tab:red")
        axes[2].plot(t, n, label="n", color="tab:purple")
        axes[2].legend(loc="upper left")
        plt.savefig(output_filepath) if output_filepath is not None else None
        plt.show() if show else None
        plt.close()


def main(output_dir):

    # Initialize the neuron simulation, and print out Nernst and equilibrium properties
    neuron = Neuron()
    print("\nNernst Potentials (mV): ")
    pprint("K", neuron.ion_K.nernst_potential)
    pprint("Na", neuron.ion_Na.nernst_potential)
    pprint("Ca", neuron.ion_Ca.nernst_potential)
    pprint("Cl", neuron.ion_Cl.nernst_potential)
    print("\nEquilibrium Potential (mV): ")
    pprint("", neuron.init_membrane_potential)
    print("\nChannel resting points: ")
    pprint("m", neuron.init_m)
    pprint("h", neuron.init_h)
    pprint("n", neuron.init_n)

    # Run and plot single action potential with various stimulus magnitudes
    for stim_magnitude in range(100, 301, 25):
        output_dict = neuron.run(stim_magnitude=stim_magnitude, stim_time=0.10, warmup_time=2.00, run_time=10.00)
        Neuron.plot(
            **output_dict,
            show=False,
            output_filepath=output_dir + f"{stim_magnitude}.png")

    # Identify the stim_magnitude threshold for firing
    iterations = 256
    voltage_to_check = 40
    stim_magnitude_bound_upper = 175
    stim_magnitude_bound_lower = 150
    output_dict_upper = None
    output_dict_lower = None
    for _ in range(iterations):
        stim_magnitude = (stim_magnitude_bound_lower + stim_magnitude_bound_upper) / 2
        output_dict = neuron.run(stim_magnitude=stim_magnitude, stim_time=0.10, warmup_time=2.00, run_time=60.00)
        if np.max(output_dict["V_m"]) > voltage_to_check:
            stim_magnitude_bound_upper = stim_magnitude
            output_dict_upper = output_dict
        else:
            stim_magnitude_bound_lower = stim_magnitude
            output_dict_lower = output_dict
    Neuron.plot(
        **output_dict_upper,
        show=False,
        output_filepath=output_dir + f"upper.png")
    Neuron.plot(
        **output_dict_lower,
        show=False,
        output_filepath=output_dir + f"lower.png")
    voltage_threshold_bound_upper = output_dict_upper["V_m"][find_local_maxima(output_dict_upper["V_m"])[0]]
    voltage_threshold_bound_lower = output_dict_lower["V_m"][find_local_maxima(output_dict_lower["V_m"])[0]]
    print("\nThreshold Stim Current (uA/cm2):")
    pprint("UBound", stim_magnitude_bound_upper, decimals=16)
    pprint("LBound", stim_magnitude_bound_lower, decimals=16)
    print("\nThreshold Voltage (mV):")
    pprint("UBound", voltage_threshold_bound_upper, decimals=16)
    pprint("LBound", voltage_threshold_bound_lower, decimals=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="where to output plot images")
    args = parser.parse_args()
    main(args.output_dir)
