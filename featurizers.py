import numpy as np
from itertools import groupby
from scipy.signal import gaussian
from pymatgen import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator


def one_hot_chem_comp(structure):
    d = {}
    for key, group in groupby(structure.atomic_numbers):
        d.update({key:len(list(group))})

    one_hot_chem = np.zeros(101)
    one_hot_chem[list(d.keys())] = list(d.values())
    one_hot_chem = (one_hot_chem / sum(list(d.values())))[1:]

    return one_hot_chem


def peaks_with_gauss(structure, degree_min=0.0, degree_max=90.0, step=0.05):
    spectrum = XRDCalculator().get_pattern(structure)
    positions, intensities = spectrum.x, spectrum.y
    num_points = int((degree_max - degree_min) / step + 1)
    num_positions = [int(position/step + 1) for position in positions]
        
    def peak(num_points, position, intensity, M=31, std=3):
        k = np.zeros(num_points)
        mn = int(position - (M - 1) / 2)
        mx = int(position + (M - 1) / 2 + 1)
        diff = mx if mx <= num_points else num_points - mn if mn >= 0 else 0
        k[mn if mn >= 0 else 0:mx if mx <= num_points else num_points] = intensity * gaussian(M, std=std)[:diff]
        return k
    
    final_intensity = sum([peak(num_points, p, i) for p, i in zip(num_positions, intensities)])
    
    return np.linspace(degree_min, degree_max, num_points), final_intensity
