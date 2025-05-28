from utils import *
import quantities as pq
h = pq.Quantity(6.62607015* 10**-34, 'J*s')
c  = pq.Quantity(299792458, 'm/s')

def energy_kev_to_joule(energy_kev):
    """converts energy in kev to joules"""
    return energy_kev * 1.60217662e-16

def fresnel_calculator(energy_kev = None, lam = None, detector_pixel_size = None, distance_sample_detector = None):
    """calculates the fresnel number, the unit of energy must be in kev, and the unit of the other parameters must be in meters"""
    if energy_kev is not None:
        lam = 6.626 * 10**(-34) * 299792458 / energy_kev_to_joule(energy_kev)
    assert detector_pixel_size is not None, "detector_pixel_size must be given"
    assert distance_sample_detector is not None, "distance_sample_detector must be given"
    return detector_pixel_size**2/(lam*distance_sample_detector)

def eneryg_J(energy):
    """if an energy is given without the unit, it automatically assumes it's in keV"""
    if type(energy) == pq.quantity.Quantity:
        if energy.dimensionality == pq.Quantity(1, 'keV').dimensionality:
            energy = energy.rescale('J')
        elif energy.dimensionality == pq.Quantity(1, 'J').dimensionality:
            energy = energy
        elif energy.dimensionality == pq.Quantity(1, 'eV').dimensionality:
            energy = energy.rescale('J')
        elif energy.dimensionality == pq.Quantity(1, 'm').dimensionality:
            wavelength = energy
            energy = energy_from_wavelength(wavelength)
    else:
        if type(energy) == str:
            energy = float(energy)
        energy = pq.Quantity(energy, 'keV').rescale('J')
    return energy

def wavelength_m(lam):
    if type(lam) == pq.quantity.Quantity:
        if lam.dimensionality == pq.Quantity(1, 'm').dimensionality:
            lam = lam.rescale('m')
        elif lam.dimensionality == pq.Quantity(1, 'nm').dimensionality:
            lam = lam.rescale('m')
        elif lam.dimensionality == pq.Quantity(1, 'A').dimensionality:
            lam = lam.rescale('m')
        elif lam.dimensionality == pq.Quantity(1, 'keV').dimensionality:
            lam = lam.rescale('m')
    else:
        lam = pq.Quantity(lam, 'm')
    return lam

def wavelength_from_energy(energy):
    """if an energy is given without the unit, it automatically assumes it's in keV"""
    energy = eneryg_J(energy)
    print("energy", energy)
    return h*c/energy

def energy_from_wavelength(lam):
    h = pq.Quantity(6.62607015* 10**-34, 'J*s')
    c  = pq.Quantity(299792458, 'm/s')
    lam = wavelength_m(lam)
    energy = h*c/lam
    return energy.rescale('keV')

def wave_number(energy):
    """if an energy is given without the unit, it automatically assumes it's in keV"""
    lam = wavelength_from_energy(energy)
    wavenumber = 2*np.pi/lam
    return wavenumber

def energy_from_wave_number(wave_number):
    lam = 2*np.pi/wave_number
    return energy_from_wavelength(lam)

def fresnel_calc(energy, z, pv):
    """z and pv have to in meters"""
    if energy is None or z is None or pv is None:
        return None
    if type(energy) is not list and type(z) is not list and type(pv) is not list:
        wavelength = wavelength_from_energy(eneryg_J(energy)).magnitude
        fresnel_number = pv**2/(wavelength*z) 
    else:
        energy = [energy] if type(energy) is not list else energy
        wavelength = [wavelength_from_energy(eneryg_J(ener)).magnitude for ener in energy]
        z = [z] if type(z) is not list else z
        pv = [pv] if type(pv) is not list else pv
        fresnel_number = []
        for i in range(len(energy)):
            for j in range(len(z)):
                for k in range(len(pv)):
                    fresnel_number.append(pv[k]**2/(wavelength[i]*z[j]))
    return  fresnel_number

def ffactors(px, py, energy = None, zs = None, pv = None, fresnel_number = None):
    if fresnel_number == None:
        fresnel_number = fresnel_calc(energy, zs, pv)
    
    freq_x = fftfreq(px)
    freq_y = fftfreq(py)
    xi, eta = np.meshgrid(freq_x, freq_y)
    xi = xi.astype('float32')
    eta = eta.astype('float32')
    if type(zs) is not list:
        frequ_prefactors = 2 * np.pi  / fresnel_number
        h = np.exp(- 1j * frequ_prefactors * (xi ** 2 + eta ** 2) / 2)
    else:
        frequ_prefactors = [2 * np.pi  / fresnel_number[i] for i in range(len(zs))]
        h = [((np.exp(- 1j * frequ_prefactors[i] * (xi ** 2 + eta ** 2) / 2)).T).astype('complex64') for i in range(len(zs))]
    return h.T

def get_fresnel_from_cone(**kwargs):
    z01, z02, energy, px = kwargs['z01'], kwargs['z02'], kwargs['energy'], kwargs['pv']
    z12 = z02 - z01
    M = (z12 + z01) / z01
    dx_eff = px / M
    z_eff = z12 / M
    lam = 1.2389 / energy
    fr_eff = dx_eff ** 2 / lam / z_eff
    return fr_eff
