"""SILAM VMR to concentration conversion utilities.

This module contains conversion functions and constants for converting
SILAM volume mixing ratio (VMR) data to mass concentrations for AQI calculations.
"""

# Unit conversion constants
KG_M3_TO_UG_M3 = 1e9  # Convert kg/m³ to µg/m³

# Molar masses (kg/mole)
MOLAR_MASS_AIR = 0.02897  # kg/mole (dry air)
# Gas species molar masses from SILAM metadata
MOLAR_MASS_O3 = 0.048  # kg/mole
MOLAR_MASS_NO2 = 0.046  # kg/mole
MOLAR_MASS_SO2 = 0.064  # kg/mole
MOLAR_MASS_CO = 0.028  # kg/mole


def convert_vmr_to_concentration(vmr, air_density, molar_mass):
    """
    Convert volume mixing ratio (VMR) to mass concentration in µg/m³.

    Args:
        vmr: Volume mixing ratio in mole/mole (mole pollutant per mole air)
        air_density: Air density in kg/m³
        molar_mass: Molar mass of the pollutant in kg/mole

    Returns:
        Concentration in µg/m³

    Formula:
        concentration (µg/m³) = VMR (mole/mole) * air_density (kg/m³) *
                                (molar_mass_pollutant / molar_mass_air) * KG_M3_TO_UG_M3

    Note: SILAM's vmr_*_gas variables are true volume mixing ratios (mole/mole),
    as confirmed by the SILAM metadata (units: mole/mole, silam_amount_unit: mole).
    The conversion requires both the pollutant's molecular weight and air's molecular weight.
    """
    # Volume mixing ratio conversion to mass concentration
    # VMR is mole_pollutant/mole_air
    # mass_concentration = VMR * (air_density / molar_mass_air) * molar_mass_pollutant
    return vmr * air_density * (molar_mass / MOLAR_MASS_AIR) * KG_M3_TO_UG_M3
