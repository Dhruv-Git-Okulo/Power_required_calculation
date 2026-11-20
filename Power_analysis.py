import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER INPUTS
# ============================================================

# Aircraft parameters
m = 10.0                  # kg
Cl = 1.10                  # -
b = 5                   # m
S = 2.0                  # m^2
Q = 1.2                    # Form factor multiplier
t_c = 0.1214               # Maximum thickness / chord
e = 0.85                   # Oswald efficiency

# Reference climb velocity
# Kept for reference only. CD0 is now calculated at every
# velocity in the velocity sweep.
V_climb = 15.0             # m/s

# System efficiencies
Eff_bat = 0.90
Eff_GB = 0.85
Eff_motor = 0.95
Eff_prop = 0.85

# Velocity sweep
V_max = 45.0               # m/s
N_V = 500

# Output directory
OUTPUT_DIR = "Power_Required_Analysis"


# ============================================================
# ATMOSPHERIC DATA
# ============================================================

# Altitude : (Density [kg/m^3], Dynamic viscosity [kg/(m.s)])
ATMOSPHERE = {
    100:     (1.14e+00, 1.86e-05),
    500:     (1.10e+00, 1.85e-05),
    1000:    (1.05e+00, 1.83e-05),
    1500:    (1.01e+00, 1.81e-05),
    2000:    (9.58e-01, 1.80e-05),
    2500:    (9.14e-01, 1.78e-05),
    3000:    (8.70e-01, 1.77e-05),
    3500:    (8.29e-01, 1.75e-05),
    4000:    (7.88e-01, 1.73e-05),
    4500:    (7.50e-01, 1.72e-05),
    5000:    (7.13e-01, 1.70e-05),
    5500:    (6.77e-01, 1.69e-05),
    6000:    (6.43e-01, 1.67e-05),
    6500:    (6.10e-01, 1.65e-05),
    7000:    (5.78e-01, 1.64e-05),
    7500:    (5.47e-01, 1.62e-05),
    8000:    (5.18e-01, 1.60e-05),
    8500:    (4.90e-01, 1.58e-05),
    9000:    (4.64e-01, 1.57e-05),
    9500:    (4.38e-01, 1.55e-05),
    10000:   (4.13e-01, 1.53e-05),
    10500:   (3.90e-01, 1.52e-05),
    11000:   (3.67e-01, 1.50e-05),
    11500:   (3.46e-01, 1.48e-05),
    12000:   (3.25e-01, 1.46e-05),
    12500:   (3.06e-01, 1.45e-05),
    13000:   (2.87e-01, 1.43e-05),
    13500:   (2.70e-01, 1.41e-05),
    14000:   (2.53e-01, 1.39e-05),
    14500:   (2.37e-01, 1.37e-05),
    15000:   (2.21e-01, 1.35e-05),
    15500:   (2.07e-01, 1.34e-05),
    16000:   (1.93e-01, 1.32e-05),
    16500:   (1.76e-01, 1.32e-05),
    17000:   (1.61e-01, 1.33e-05),
    17500:   (1.47e-01, 1.34e-05),
    18000:   (1.34e-01, 1.34e-05),
    18500:   (1.23e-01, 1.35e-05),
    19000:   (1.12e-01, 1.36e-05),
    19500:   (1.03e-01, 1.36e-05),
    20000:   (9.43e-02, 1.37e-05),
    20500:   (8.64e-02, 1.38e-05),
    21000:   (7.92e-02, 1.38e-05),
    21500:   (7.27e-02, 1.39e-05),
    22000:   (6.67e-02, 1.40e-05),
    22500:   (6.12e-02, 1.40e-05),
    23000:   (5.62e-02, 1.41e-05),
    23500:   (5.17e-02, 1.41e-05),
}


# ============================================================
# BASIC AIRCRAFT PARAMETERS
# ============================================================

g = 9.81

MTOW = m * g

MAC = S / b

AR = (b ** 2) / S

eff_sys = (
    Eff_bat
    * Eff_GB
    * Eff_motor
    * Eff_prop
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# POWER CALCULATION
# ============================================================

def calculate_power(
    altitude,
    rho,
    mu,
    V_max,
    N_V
):
    """
    Calculate aerodynamic and electrical power requirements
    over the velocity range from stall speed to V_max.

    Returns:
        df
        V_stall
        V_min_power
        P_min_power
        CD0_min_power
        Re_min_power
        Cf_min_power
    """

    # --------------------------------------------------------
    # Stall speed
    # --------------------------------------------------------

    V_stall = np.sqrt(
        (2.0 * MTOW)
        / (rho * S * Cl)
    )

    # --------------------------------------------------------
    # Check whether V_max is above stall
    # --------------------------------------------------------

    if V_stall >= V_max:

        raise ValueError(
            f"At {altitude} m, stall speed "
            f"({V_stall:.2f} m/s) is greater than or equal to "
            f"V_max ({V_max:.2f} m/s)."
        )

    # --------------------------------------------------------
    # Velocity sweep
    # --------------------------------------------------------

    V = np.linspace(
        V_stall,
        V_max,
        N_V
    )

    # --------------------------------------------------------
    # Reynolds number
    # --------------------------------------------------------

    Re = (
        rho
        * V
        * MAC
        / mu
    )

    # --------------------------------------------------------
    # Laminar Blasius skin friction coefficient
    # --------------------------------------------------------

    Cf = 1.32824 / np.sqrt(Re)

    # --------------------------------------------------------
    # Hoerner form factor
    # --------------------------------------------------------

    FF = (
        1.0
        + 2.0 * t_c
        + 60.0 * t_c ** 4
    )

    # --------------------------------------------------------
    # Wetted area
    # --------------------------------------------------------

    Swet = 2.0 * S

    # --------------------------------------------------------
    # Velocity-dependent zero-lift parasite drag coefficient
    # --------------------------------------------------------

    CD0 = (
        Swet
        * Q
        * Cf
        * FF
        / S
    )

    # --------------------------------------------------------
    # Induced drag factor
    # --------------------------------------------------------

    k = 1.0 / (
        np.pi
        * e
        * AR
    )

    # --------------------------------------------------------
    # Aerodynamic drag
    # --------------------------------------------------------

    q_dynamic = (
        0.5
        * rho
        * V ** 2
    )

    D_parasite = (
        q_dynamic
        * S
        * CD0
    )

    D_induced = (
        2.0
        * MTOW ** 2
        * k
        / (
            rho
            * V ** 2
            * S
        )
    )

    D_total = (
        D_parasite
        + D_induced
    )

    # --------------------------------------------------------
    # Aerodynamic power
    # --------------------------------------------------------

    P_parasite_aero = (
        D_parasite
        * V
    )

    P_induced_aero = (
        D_induced
        * V
    )

    P_total_aero = (
        P_parasite_aero
        + P_induced_aero
    )

    # --------------------------------------------------------
    # Electrical power
    # --------------------------------------------------------

    P_parasite = (
        P_parasite_aero
        / eff_sys
    )

    P_induced = (
        P_induced_aero
        / eff_sys
    )

    P_total = (
        P_total_aero
        / eff_sys
    )

    # --------------------------------------------------------
    # Find minimum power
    # --------------------------------------------------------

    min_index = np.argmin(P_total)

    V_min_power = V[min_index]

    P_min_power = P_total[min_index]

    CD0_min_power = CD0[min_index]

    Re_min_power = Re[min_index]

    Cf_min_power = Cf[min_index]

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame({

        "Altitude_m":
            altitude,

        "Density_kgm3":
            rho,

        "Dynamic_Viscosity_kgms":
            mu,

        "Velocity_mps":
            V,

        "Stall_Speed_mps":
            V_stall,

        "Reynolds_Number":
            Re,

        "Cf":
            Cf,

        "CD0":
            CD0,

        "Dynamic_Pressure_Pa":
            q_dynamic,

        "Parasite_Drag_N":
            D_parasite,

        "Induced_Drag_N":
            D_induced,

        "Total_Drag_N":
            D_total,

        "Parasite_Power_Aero_W":
            P_parasite_aero,

        "Induced_Power_Aero_W":
            P_induced_aero,

        "Total_Power_Aero_W":
            P_total_aero,

        "Parasite_Power_W":
            P_parasite,

        "Induced_Power_W":
            P_induced,

        "Total_Electrical_Power_W":
            P_total,

    })

    return (
        df,
        V_stall,
        V_min_power,
        P_min_power,
        CD0_min_power,
        Re_min_power,
        Cf_min_power
    )


# ============================================================
# PLOT 1
# TOTAL ELECTRICAL POWER VS AIRSPEED
# ALL ALTITUDES
# ============================================================

def create_combined_power_plot(all_data):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for df in all_data:

        altitude = df["Altitude_m"].iloc[0]

        ax.plot(
            df["Velocity_mps"],
            df["Total_Electrical_Power_W"] / 1000,
            linewidth=1.5,
            label=f"{int(altitude):,} m"
        )

    ax.set_xlabel(
        "Airspeed (m/s)",
        fontsize=12
    )

    ax.set_ylabel(
        "Electrical Power Required (kW)",
        fontsize=12
    )

    ax.set_title(
        "Total Electrical Power Required vs Airspeed",
        fontsize=14
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        fontsize=7,
        ncol=3
    )

    plt.tight_layout()

    filepath = os.path.join(
        OUTPUT_DIR,
        "01_Total_Power_vs_Airspeed_All_Altitudes.png"
    )

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# PLOT 2
# 3D MINIMUM POWER
# ============================================================

def create_3d_minimum_power_plot(minimum_results):

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    altitude = minimum_results[
        "Altitude_m"
    ]

    V_min = minimum_results[
        "Minimum_Power_Airspeed_mps"
    ]

    P_min = minimum_results[
        "Minimum_Electrical_Power_kW"
    ]

    ax.scatter(
        V_min,
        altitude,
        P_min,
        s=40
    )

    ax.set_xlabel(
        "Minimum-Power Airspeed (m/s)",
        fontsize=11
    )

    ax.set_ylabel(
        "Altitude (m)",
        fontsize=11
    )

    ax.set_zlabel(
        "Minimum Electrical Power (kW)",
        fontsize=11
    )

    ax.set_title(
        "Minimum Electrical Power with Altitude",
        fontsize=14
    )

    plt.tight_layout()

    filepath = os.path.join(
        OUTPUT_DIR,
        "02_3D_Minimum_Power.png"
    )

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# PLOT 3
# ALTITUDE VS MINIMUM-POWER SPEED
# ============================================================

def create_altitude_vs_min_power_speed_plot(
    minimum_results
):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.plot(
        minimum_results["Altitude_m"],
        minimum_results[
            "Minimum_Power_Airspeed_mps"
        ],
        linewidth=2
    )

    ax.set_xlabel(
        "Altitude (m)",
        fontsize=12
    )

    ax.set_ylabel(
        "Minimum-Power Airspeed (m/s)",
        fontsize=12
    )

    ax.set_title(
        "Minimum-Power Airspeed vs Altitude",
        fontsize=14
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    filepath = os.path.join(
        OUTPUT_DIR,
        "03_Minimum_Power_Airspeed_vs_Altitude.png"
    )

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# PLOT 4
# ALTITUDE VS MINIMUM POWER
# ============================================================

def create_altitude_vs_min_power_plot(
    minimum_results
):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.plot(
        minimum_results["Altitude_m"],
        minimum_results[
            "Minimum_Electrical_Power_kW"
        ],
        linewidth=2
    )

    ax.set_xlabel(
        "Altitude (m)",
        fontsize=12
    )

    ax.set_ylabel(
        "Minimum Electrical Power (kW)",
        fontsize=12
    )

    ax.set_title(
        "Minimum Electrical Power vs Altitude",
        fontsize=14
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    filepath = os.path.join(
        OUTPUT_DIR,
        "04_Minimum_Power_vs_Altitude.png"
    )

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# PLOT 5
# POWER CURVES COLOUR-CODED BY ALTITUDE
# ============================================================

def create_power_curve_altitude_plot(
    all_data
):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    altitude_values = [
        df["Altitude_m"].iloc[0]
        for df in all_data
    ]

    cmap = plt.cm.viridis

    norm = plt.Normalize(
        min(altitude_values),
        max(altitude_values)
    )

    for df in all_data:

        altitude = df[
            "Altitude_m"
        ].iloc[0]

        ax.plot(
            df["Velocity_mps"],
            df["Total_Electrical_Power_W"] / 1000,
            color=cmap(norm(altitude)),
            linewidth=1.2
        )

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )

    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax
    )

    cbar.set_label(
        "Altitude (m)",
        fontsize=11
    )

    ax.set_xlabel(
        "Airspeed (m/s)",
        fontsize=12
    )

    ax.set_ylabel(
        "Electrical Power Required (kW)",
        fontsize=12
    )

    ax.set_title(
        "Electrical Power Required vs Airspeed",
        fontsize=14
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    filepath = os.path.join(
        OUTPUT_DIR,
        "05_Power_Curves_Altitude_Colour_Map.png"
    )

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# PLOT 6
# 100 m AND 20 km COMPARISON
#
# Includes:
#   - Total power
#   - Non-lifting parasitic power
#   - Induced power
#   - Stall region
#   - Stall boundary
#   - Minimum-power point
#   - Minimum-power annotation
# ============================================================

def create_100m_20km_comparison_plot(
    results,
    V_max
):

    selected_altitudes = [
        100,
        20000
    ]

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    # --------------------------------------------------------
    # Plot each selected altitude
    # --------------------------------------------------------

    for altitude in selected_altitudes:

        if altitude not in results:

            print(
                f"Warning: {altitude} m is not available "
                f"in the results."
            )

            continue

        result = results[altitude]

        df = result["df"]

        V_stall = result[
            "V_stall"
        ]

        V_min_power = result[
            "V_min_power"
        ]

        P_min_power = result[
            "P_min_power"
        ]

        # ----------------------------------------------------
        # Total power
        # ----------------------------------------------------

        ax.plot(
            df["Velocity_mps"],
            df["Total_Electrical_Power_W"] / 1000,
            linewidth=2.5,
            label=(
                f"Total power - "
                f"{altitude:,} m"
            )
        )

        # ----------------------------------------------------
        # Non-lifting parasitic power
        # ----------------------------------------------------

        ax.plot(
            df["Velocity_mps"],
            df["Parasite_Power_W"] / 1000,
            linestyle="--",
            linewidth=2.0,
            label=(
                f"Non-lifting parasitic - "
                f"{altitude:,} m"
            )
        )

        # ----------------------------------------------------
        # Induced power
        # ----------------------------------------------------

        ax.plot(
            df["Velocity_mps"],
            df["Induced_Power_W"] / 1000,
            linestyle=":",
            linewidth=2.0,
            label=(
                f"Induced power - "
                f"{altitude:,} m"
            )
        )

        # ----------------------------------------------------
        # Stall boundary
        # ----------------------------------------------------

        ax.axvline(
            V_stall,
            linestyle="--",
            linewidth=1.2,
            alpha=0.7
        )

        # ----------------------------------------------------
        # Minimum power point
        # ----------------------------------------------------

        ax.scatter(
            V_min_power,
            P_min_power / 1000,
            s=70,
            zorder=5,
            label=(
                f"Minimum power - "
                f"{altitude:,} m"
            )
        )

        # ----------------------------------------------------
        # Annotation
        # ----------------------------------------------------

        ax.annotate(
            (
                f"{altitude:,} m\n"
                f"V = {V_min_power:.2f} m/s\n"
                f"P = {P_min_power / 1000:.2f} kW"
            ),
            xy=(
                V_min_power,
                P_min_power / 1000
            ),
            xytext=(
                12,
                15
            ),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.8
            )
        )

    # --------------------------------------------------------
    # Determine maximum y-value
    # --------------------------------------------------------

    ymax = 0.0

    for altitude in selected_altitudes:

        if altitude not in results:
            continue

        df = results[altitude]["df"]

        current_max = (
            df["Total_Electrical_Power_W"].max()
            / 1000
        )

        ymax = max(
            ymax,
            current_max
        )

    # --------------------------------------------------------
    # Stall region
    #
    # Use the lowest stall speed of the selected
    # configurations as the boundary for the region.
    # --------------------------------------------------------

    available_stalls = [
        results[altitude]["V_stall"]
        for altitude in selected_altitudes
        if altitude in results
    ]

    if available_stalls:

        lowest_stall = min(
            available_stalls
        )

        ax.axvspan(
            0,
            lowest_stall,
            alpha=0.15,
            label="Below stall speed"
        )

    # --------------------------------------------------------
    # Axis limits
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        V_max
    )

    ax.set_ylim(
        0,
        ymax * 1.15
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Airspeed (m/s)",
        fontsize=12
    )

    ax.set_ylabel(
        "Electrical Power Required (kW)",
        fontsize=12
    )

    ax.set_title(
        "Power Required vs Airspeed: 100 m and 20 km",
        fontsize=14
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        fontsize=9,
        loc="best"
    )

    plt.tight_layout()

    filepath = os.path.join(
        OUTPUT_DIR,
        "06_Power_vs_Airspeed_100m_20km_Comparison.png"
    )

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# SAVE DETAILED DATA FOR EACH ALTITUDE
# ============================================================

def save_altitude_data(
    altitude,
    df
):

    filename = (
        f"Power_Analysis_"
        f"{int(altitude):05d}m.csv"
    )

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    df.to_csv(
        filepath,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HAPS POWER REQUIRED ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Print aircraft information
    # --------------------------------------------------------

    print()
    print("Aircraft Parameters")
    print("-" * 70)

    print(
        f"Mass                    : {m:.2f} kg"
    )

    print(
        f"MTOW                    : {MTOW:.2f} N"
    )

    print(
        f"Wing area               : {S:.2f} m^2"
    )

    print(
        f"Wingspan                : {b:.2f} m"
    )

    print(
        f"Mean aerodynamic chord  : {MAC:.3f} m"
    )

    print(
        f"Aspect ratio            : {AR:.3f}"
    )

    print(
        f"CL                      : {Cl:.3f}"
    )

    print(
        f"Oswald efficiency       : {e:.3f}"
    )

    print(
        f"Thickness/chord         : {t_c:.4f}"
    )

    print()
    print("Electrical Efficiencies")
    print("-" * 70)

    print(
        f"Battery efficiency      : {Eff_bat:.3f}"
    )

    print(
        f"Gearbox efficiency      : {Eff_GB:.3f}"
    )

    print(
        f"Motor efficiency        : {Eff_motor:.3f}"
    )

    print(
        f"Propeller efficiency    : {Eff_prop:.3f}"
    )

    print(
        f"Overall efficiency      : {eff_sys:.4f}"
    )

    print()
    print("Velocity Sweep")
    print("-" * 70)

    print(
        f"Maximum velocity        : {V_max:.2f} m/s"
    )

    print(
        f"Number of velocity pts  : {N_V}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_data = []

    minimum_results = []

    results = {}

    # --------------------------------------------------------
    # Loop through all altitudes
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CALCULATING POWER")
    print("=" * 70)

    for altitude in sorted(
        ATMOSPHERE.keys()
    ):

        rho, mu = ATMOSPHERE[
            altitude
        ]

        try:

            (
                df,
                V_stall,
                V_min_power,
                P_min_power,
                CD0_min_power,
                Re_min_power,
                Cf_min_power
            ) = calculate_power(
                altitude=altitude,
                rho=rho,
                mu=mu,
                V_max=V_max,
                N_V=N_V
            )

        except ValueError as error:

            print(
                f"\nERROR at {altitude} m:"
            )

            print(error)

            continue

        # ----------------------------------------------------
        # Store complete data
        # ----------------------------------------------------

        all_data.append(
            df
        )

        # ----------------------------------------------------
        # Store results dictionary
        #
        # This is required for the 100 m vs 20 km plot.
        # ----------------------------------------------------

        results[altitude] = {

            "df": df,

            "V_stall": V_stall,

            "V_min_power":
                V_min_power,

            "P_min_power":
                P_min_power,

            "CD0_min_power":
                CD0_min_power,

            "Re_min_power":
                Re_min_power,

            "Cf_min_power":
                Cf_min_power,
        }

        # ----------------------------------------------------
        # Store minimum-power result
        # ----------------------------------------------------

        minimum_results.append({

            "Altitude_m":
                altitude,

            "Density_kgm3":
                rho,

            "Dynamic_Viscosity_kgms":
                mu,

            "Stall_Speed_mps":
                V_stall,

            "Minimum_Power_Airspeed_mps":
                V_min_power,

            "Minimum_Electrical_Power_W":
                P_min_power,

            "Minimum_Electrical_Power_kW":
                P_min_power / 1000,

            "Reynolds_at_MinPower":
                Re_min_power,

            "Cf_at_MinPower":
                Cf_min_power,

            "CD0_at_MinPower":
                CD0_min_power,

        })

        # ----------------------------------------------------
        # Save altitude CSV
        # ----------------------------------------------------

        save_altitude_data(
            altitude,
            df
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"{altitude:6.0f} m | "
            f"Vstall = {V_stall:6.2f} m/s | "
            f"VminP = {V_min_power:6.2f} m/s | "
            f"Pmin = {P_min_power / 1000:7.2f} kW | "
            f"Re = {Re_min_power:10.3e} | "
            f"CD0 = {CD0_min_power:.5f}"
        )

    # --------------------------------------------------------
    # Convert minimum results to DataFrame
    # --------------------------------------------------------

    minimum_results = pd.DataFrame(
        minimum_results
    )

    # --------------------------------------------------------
    # Sort by altitude
    # --------------------------------------------------------

    minimum_results.sort_values(
        "Altitude_m",
        inplace=True
    )

    minimum_results.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # Save minimum-power summary
    # --------------------------------------------------------

    summary_filepath = os.path.join(
        OUTPUT_DIR,
        "Minimum_Power_Summary.csv"
    )

    minimum_results.to_csv(
        summary_filepath,
        index=False
    )

    # --------------------------------------------------------
    # Combined complete dataset
    # --------------------------------------------------------

    if all_data:

        combined_data = pd.concat(
            all_data,
            ignore_index=True
        )

        combined_filepath = os.path.join(
            OUTPUT_DIR,
            "Complete_Power_Analysis_All_Altitudes.csv"
        )

        combined_data.to_csv(
            combined_filepath,
            index=False
        )

    # --------------------------------------------------------
    # Create plots
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)

    # Plot 1
    create_combined_power_plot(
        all_data
    )

    # Plot 2
    create_3d_minimum_power_plot(
        minimum_results
    )

    # Plot 3
    create_altitude_vs_min_power_speed_plot(
        minimum_results
    )

    # Plot 4
    create_altitude_vs_min_power_plot(
        minimum_results
    )

    # Plot 5
    create_power_curve_altitude_plot(
        all_data
    )

    # Plot 6
    create_100m_20km_comparison_plot(
        results,
        V_max
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MINIMUM POWER SUMMARY")
    print("=" * 70)

    print(
        minimum_results[
            [
                "Altitude_m",
                "Stall_Speed_mps",
                "Minimum_Power_Airspeed_mps",
                "Minimum_Electrical_Power_kW",
                "Reynolds_at_MinPower",
                "Cf_at_MinPower",
                "CD0_at_MinPower"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Output directory: "
        f"{os.path.abspath(OUTPUT_DIR)}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
