Power Required Analysis – User Guide
1. Overview
This Python script calculates the aircraft's aerodynamic and electrical power requirements across multiple altitudes and airspeeds. It also identifies the minimum-power airspeed at each altitude and generates plots and CSV files for further analysis.
2. Requirements
Install the required Python libraries:
pip install numpy pandas matplotlib


3. User Inputs
Modify the parameters under the USER INPUTS section at the beginning of the script.
Parameter	Description	Unit
m	Aircraft mass	kg
Cl	Lift coefficient used for stall calculation	-
b	Wingspan	m
S	Wing area	m²
Q	Form factor multiplier	-
t_c	Maximum thickness-to-chord ratio	-
e	Oswald efficiency factor	-
Eff_bat	Battery efficiency	-
Eff_GB	Gearbox efficiency	-
Eff_motor	Motor efficiency	-
Eff_prop	Propeller efficiency	-
V_max	Maximum velocity for analysis	m/s
N_V	Number of velocity points	-

The code automatically calculates MTOW, MAC, aspect ratio and overall system efficiency.

Note: V_climb is retained as a reference input and is not used in the calculations.
4. Atmospheric Inputs
The ATMOSPHERE dictionary contains atmospheric properties at different altitudes.
ATMOSPHERE = {    100: (1.14e-00, 1.86e-05),    500: (1.10e-00, 1.85e-05),    20000: (9.43e-02, 1.37e-05),}



Each entry follows the format:
Altitude: (Density, Dynamic Viscosity)



Add, modify or remove altitude entries as required. Ensure that density and viscosity values are consistent with the selected atmospheric model.
5. Execution
1. Save the script as Power_Required_Analysis.py.
2. Modify the aircraft parameters and atmospheric data.
3. Set the maximum velocity and velocity resolution.
4. Run the script:
python Power_Required_Analysis.py


5. The script automatically performs the analysis for all specified altitudes.
6. Calculation Methodology
At each altitude, the code:
1. Calculates stall speed.
2. Generates a velocity sweep from stall speed to maximum velocity.
3. Calculates Reynolds number and skin friction coefficient.
4. Estimates parasite drag coefficient using the form factor and wetted area.
5. Calculates parasite drag and induced drag.
6. Determines total aerodynamic drag and aerodynamic power.
7. Converts aerodynamic power into electrical power using overall system efficiency.
8. Identifies the minimum electrical power and corresponding airspeed.
7. Outputs
All outputs are automatically saved in the following directory:
Power_Required_Analysis/


Generated plots:
File	Description
01_Total_Power_vs_Airspeed_All_Altitudes.png	Electrical power vs airspeed for all altitudes
02_3D_Minimum_Power.png	3D representation of minimum power, airspeed and altitude
03_Minimum_Power_Airspeed_vs_Altitude.png	Minimum-power airspeed vs altitude
04_Minimum_Power_vs_Altitude.png	Minimum electrical power vs altitude
05_Power_Curves_Altitude_Colour_Map.png	Power curves colour-coded by altitude
06_Power_vs_Airspeed_100m_20km_Comparison.png	Comparison of power components at 100 m and 20 km

Generated CSV files:
- Power_Analysis_XXXXXm.csv – Detailed power analysis at each altitude.
- Minimum_Power_Summary.csv – Minimum-power operating conditions at all altitudes.
- Complete_Power_Analysis_All_Altitudes.csv – Combined detailed dataset for all altitudes.
8. Important Considerations
- All aircraft dimensions must be entered in SI units.
- V_max must be greater than the calculated stall speed at every analysed altitude.
- The code assumes constant aircraft weight and lift coefficient for stall-speed calculation.
- Skin friction is estimated using the laminar Blasius correlation.
- System efficiencies are assumed constant across the operating envelope.
- The analysis calculates steady, level-flight power requirements; climb power, acceleration and manoeuvring requirements are not included.
Note: The results are preliminary analytical estimates and should be validated against appropriate aerodynamic and propulsion performance data before design use.
