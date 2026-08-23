# Literature Review

## Paper

### Title

Microscopic Traffic Simulation using SUMO

### Authors

Pablo Alvarez Lopez, Michael Behrisch, Laura Bieker-Walz, Jakob Erdmann, Yun-Pang Flötteröd, Robert Hilbrich, Leonhard Lücken, Johannes Rummel, Peter Wagner, and Evamarie Wießner

### Year

2018

### Venue / Journal / Conference

2018 21st International Conference on Intelligent Transportation Systems (ITSC), Maui, Hawaii, USA, November 4–7, 2018.

# 1. Objective & Problem Formulation

- The paper presents a comprehensive overview of the **Simulation of Urban MObility (SUMO)** open-source microscopic traffic simulator and its associated tools, models, extensions, and interfaces.
- The primary objective is to describe how SUMO can be used to generate, simulate, adapt, validate, and evaluate detailed traffic scenarios.
- The paper addresses the need for traffic simulation capable of representing individual vehicles and their dynamics when evaluating traffic-management solutions.
- The motivation is that accurate knowledge of traffic conditions and dynamics is important for implementing and evaluating traffic-management strategies.
- Microscopic simulation is particularly relevant when individual vehicle routes, vehicle behavior, trajectories, or emissions need to be represented.
- The paper covers the complete simulation workflow, including:
  - Network construction
  - Infrastructure preparation
  - Traffic-demand generation
  - Traffic assignment
  - Demand adaptation
  - Multi-/intermodal transportation
  - Pedestrian simulation
  - Emission modeling
  - Simulation validation
  - Modeling enhancements
  - Simulator coupling
- The paper is primarily a **technical framework and overview paper**, rather than a machine-learning traffic-forecasting study.

# 2. Existing Systems & Background

- The paper distinguishes four levels of traffic simulation:
  - **Macroscopic:** represents average vehicle dynamics such as traffic density.
  - **Microscopic:** represents individual vehicles and their dynamics.
  - **Mesoscopic:** combines macroscopic and microscopic approaches.
  - **Submicroscopic:** represents individual vehicles together with internal vehicle functions such as gear shifting.
- Macroscopic approaches generally offer faster execution, while microscopic and submicroscopic approaches provide greater detail.
- Microscopic approaches are particularly useful when individual routes or emissions need to be represented.
- **PTV Vissim** is described as a commercial traffic-simulation system providing a user-friendly interface and 3D visualization.
- **MATSim** is described as an open-source, activity-based traffic-simulation framework.
- SUMO is presented as an open-source traffic simulator distributed under the **Eclipse Public License V2**.
- A SUMO simulation scenario requires:
  - Network information
  - Additional traffic infrastructure
  - Traffic demand
- Network information can represent roads, footpaths, tracks, waterways, bike lanes, and other transportation infrastructure.
- Traffic simulation models are stochastic, and the paper notes that multiple simulation runs are generally required to obtain statistical conclusions.

# 3. Limitations of Existing Systems & Research Gaps

- Available input data frequently lack the level of detail required for microscopic simulation.
- This mismatch between available data and required simulation detail makes network and infrastructure preparation challenging and can require manual refinement.
- Imported or generated traffic demand may not accurately represent actual traffic conditions because of:
  - Limited demand data
  - Daily activity changes
  - Unexpected events
  - Road or area closures
  - Changes in route or parking requirements
- Some demand-generation and routing approaches have specific limitations:
  - DFROUTER has known deficiencies for highly meshed networks such as city networks.
  - JTRROUTER may cause vehicles to run in circles in city areas because traffic volumes are split according to turning ratios and already-used edges are not considered.
- The Real-World Bologna scenario may be too small for more complex or larger evaluations, and its traffic demand may be insufficient for some evaluations.
- Microsimulation validation is challenging because validity can concern multiple aspects of the simulation, including input data, transport demand, and simulated behavior.
- The authors emphasize that there is no single solution for validating microsimulation and that continuous scrutiny is required.
- Manual parameter calibration is difficult to reproduce, motivating the use of numerical optimization methods.
- The paper does not formulate a conventional machine-learning research gap. Its practical contribution is instead centered on providing an integrated open-source environment for detailed traffic-scenario construction, simulation, demand adaptation, validation, model enhancement, and coupling with other simulators.

# 4. Proposed System & Technical Architecture

- The paper presents **SUMO** as an open-source microscopic traffic-simulation framework supported by a collection of tools for constructing, simulating, adapting, validating, and analyzing traffic scenarios.
- The overall workflow is:

  **Network and Infrastructure Data → Network Construction/Refinement → Traffic-Demand Generation/Import → Traffic Assignment/Route Generation → Microscopic Simulation → Demand Adaptation/Calibration → Validation → Simulation Outputs**

- SUMO represents individual vehicles and their interactions within an explicit transportation network.
- The transportation network contains nodes, unidirectional edges, lanes, junctions, intersection movements, right-of-way rules, and geometric information.
- Traffic demand can be represented as:
  - Individual trips
  - Flows
  - Routes
- Demand can originate from:
  - O-D matrices
  - Detector information
  - Activity-based models
  - Agent-based models
  - Synthetic random trips
- SUMO also provides mechanisms for:
  - Dynamic route adaptation
  - Public-transport simulation
  - Pedestrian simulation
  - Emission simulation
  - Vehicle-dynamics modeling
  - External simulator coupling
- Simulation outputs can include traffic information aggregated over edges/lanes, vehicle/person trips, or the whole simulation, as well as traffic-light information, trajectories, emissions, and energy-related measures.

### **Architectural Breakdown:**

- *Spatial Topology / Graph Construction:* SUMO represents spatial structure explicitly through **nodes, unidirectional edges, lanes, junctions, intersection movements, right-of-way rules, and geometry**. Edges can represent streets, waterways, tracks, bike lanes, and walkways. Lanes contain attributes such as width, speed limit, and access permissions. If an attribute changes along a road, the road stretch must be represented as a sequence of edges. **NETCONVERT** imports networks from sources including OpenStreetMap, OpenDRIVE, Shapefile, MATSim, and Vissim and uses heuristics to refine missing network information. **NETEDIT** provides graphical network creation, analysis, editing, and manual refinement. **OSMWebWizard** provides browser-based scenario preparation using OpenStreetMap data.
- *Spatial Encoding Module:* **Not discussed — SUMO represents spatial structure explicitly through the transportation network rather than a learned spatial encoder.**
- *Temporal Encoding Module:* SUMO represents time through departure times, traffic flows, routes, time intervals, simulation-step lengths, and action points. Traffic demand can be distributed over selected time intervals. Vehicle states are updated at discrete simulation steps. Action Points can decouple effective driver reaction time from the simulation-step length.
- *Loss Function & Optimization:* **Not discussed as a machine-learning loss function.** For calibration, the paper describes minimizing differences between measured and simulated data and identifies RMSE as one possible measure of closeness. Numerical optimization, including nonlinear minimization routines, is recommended instead of manual parameter tuning.
- *Network Generation:* NETCONVERT can import and heuristically refine network data. It can synthesize missing traffic-light plans, right-of-way rules, and intersection geometries. NETEDIT supports subsequent manual refinement and additional infrastructure.
- *Traffic-Demand Generation:* SUMO supports individual trips, flows, and routes. ACTIVITYGEN generates synthetic demand using an activity-based model. Flowrouter uses detector data as road-capacity information. DFROUTER generates routes using detector flows. JTRROUTER uses traffic volumes and turning ratios. randomTrips.py generates synthetic trips. OD2TRIPS converts O-D matrices into individual trips.
- *Traffic Assignment:* DUAROUTER, MAROUTER, and Oneshot support traffic assignment based on principles including **User Equilibrium (UE), Stochastic User Equilibrium (SUE), and fastest-route selection at a given departure time**.
- *Demand Adaptation:* Cadyts adjusts route distributions using measured traffic flows. SUMO calibrators use measured flows and speeds to insert/remove vehicles and adjust speeds. REROUTER can modify routes and destinations based on route, parking, and current traffic-state information.
- *Vehicle Dynamics:* SUMO uses a first-order **Euler integration scheme** by default. A **ballistic integration scheme** is also available and represents constant acceleration during a simulation step, producing smoother trajectories than abrupt Euler-based speed changes.
- *Action Points:* Action points allow the effective reaction time to differ from the simulation-step length. Between action points, positional updates can be handled using uniform increments under constant acceleration, while full behavioral decision logic is evaluated at action points. This can support faster simulation using short simulation steps.
- *Sublane Model:* The Sublane Model represents heterogeneous traffic and reduced lane discipline using lane width, vehicle width, configurable lateral resolution, overtaking speed, lateral distance keeping, and virtual-lane formation. It also permits overtaking within a single lane.
- *Oncoming-Lane Overtaking:* An optional model allows overtaking through an opposing-direction lane. It requires network information about lane oppositionality and is not compatible with the Sublane Model.
- *Multi-/Intermodal Simulation:* Persons are the central modeling element for intermodal traffic. A person can have an individual travel plan involving walking, individual vehicles, public transport, and non-mobility activities. SUMO can calculate fastest routes across combinations of available modes.
- *Public Transport:* Public transport is represented through vehicles, lines, stops, and schedules. Stop information can be obtained from sources such as OpenStreetMap. Access facilities can model transfers between road and rail networks with fixed transfer-time penalties.
- *Pedestrian Simulation:* SUMO provides configurable pedestrian models. The **Non-Interacting Model** does not represent pedestrian-pedestrian or pedestrian-vehicle interactions. The **Striping Model** represents interactions along piecewise-linear paths and uses narrow stripes along the walking direction. Pedestrian crossings are represented as part of the road network and can be generated through NETCONVERT or edited through NETEDIT.
- *Emission Modeling:* Vehicle types and emission classes can be used to model **CO, CO₂, NOx, PMx, HC, and fuel consumption**. The paper discusses HBEFA and PHEM as sources/models for emission information.
- *Simulator Coupling:* **TraCI (Traffic Control Interface)** provides socket-based communication between SUMO and external programs, allowing external applications to retrieve data and start, stop, or modify simulations. The paper discusses applications involving iTETRIS, COLOMBO, ns-3, Veins, OMNeT++, and Plexe.

# 5. Main Contributions

- Provides a comprehensive technical overview of **SUMO as an open-source microscopic traffic simulator**, including scenario preparation, simulation, validation, and evaluation.
- Describes mechanisms for converting, refining, and manually editing transportation networks through **NETCONVERT, NETEDIT, and OSMWebWizard**.
- Presents multiple mechanisms for generating, assigning, calibrating, and adapting traffic demand from real-world and synthetic information.
- Demonstrates SUMO's support for **multi-/intermodal transportation, public transport, pedestrians, vehicle dynamics, and emissions**.
- Describes validation mechanisms, modeling enhancements, and external simulator coupling through **TraCI**, including applications involving vehicular communications and autonomous-vehicle platooning.

# 6. Experimental Setup

- **Datasets:**
  - The primary real-world example is the **Real-World Bologna scenario**.
  - The scenario covers an area of the inner city of **Bologna, Italy**, between Andrea Costa and Pasubio and including a football stadium and hospital.
  - The municipality provided traffic demand and traffic-network information in the form of a **Vissim scenario**.
  - The information could be automatically and manually converted and extended into a SUMO scenario.
  - A major extension added lanes explicitly restricted to buses.
  - Traffic-light positions and signal timing plans were supplied through Bologna municipal telemetry data.
  - Traffic demand is based on real-world **induction-loop detector measurements**.
  - Vehicle-count data were available for **11, 12, and 13 November 2008**.
  - Measurements were recorded every **5 minutes**.
- **Data Scope & Exclusions:**
  - Detector data were smoothed using a **Savitzky–Golay filter**.
  - Traffic activity was low at night and increased during morning and afternoon rush hours.
  - A one-hour period representing a typical **6am–9am rush-hour period** was created for the scenario.
  - The scenario also includes public transport, bus stops, special lanes, bus-only lanes, and fictional person trips for demonstrating intermodal functionality.
  - Exact number of sensors/nodes: **Not discussed.**
  - Holiday exclusions: **Not discussed.**
  - Seasonal exclusions: **Not discussed.**
  - Explicit speed-threshold filtering: **Not discussed.**
- **Evaluation Metrics:**
  - **RMSE** is discussed as one possible measure of closeness between simulation output and measured data.
  - Percentage deviation between simulated and measured traffic counts is used for network-level validation.
  - Vehicle-gap error is discussed for microscopic behavior.
  - Qualitative visual comparison is also used.
  - Linear-fit parameters can provide additional measures for comparing simulated and measured traffic counts.
- **Baseline Models:**
  - **PTV Vissim** and **MATSim** are discussed as existing traffic-simulation systems.
  - They are not presented as conventional ML benchmark baselines.
  - No neural traffic-forecasting baselines are reported.
  - No standard MAE/MAPE forecasting benchmark is provided.
- **Traffic-Demand Generation Inputs:**
  - ACTIVITYGEN can use population and contextual information such as inhabitants, households, retirement age, unemployment probability, age brackets, work hours, population/work-position distributions, city boundaries, schools, and bus lines.
  - O-D matrices represent demand between traffic-analysis zones and can contain demand for transport modes such as passenger cars and trucks.
  - O-D matrices cannot directly be used in SUMO; **OD2TRIPS** converts them into individual trips.
  - Departure times for converted O-D trips can be generated using uniform or random distributions.
  - SUMO can also couple with the **TAPAS agent-based demand model developed by DLR**.
- **Emission Setup:**
  - Vehicle distributions account for emissions, acceleration, and deceleration.
  - Different emission classes are assigned to vehicle types.
  - HBEFA can provide information for CO, CO₂, NOx, PMx, HC, and fuel consumption.
  - PHEM is also described as an emission model used in connection with SUMO.

# 7. Experimental Findings & Performance Breakdown

- **Quantitative Results:**
  - For microscopic behavior such as the gap between two vehicles, the paper reports typical validation errors of **10–20%** for different microscopic models.
  - At the network level, a typical validation criterion indicates that **more than 85% of simulated counts should deviate less than 15% from measured counts**.
  - The Bologna scenario provides a network-level comparison between simulated traffic counts and measured induction-loop counts.
  - Perfect agreement between simulation and measurement would place the results on the diagonal of the comparison plot.
  - Linear-fit parameters can provide additional metrics for comparing simulated and measured values.
  - RMSE is discussed as a possible closeness measure.
  - The paper does not report MAE, MAPE, R², CRPS, or other standard traffic-forecasting metrics.
  - No 15-minute, 30-minute, or 60-minute forecasting horizons are reported.
- **Reference-Based Findings:**
  - Simulation can also be evaluated against values from sources such as the **Highway Capacity Manual (HCM)** and the German **HBS**.
  - SUMO simulation can come fairly close to handbook curves representing the fundamental diagram.
  - Traffic-flow breakdown is described as more complicated, and differences can occur between different models.
- **Calibration Findings:**
  - The San Pablo Dam Road example is used to illustrate calibration by finding parameter values that minimize differences between measured data and simulation.
  - The authors advise against manual parameter minimization because hand-tuned results are difficult to reproduce.
  - Numerical optimization tools such as nonlinear minimization routines are recommended.
- **Relative Improvement:**
  - **Not discussed.** No percentage improvement over a baseline traffic-prediction or simulation model is reported.
- **Ablation Study Findings:**
  - **Not discussed.**

# 8. Strengths

- Detailed microscopic representation of individual vehicles and their dynamics.
- Explicit node-edge-lane transportation-network representation.
- Representation of intersection movements and right-of-way rules.
- Automated network conversion and heuristic refinement.
- Manual network and infrastructure editing.
- Support for multiple external network-data sources.
- Multiple traffic-demand representations and generation mechanisms.
- Detector-based traffic-demand generation.
- O-D-based demand conversion.
- Activity-based and agent-based demand generation.
- Demand calibration and adaptation.
- Dynamic route modification.
- Multi-/intermodal transportation support.
- Public-transport simulation.
- Pedestrian simulation.
- Emission and fuel-consumption modeling.
- Multiple vehicle-dynamics and numerical-integration mechanisms.
- Support for heterogeneous and lateral traffic behavior through the Sublane Model.
- Qualitative and quantitative validation.
- Reference-based validation using HCM/HBS information.
- External simulator coupling through TraCI.
- Support for vehicular-communication simulation.
- Extension through Veins and Plexe for communication and autonomous-vehicle platooning applications.
- Open-source availability and an extensive set of tools for scenario generation, simulation, validation, and analysis.

# 9. Remaining Limitations & Vulnerabilities

## Author-Mentioned Limitations

- The paper identifies a **frequent mismatch between available input data and the level of detail required for microscopic simulation**, making network and infrastructure preparation challenging.
- The Real-World Bologna scenario may be **too small for more complex or larger evaluations**, and its traffic demand may be too low for some evaluations.
- DFROUTER has known deficiencies for **highly meshed networks such as city networks**.
- JTRROUTER may cause vehicles to **run in circles in city areas** because traffic volumes are split according to turning ratios and already-used edges are not considered.
- Limited traffic-demand data, changes in daily activity, and unexpected events can cause generated or imported traffic demand to differ from actual traffic conditions.
- The validation discussion emphasizes that there is **“no one-stop-shop”** for microsimulation validation and that permanent vigilance and scrutiny are necessary.
- Manual parameter minimization is discouraged because hand-tuned calibration results are difficult to reproduce; numerical optimization is recommended.
- The optional oncoming-lane overtaking model requires lane-oppositionality information and is **not compatible with the Sublane Model**.
- The paper's Outlook indicates that additional SUMO features and extensions were still planned, particularly for railway simulation and traffic-demand definition.
- Planned railway extensions include:
  - Network and timetable import from public transit
  - Improved support for bidirectional track usage
  - Individual rail-car and coupling modeling
  - Additional train dynamic models
  - Graphical tools for defining traffic demand

## Broader Impact & Ethical Considerations

**Not discussed.**

The paper does not contain a dedicated Broader Impact, Ethics, Societal Implications, or Environmental Impact section. Although emissions and fuel consumption are modeled technically, the paper does not frame these capabilities as an explicit ethical, societal, or environmental-impact analysis.

# 10. Sensor-Centric & Causal Perspective

- **Sensor Reliability & Data Quality:**
  - The Real-World Bologna scenario uses real-world **induction-loop detector data**.
  - Measurements were collected every **5 minutes**.
  - Detector measurements were smoothed using a **Savitzky–Golay filter**.
  - Detector data are used for traffic-demand and route-generation processes.
  - SUMO calibrators use measured flows and speeds to adjust simulated traffic.
  - Calibrators can insert or remove vehicles and adjust road speeds to make simulation output comply with measured conditions.
  - The paper also discusses removing jammed vehicles to avoid invalid jam spillback into a calibrator.
  - Measured detector counts are used for network-level validation.
  - Explicit treatment of sensor failure, sensor drift, stuck readings, zero-dropouts, hardware degradation, or missing-data imputation is **Not discussed.**
- **Spatial Equity & Disparity:**
  - **Not discussed.** The paper does not evaluate regional error disparities across different geographic sensor locations.
- **Causal Reasoning Level:**
  - The paper focuses on microscopic simulation, calibration, and scenario evaluation rather than formal causal inference.
  - It does not formulate a Pearl causal model.
  - It does not explicitly identify its methodology as Pearl Level 1, Level 2, or Level 3.
  - SUMO can modify traffic demand, routes, speeds, destinations, and network conditions, but these capabilities are not presented as a formal causal-intervention or counterfactual framework.
  - Therefore, the paper's exact Pearl causal level is **Not established**.

# 11. Terminology & Acronym Collision Notes

- **SUMO:** The central simulator name; relatively specific in traffic simulation but potentially ambiguous outside the field.
- **UE:** User Equilibrium; a generic transportation acronym that can occur in unrelated transportation literature.
- **SUE:** Stochastic User Equilibrium; similarly generic within transportation research.
- **HCM:** Highway Capacity Manual; potentially ambiguous outside transportation.
- **HBS:** German transportation/traffic-capacity terminology; potentially ambiguous outside its domain.
- **TraCI:** Traffic Control Interface; relatively specific to SUMO.
- **Cadyts:** Named traffic-demand/route-choice calibration tool.
- **PHEM:** Passenger car and Heavy duty vehicle Emission Model; specialized but potentially ambiguous outside emission research.
- **HBEFA:** Handbook Emission Factors for Road Transport; specialized emission-model terminology.
- **DFROUTER:** SUMO-specific route-generation tool.
- **JTRROUTER:** SUMO-specific route-generation tool.
- **REROUTER:** SUMO-specific dynamic rerouting mechanism.
- No specific collision with common graph-based traffic-forecasting architectures such as DCRNN or STGCN is established by this paper.

# 12. Keywords (20–30)

- Microscopic traffic simulation
- SUMO
- Intelligent transportation systems
- Traffic network modeling
- Network topology
- NETCONVERT
- NETEDIT
- OSMWebWizard
- Traffic-demand generation
- Traffic-demand adaptation
- Traffic assignment
- Induction-loop detectors
- Cadyts
- Dynamic rerouting
- O-D matrix
- ACTIVITYGEN
- Intermodal traffic
- Public transport simulation
- Pedestrian simulation
- Vehicle dynamics
- Sublane Model
- Emission modeling
- HBEFA
- PHEM
- Simulation calibration
- Simulation validation
- TraCI
- Simulator coupling
- Vehicular communication
- Bologna traffic scenario