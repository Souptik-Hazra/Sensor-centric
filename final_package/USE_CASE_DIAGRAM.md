# UML Use Case Diagram

## Use Case Diagram - Forecasting, Regional Analysis, and Maintenance Decision Support

```plantuml
@startuml
top to bottom direction
skinparam packageStyle rectangle
skinparam linetype ortho
skinparam nodesep 28
skinparam ranksep 30
skinparam shadowing false
skinparam backgroundColor white
skinparam defaultTextAlignment center
skinparam ArrowColor #4b5563
skinparam ArrowThickness 1.2
skinparam actorBorderColor #334155
skinparam actorFontColor #111827
skinparam actorStereotypeFontColor #475569
skinparam usecaseBorderColor #334155
skinparam usecaseFontColor #111827
skinparam usecaseBackgroundColor #f8fafc
skinparam packageBorderColor #334155
skinparam packageFontColor #111827
skinparam packageBackgroundColor #ffffff
skinparam rectangleBorderColor #64748b
skinparam rectangleFontColor #111827
skinparam rectangleBackgroundColor #ffffff
skinparam noteBackgroundColor #fff7ed
skinparam noteBorderColor #fb923c

actor "Transportation Planner" as TP
actor "Traffic Data Provider" as TDP
actor "Sensor Network" as SN
actor "External Data Repository" as EDR

top to bottom direction

rectangle "Proposed Traffic Forecasting and Digital Twin System" {

  rectangle "Forecasting and Regional Analysis" as FRA #dbeafe {
    usecase "Provide / Select Traffic Data" as UC2
    usecase "Assess Sensor Reliability" as UC3
    usecase "Generate Traffic Forecasts" as UC4
    usecase "Analyze Regional Prediction Disparities" as UC5
    usecase "Perform Structural Causal Analysis" as UC6
    usecase "View Analysis Results" as UC7
  }

  rectangle "Maintenance Decision Support" as MDS #dcfce7 {
    usecase "Define Maintenance Scenario" as UC8
    usecase "Configure Sensor Intervention" as UC9
    usecase "Simulate Intervention" as UC10
    usecase "Evaluate Regional Equity" as UC11
    usecase "Compare Scenario Outcomes" as UC12
    usecase "View Simulation Results" as UC13
    usecase "Generate Decision-Support Report" as UC14
  }
}

' Human actor entry points
TP -right- UC2
TP -down- UC8
TP -right- UC7
TP -right- UC13
TP -right- UC14

' Technical integration touchpoints
TDP -down- UC2
EDR -down- UC2
EDR -down- UC7
EDR -down- UC13
SN -down- UC3
SN -down- UC10

' Functional dependencies
UC2 ..> UC3 : <<include>>
UC3 ..> UC4 : <<include>>
UC4 ..> UC5 : <<include>>
UC5 ..> UC6 : <<include>>
UC6 ..> UC7 : <<include>>

UC8 ..> UC9 : <<include>>
UC9 ..> UC10 : <<include>>
UC10 ..> UC11 : <<include>>
UC11 ..> UC12 : <<include>>
UC12 ..> UC13 : <<include>>
UC13 ..> UC14 : <<include>>
@enduml
```

This diagram keeps the proposal-stage system conceptually clean while showing both major use-case areas. The Transportation Planner is the primary human actor, and the technical actors represent external integration points for traffic data, sensors, and stored results.
