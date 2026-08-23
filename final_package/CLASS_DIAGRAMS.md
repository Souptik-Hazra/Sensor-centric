# UML Class Diagrams

## Figure 1. Conceptual UML Class Diagram - Structural Causal Digital Twin Framework

```plantuml
@startuml
skinparam shadowing false
skinparam classAttributeIconSize 0
skinparam backgroundColor white
skinparam defaultTextAlignment center
skinparam packageStyle rectangle
skinparam ArrowColor #2563eb
skinparam ClassBorderColor #1e3a8a
skinparam ClassFontColor #0f172a
skinparam ClassBackgroundColor #eff6ff
skinparam packageBorderColor #93c5fd
skinparam packageBackgroundColor #ffffff

class TrafficDataProcessor {
  - trafficData
  - sensorMetadata
  + preprocessData()
  + assessSensorReliability()
}

class ForecastingCoordinator {
  - modelPortfolio
  - forecastResults
  + configureForecasting()
  + generateForecast()
}

class CausalAnalysisCoordinator {
  - causalSpecification
  - analysisResults
  + analyzeDisparity()
  + performCausalAnalysis()
}

class DigitalTwinCoordinator {
  - simulationState
  - maintenanceScenario
  + defineScenario()
  + simulateIntervention()
}

class DecisionSupportCoordinator {
  - analysisResults
  - visualizationArtifacts
  + displayResults()
  + generateReport()
}

TrafficDataProcessor --> ForecastingCoordinator : provides data
TrafficDataProcessor --> CausalAnalysisCoordinator : supplies reliability inputs
ForecastingCoordinator --> CausalAnalysisCoordinator : shares forecast outputs
CausalAnalysisCoordinator --> DigitalTwinCoordinator : informs scenario evaluation
DigitalTwinCoordinator --> DecisionSupportCoordinator : returns simulation outputs
ForecastingCoordinator --> DecisionSupportCoordinator : provides forecasts

@enduml
```

This conceptual diagram presents the major software responsibilities proposed for the framework and shows how information flows from telemetry preparation through forecasting, causal analysis, digital-twin simulation, and decision support. The horizontal package layout improves readability and emphasizes the high-level architecture rather than implementation details.

## Figure 2. Conceptual UML Class Diagram - Sensor Reliability and Causal Analysis

```plantuml
@startuml
skinparam shadowing false
skinparam classAttributeIconSize 0
skinparam backgroundColor white
skinparam defaultTextAlignment center
skinparam packageStyle rectangle
skinparam ArrowColor #0f766e
skinparam ClassBorderColor #134e4a
skinparam ClassFontColor #111827
skinparam ClassBackgroundColor #f0fdfa
skinparam packageBorderColor #5eead4
skinparam packageBackgroundColor #ffffff

class SensorTelemetry {
  - speedObservations
  - sensorMetadata
}

class SensorReliabilityEvaluator {
  - reliabilityIndicators
  + assessReliability()
}

class RoadNetwork {
  - networkStructure
  - spatialRelationships
}

class CausalModel {
  - causalStructure
  - causalVariables
  + analyzeRelationships()
}

class PredictionErrorAnalyzer {
  - predictionErrors
  - regionalAnalysis
  + analyzeDisparity()
}

SensorTelemetry --> SensorReliabilityEvaluator : informs
RoadNetwork --> SensorReliabilityEvaluator : contextualizes
SensorReliabilityEvaluator --> CausalModel : provides inputs
RoadNetwork --> CausalModel : supports structure
SensorTelemetry --> PredictionErrorAnalyzer : supplies observations
CausalModel --> PredictionErrorAnalyzer : explains disparity

@enduml
```

This diagram isolates the conceptual analytical path from telemetry and network context to reliability assessment and causal reasoning. The package structure clarifies the analytical responsibilities without introducing implementation-level details or mathematical subcomponents.

## Figure 3. Conceptual UML Class Diagram - Digital Twin and Decision Support

```plantuml
@startuml
skinparam shadowing false
skinparam classAttributeIconSize 0
skinparam backgroundColor white
skinparam defaultTextAlignment center
skinparam packageStyle rectangle
skinparam ArrowColor #7c3aed
skinparam ClassBorderColor #5b21b6
skinparam ClassFontColor #111827
skinparam ClassBackgroundColor #faf5ff
skinparam packageBorderColor #c4b5fd
skinparam packageBackgroundColor #ffffff

class DigitalTwin {
  - systemState
  - simulatedState
  + evaluateScenario()
}

class MaintenanceScenario {
  - targetSensors
  - interventionType
  + defineScenario()
}

class CounterfactualSimulation {
  - simulationParameters
  - simulationResults
  + simulateIntervention()
}

class EquityAssessment {
  - regionalMetrics
  + assessEquity()
}

class DecisionSupport {
  - decisionInformation
  - report
  + presentResults()
  + generateReport()
}

MaintenanceScenario --> CounterfactualSimulation : configures
DigitalTwin --> CounterfactualSimulation : evaluates
CounterfactualSimulation --> EquityAssessment : produces outcomes
EquityAssessment --> DecisionSupport : summarizes equity
DigitalTwin --> DecisionSupport : supplies system state
CounterfactualSimulation --> DecisionSupport : supplies evidence

@enduml
```

This diagram captures the proposed maintenance decision workflow inside the digital twin. The package grouping makes the scenario, simulation, equity, and reporting stages explicit while keeping the figure compact and proposal-stage appropriate.

## Figure 4. Conceptual UML Class Diagram - Forecasting Model Hierarchy

```plantuml
@startuml
skinparam shadowing false
skinparam classAttributeIconSize 0
skinparam backgroundColor white
skinparam defaultTextAlignment center
skinparam packageStyle rectangle
skinparam ArrowColor #b45309
skinparam ClassBorderColor #92400e
skinparam ClassFontColor #111827
skinparam ClassBackgroundColor #fffbeb
skinparam packageBorderColor #fbbf24
skinparam packageBackgroundColor #ffffff

abstract class TrafficForecastingModel {
  + train()
  + forecast()
  + evaluate()
}

class HistoricalAverage
class DLinear
class DCRNN
class GraphWaveNet

TrafficForecastingModel <|-- HistoricalAverage
TrafficForecastingModel <|-- DLinear
TrafficForecastingModel <|-- DCRNN
TrafficForecastingModel <|-- GraphWaveNet

@enduml
```

This hierarchy presents the forecasting approaches as alternative implementations of a shared conceptual forecasting model. The abstract base class and package separation support comparative evaluation while avoiding internal architectural details that belong in the methodology chapter rather than the class diagram.
