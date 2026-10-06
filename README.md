# End-to-End LLM-Powered Machine Quality Analysis and Prediction System

An AI-powered proof of concept for exploring, analyzing, and forecasting machine operational data through structured data processing, machine learning, and a Large Language Model (LLM) interface.

> **Note:** This repository is an intentionally scaled-down public version of a significantly larger proof-of-concept system I previously developed. The original development environment contained substantially more files, assets, integrations, model-related resources, and experimental components, making it impractical to publish on GitHub in its original form.
>
> Because the project was developed through rapid experimentation, not every component present during development was necessarily part of the final runtime pipeline. This repository therefore focuses on the core functionality that I can confidently document and reproduce.
>
> Proprietary data, credentials, internal integrations, organization-specific resources, and unnecessary large artifacts have also been excluded.

---

## Overview

This project was developed to explore whether machine operational data could be combined with traditional data analysis, machine-learning forecasting, and an LLM-based conversational interface.

Instead of requiring users to manually inspect raw machine records, the system processes machine data, performs the relevant analysis, and provides structured analytical results to an LLM.

The LLM then converts those results into a more accessible natural-language response.

The simplified implementation demonstrates:

- Natural-language query handling
- Query routing
- Machine data preprocessing
- Run-based and timestamp-based data processing
- Trend analysis
- Statistical visualization
- Daily machine behaviour analysis
- Machine-learning-based forecasting
- Structured context generation
- LLM-assisted response generation
- Vercel AI SDK integration

The main architectural idea is to separate deterministic computation from generative output.

The LLM is therefore not expected to perform all calculations directly.

---

## Development Context

This project was originally developed as a **proof of concept (PoC)** that I independently designed and implemented.

I was the **sole technical owner** responsible for the technical development of the project, including:

- System architecture
- Data-processing pipeline
- Analytical components
- Machine-learning experimentation
- LLM integration
- Query-handling logic
- Application integration
- Technical experimentation
- Overall implementation

The primary objective was to determine whether machine operational data, analytical methods, machine-learning models, and an LLM interface could be combined into a useful end-to-end workflow.

Because the project was initially built as a PoC, development focused heavily on:

- Rapid experimentation
- Validating the core concept
- Testing alternative approaches
- Integrating multiple technical components
- Determining which components were useful in practice

As a result, the original codebase was more exploratory than production-oriented.

It contained experimental code, intermediate artifacts, alternative implementations, model files, and development resources that were not necessarily part of the final runtime path.

This repository is therefore a simplified and cleaned public representation of that work rather than an exact copy of the original development environment.

---

## Architecture

## Architecture

The diagram below illustrates the architecture of the scaled-down public version included in this repository.

<p align="center">
  <img src="docs/ machine condition analyzer architecture.drawio.png" alt="System Architecture" width="800">
</p>


At a high level:

```text
User Query
    |
    v
Query Recognition
    |
    +----------------------+
    |                      |
    | Recognized           | Not Recognized
    v                      v
Analysis Pipeline     Template Response
    |
    v
Structured Results
    |
    v
Vercel AI SDK
    |
    v
LLM
    |
    v
Response to User
```

Machine data enters the analytical pipeline separately:

```text
Machine Data
    |
    +--> Run Identifier
    |
    +--> Timestamp
    |
    +--> Machine Value
    |
    v
Preprocessing
```

The resulting information can then be used by analytical, visualization, forecasting, and LLM components.

---

## Machine Data Processing

The system works with structured machine records containing information such as:

- Run identifiers
- Timestamps
- Machine measurement values
- Sequential operating measurements

Before analysis, the records are processed into representations that are easier for downstream components to consume.

Depending on the operation, preprocessing can include:

- Run-based grouping
- Timestamp processing
- Sequential machine-value extraction
- Key-value representations
- Time-series summaries
- Structured analytical context generation

The purpose of this stage is to avoid sending unprocessed machine records directly into the LLM whenever dedicated processing can be performed first.

---

## Query Handling

Incoming user queries are evaluated before being passed through the analytical system.

Recognized requests can trigger specific operations such as:

```text
User Query
    |
    +--> Trend Analysis
    |
    +--> Statistical Analysis
    |
    +--> Behaviour Analysis
    |
    +--> Machine Learning Forecast
```

Other supported interactions can be handled using predefined application logic or template responses.

This allows the application to combine:

- Deterministic application logic
- Data analysis
- Machine learning
- Generative AI

rather than forwarding every request directly to the language model.

---

## Trend Analysis

Machine measurements can be analyzed over time to identify changes in operational behaviour.

Example query:

```text
Show the recent trend of the machine.
```

The corresponding machine records can be retrieved and processed before the resulting trend information is presented to the user or supplied to the LLM.

---

## Statistical Analysis

The system can generate statistical representations and visualizations of machine values.

Examples include:

- Line charts
- Box plots
- Time-based summaries
- Daily machine visualizations

These analytical outputs provide additional context for understanding how machine values behave over time.

---

## Daily Behaviour Analysis

Machine records can also be grouped and analyzed over a particular day or operating period.

Example:

```text
How has the machine behaved today?
```

The system can calculate or summarize the relevant machine behaviour before providing the resulting information to the LLM.

---

## Machine Learning Forecasting

The project includes a machine-learning forecasting component for estimating future machine values.

The simplified public implementation includes approaches such as:

- Linear Regression
- Random Forest

A simplified forecasting workflow is:

```text
Historical Machine Data
        |
        v
Preprocessing
        |
        v
Machine Learning Model
        |
        +--> Linear Regression
        |
        +--> Random Forest
        |
        v
Prediction
        |
        v
Structured Prediction Result
```

Depending on the experiment and configuration, forecasts can be produced across multiple future horizons.

Examples explored in the project include:

```text
3 minutes
6 minutes
12 minutes
24 minutes
48 minutes
```

The resulting prediction can then become part of the structured context supplied to the language model.

Example query:

```text
What is the predicted machine value for the next 12 minutes?
```

---

## LLM Integration

The LLM functions primarily as a conversational and interpretation layer.

It is not intended to replace the deterministic analytical components.

Instead, the workflow follows a pattern similar to:

```text
Raw Machine Data
        |
        v
Data Processing
        |
        v
Analysis / Prediction
        |
        v
Structured Result
        |
        v
LLM
        |
        v
Natural-Language Response
```

This separation allows numerical operations and machine-learning predictions to remain explicit instead of relying entirely on generative reasoning.

It also makes the underlying analytical output easier to inspect independently from the final LLM response.

---

## Vercel AI SDK Integration

The application uses the **Vercel AI SDK** as part of the interface between the application logic and the language model.

A simplified interaction looks like:

```text
Analysis Result
      |
      v
Application Layer
      |
      v
Vercel AI SDK
      |
      v
LLM
      |
      v
User Response
```

This enables the analytical pipeline to be exposed through a conversational interface.

---

## Example Queries

Example questions supported by the general architecture include:

### Machine Behaviour

```text
How has the machine behaved today?
```

### Trend Analysis

```text
Show the recent trend of the machine.
```

### Statistical Analysis

```text
Are there unusual values in today's machine data?
```

### Forecasting

```text
What is the predicted machine value for the next 12 minutes?
```

The system determines the relevant processing operation before generating the final response.

---

## Design Principles

### 1. Separate computation from generation

Numerical analysis and prediction should be handled by dedicated analytical components whenever possible.

The LLM is primarily responsible for interpreting and communicating the results.

---

### 2. Process data before sending it to the LLM

Raw machine records are transformed into more useful structured representations before being supplied to the model.

These can include:

- Summaries
- Analytical results
- Forecasting outputs
- Structured machine information

---

### 3. Route requests to the appropriate operation

Different queries may require different processing paths.

For example:

```text
Trend Analysis
Statistical Analysis
Daily Behaviour Analysis
Machine Learning Forecasting
Template Response
```

The application can select the appropriate operation rather than treating the LLM as the only processing component.

---

### 4. Keep analytical outputs inspectable

Machine-learning predictions and analytical outputs remain conceptually separate from the natural-language response.

This means that the underlying result can be examined independently from how the LLM describes it.

---

## Simplified End-to-End Flow

```text
                     User Query
                         |
                         v
                 Query Recognition
                         |
                         v
                  Analysis Routing
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
           Trend     Behaviour      ML
          Analysis    Analysis   Forecasting
             |           |           |
             +-----------+-----------+
                         |
                         v
               Structured Results
                         |
                         v
                 Application Layer
                         |
                         v
                  Vercel AI SDK
                         |
                         v
                        LLM
                         |
                         v
                 Response to User
```

Machine information is handled through a separate processing path:

```text
Machine Records
      |
      +--> Run Identifier
      |
      +--> Timestamp
      |
      +--> Measurement
      |
      v
Preprocessing
      |
      v
Analysis / Prediction
```

---

## Technology Overview

The project demonstrates concepts and technologies including:

- Python
- Machine Learning
- Linear Regression
- Random Forest
- Time-series data processing
- Structured data preprocessing
- Statistical analysis
- Data visualization
- Large Language Models
- Natural-language interfaces
- Query routing
- Structured LLM context generation
- Vercel AI SDK

---

## Repository Scope

This repository does **not** contain the complete original project.

The original development environment was significantly larger and included additional:

- Source files
- Development assets
- Model-related resources
- Experimental implementations
- Integration code
- Intermediate artifacts
- Internal resources
- Environment-specific components

Publishing the complete original project was impractical because of its overall size and because some resources were specific to the original environment.

Some experimental components present in the original project may also not have been part of the final runtime pipeline.

For this reason, the public repository focuses only on the components that can be confidently documented and reproduced.

The public version intentionally excludes:

- Proprietary datasets
- Internal machine data
- API credentials and secrets
- Organization-specific integrations
- Internal infrastructure configuration
- Large development artifacts
- Unnecessary model files
- Experimental components that are not required to demonstrate the main system architecture

The goal is to preserve the main technical concepts without publishing unnecessary or sensitive resources.

---

## Original Project vs. Public Repository

```text
Original Development Project
        |
        |-- Larger codebase
        |-- Additional assets
        |-- Internal integrations
        |-- Model-related resources
        |-- Development experiments
        |-- Intermediate artifacts
        |-- Organization-specific resources
        |
        v
Filtering / Simplification
        |
        v
Public GitHub Version
        |
        |-- Core processing workflow
        |-- Data analysis
        |-- ML forecasting
        |-- LLM integration
        |-- Representative implementation
        |-- Documentation
```

This public repository should therefore be treated as a **representative and intentionally scaled-down implementation**, rather than a byte-for-byte copy of the original development project.

---

## Proof-of-Concept Limitations

This project was created primarily to demonstrate technical feasibility.

It should therefore **not be interpreted as a production-hardened or production-scale implementation**.

During the PoC phase, the priority was validating the technical concept rather than optimizing the project for:

- Long-term maintainability
- Large development teams
- High traffic
- Distributed execution
- Production observability
- Automated deployment
- Strict service boundaries

The original development process also involved rapid experimentation, meaning some parts of the codebase would benefit from further consolidation and restructuring.

---

## Path Toward Production Scale

Before scaling this system into a larger production environment, I would first perform a significant engineering cleanup and refactoring phase.

Areas that would need additional work include:

### Codebase Refactoring

The experimental PoC structure should be reorganized into clearer modules with well-defined responsibilities.

For example:

```text
Application
    |
    +--> Query Routing
    |
    +--> Data Access
    |
    +--> Analytics
    |
    +--> Forecasting
    |
    +--> LLM Integration
    |
    +--> API Layer
```

This would reduce coupling between components and make the system easier to maintain and test.

### Testing

A production version should introduce:

- Unit tests
- Integration tests
- API tests
- Regression tests
- LLM evaluation cases
- Machine-learning validation tests

### Error Handling

Production components would need more systematic handling for:

- Data-processing failures
- Model failures
- LLM API failures
- Invalid requests
- Timeouts
- External dependency failures

### Observability

Production operation would require structured:

- Logging
- Metrics
- Tracing
- Model-call monitoring
- Latency measurement
- Failure tracking

### Configuration Management

Environment-specific settings should be separated cleanly from application code and managed through dedicated configuration mechanisms.

### Security

A production system should introduce additional controls such as:

- Secret management
- Authentication
- Authorization
- Input validation
- Output validation
- Access controls
- Data protection policies

### Deployment

The individual components could also be prepared for more reproducible deployment using technologies such as:

- Containers
- CI/CD pipelines
- Cloud infrastructure
- Automated testing
- Environment-specific deployment configuration

### Scalability

If usage increased substantially, the architecture would need to consider:

- Concurrent requests
- Background processing
- Caching
- Database performance
- Queue-based workloads
- Service isolation
- Horizontal scaling

The PoC demonstrates the core technical idea.

Turning it into a large-scale production system would require a separate engineering phase focused on reliability, maintainability, security, observability, and operational scalability.

---

## Project Motivation

Machine operational data can become difficult to inspect manually as the amount of historical information grows.

Traditional dashboards can visualize this information, but users may still need to:

- Navigate through large amounts of data
- Locate relevant operating periods
- Interpret charts manually
- Compare machine behaviours
- Understand prediction results

This project explores an alternative interaction model:

```text
Machine Data
      +
Natural-Language Question
      |
      v
Automated Analysis
      |
      v
Machine Learning
      |
      v
Structured Results
      |
      v
LLM Interpretation
      |
      v
Human-Readable Response
```

The objective is not to replace traditional analytical methods with an LLM.

Instead, the LLM serves as an additional interface on top of explicit data-processing, analytical, and machine-learning components.

---

## What This Project Demonstrates

This repository demonstrates experience with:

- Designing an end-to-end AI application
- Independently owning a technical proof of concept
- Combining traditional machine learning with LLMs
- Processing structured machine data
- Designing query-routing logic
- Separating deterministic computation from generative output
- Building machine-learning forecasting workflows
- Integrating analytical results with an LLM interface
- Working with real-world operational data
- Experimenting with multiple technical approaches
- Translating a larger experimental system into a smaller public implementation
- Identifying the engineering work required to move a PoC toward production

---

## Disclaimer

This repository is intended to demonstrate the technical architecture, experimentation process, and engineering approach behind the project.

It is a **proof-of-concept implementation**, not a production-ready software package.

The public repository represents only a subset of the original development environment.

Any proprietary, confidential, organization-specific, sensitive, or unnecessarily large resources from the original project have intentionally been excluded.
