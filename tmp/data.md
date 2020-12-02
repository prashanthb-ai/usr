# Data layer

* What are the datasystems producing data?
* What is the data diversity? (generic schema across blob, search index, nosql)
    - Datasets vs Users/Groups
* What are the core data operations? (CRUD, search, graph traversal)

Observations
---
* Standardized event interface decouples producers from consumers
* Value in loal subgraphs
    - MDS with a dataset and its schema
    - Where to store ACLs for datasets
    - Real value in global model?
    - Graph traversal of metadata is important

DDecisions
---
* Generic or specific types, model first?
* Integration: crawling, rest, pub/sub? read after write consistency?
    - Teams don't write to the db directly, publish into the bus instead
    - REST is great for read after write consistency
* Single source or replicated or federated
    - Single: everyone writes into one
    - Replicated: aggregate into 1, there are many
    - Federated: one, routes to the relevant
        * Existing systems use kafka to adapt
        * New systems: ?
        * Federated:
            - Call it with a dataset name
            - Gives you a hive table of kafka topic
            - How to traverse this data?
        * Federation gets you off the ground, doesn't let you traverse
* Alternatives
    - Do you need strong types?
    - Does it need to consume operations metadata (from workflow scheduler)?

## Metadata consciousness

* Services generate data written into databases
* Database changes are written to kafka
* Services generate events that are also sent to Kafka
* DW: HDFS + Hadoop - standardizing of data, analytics
* Derived datasets - features, models (optimized datastores)
* Optimized datastores render experiences back on client
    - Reports etc

## Phases

P1
--
* Crawl all the data you can
    - What datasets do we have there? what derives from what (lineage)?
    - Fetch data from everywhere: Tera data, log files, crawl hive etc
* Parse all logs you can
* ETL all data into a db (opinionated)
* Add a search index and a lookup store
* App to serve this info, make it useful - wherehows

* Issues
    - fragile pipelines
    - broken lineage
    - batch files that need crawling - freshness issues
    - surfacing something in the app takes a long time

* Evolution
    - What are our datasystems: espresso, kafka, pinot etc...
    - Who owns datasets, where are they copied, what is their schema

P2
--
* Make producers responsible for their data
    - This is the metadata change event
    - You must publish this change event whenever you have new data
    - Culture change - teams need to crawl their own datasets and produce events
* Allow clients to consume metadata without needing the app to evolve
* Some people want to write using REST and some just want to publish an event

* Issues
    - teams are accountable for custom etl
    - aggregate metadata and add a compliance layer on top
    - New thing added to metadata change event (big event)
    - Source vs reflection of truth
    - Standardized event interface decouples producers from consumers

P3
--
* Ownership of datasets
    - Who owns, what else are they producing, what dashboards are these used in
        - graph of relationships
* Apps produce events
* Events processed and routed to services/Databases
* Diverse databases for diverse use cases (Dataset vs users/groups)
    - Each database publishes its own output events
    - Output events are indexed for search and graph traversal
    - Need to audit the entire metadata graph
* Metadata graph is used as core data model for analytics?

## Metadata use cases

* Mesh of metadata services
* Governance use cases on the mesh
* Search and discovery
* AI metadata - around experiments, features and model training
    - reproducibility, auditability, visibility, consistency
* Compliant data management

## Alternatives

* Hive metastore: limited to Hadoop dataset, focused on query planning
* Apache atlas: generic model, missing strong typing on top
* Marquez: strongly opinionated model, focussed on data pipeline metadata
    - Understand data pipeline, get data from airflow
* Ground: generic model, missing stong typeing on top, research prototype
* Amundsen: web app with a FE, write through to hive metastore

