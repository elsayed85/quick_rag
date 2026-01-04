"""
Generate a comprehensive PDF document for embedding and semantic search testing.
Topic: Introduction to Distributed Systems
"""
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

file_path = "/Users/elsayed/general_projects_github/books_rag/intro_to_distributed_systems.pdf"

doc = SimpleDocTemplate(file_path, pagesize=A4)
styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=16,
    spaceAfter=12,
    textColor='#1a1a1a'
)

story = []

# Book Title
story.append(Paragraph("<b>Introduction to Distributed Systems</b>", styles["Title"]))
story.append(Paragraph("<i>A Comprehensive Guide to Modern Computing Architectures</i>", styles["Heading2"]))
story.append(PageBreak())

chapters = [
    ("Chapter 1: What Are Distributed Systems?",
     """A distributed system is a collection of independent computers that appears to its users as a single coherent system. 
     These autonomous computing elements are connected through a network and communicate by passing messages to one another. 
     The fundamental goal is to make the complexity of the distributed nature transparent to end users, who should perceive 
     the system as a single, unified platform.
     
     Distributed systems have become ubiquitous in modern computing. From cloud computing platforms like Amazon Web Services 
     and Google Cloud to social media applications, online banking systems, and streaming services, distributed systems power 
     the infrastructure that billions of people rely on daily. The rise of the internet and the exponential growth of data 
     have made distributed computing not just beneficial but essential.
     
     Key motivations for building distributed systems include resource sharing, scalability, fault tolerance, and performance. 
     By distributing workload across multiple machines, systems can handle more users and process more data than any single 
     computer could manage. Additionally, if one component fails, others can continue operating, providing higher availability 
     than centralized systems.
     
     However, distributed systems introduce significant complexity. Developers must deal with network latency, partial failures, 
     concurrency, and the lack of a global clock. These challenges require sophisticated algorithms and careful system design 
     to ensure correctness and performance."""),
    
    ("Chapter 2: Fundamental Characteristics",
     """Distributed systems exhibit several fundamental characteristics that distinguish them from centralized systems. 
     Understanding these properties is crucial for designing and reasoning about distributed applications.
     
     First, concurrency is inherent in distributed systems. Multiple processes execute simultaneously on different machines, 
     accessing and modifying shared resources. This concurrent execution can lead to race conditions and inconsistencies if 
     not properly managed through synchronization mechanisms and coordination protocols.
     
     Second, there is no global clock. In a single computer, processes can rely on a shared system clock to order events. 
     In distributed systems, each machine has its own clock, and these clocks can drift relative to one another. This makes 
     determining the order of events across different machines challenging, leading to the development of logical clocks and 
     vector clocks to establish causal relationships between events.
     
     Third, independent failures are a reality. Unlike centralized systems where failures are typically total (the entire 
     system goes down), distributed systems experience partial failures. Some components may fail while others continue 
     operating normally. The system must be designed to detect failures, recover from them, and continue providing service 
     despite these partial failures.
     
     Fourth, heterogeneity is common. Distributed systems often comprise machines with different hardware architectures, 
     operating systems, programming languages, and network protocols. Middleware layers help bridge these differences, 
     providing uniform interfaces and abstractions that hide the underlying heterogeneity from application developers."""),
    
    ("Chapter 3: Communication in Distributed Systems",
     """Communication is the foundation of distributed systems. Processes running on different machines must exchange 
     information to coordinate their actions and share data. The communication paradigm chosen significantly impacts 
     system design, performance, and reliability.
     
     Message passing is the primary communication mechanism. Unlike shared memory in single-machine systems, distributed 
     processes communicate by sending and receiving messages over a network. These messages can be sent using various 
     protocols, each with different guarantees and performance characteristics.
     
     Synchronous communication means the sender blocks until the receiver acknowledges receipt of the message. This 
     simplifies reasoning about program behavior because the sender knows exactly when the message has been delivered. 
     However, it can lead to reduced performance as processes wait for acknowledgments, and it makes the system vulnerable 
     to failures—if the receiver crashes, the sender may block indefinitely.
     
     Asynchronous communication allows the sender to continue executing immediately after sending a message, without 
     waiting for acknowledgment. This improves performance and concurrency but makes programming more complex. The sender 
     doesn't know when (or if) the message will be delivered, requiring additional mechanisms to handle failures and 
     ensure reliability.
     
     Remote Procedure Call (RPC) provides a higher-level abstraction, making distributed communication look like local 
     procedure calls. RPC frameworks handle marshaling of parameters, network transmission, and unmarshaling of results. 
     Popular RPC systems include gRPC, Apache Thrift, and Java RMI. While RPC simplifies distributed programming, it can 
     obscure the realities of network communication, such as latency and potential failures."""),
    
    ("Chapter 4: Network Performance Metrics",
     """Understanding network performance is essential for building efficient distributed systems. Two key metrics are 
     latency and throughput, and the relationship between them determines overall system performance.
     
     Latency is the time delay between initiating a network request and receiving a response. It consists of several 
     components: transmission delay (time to push bits onto the network), propagation delay (time for signals to travel 
     through the medium), queuing delay (time waiting in network buffers), and processing delay (time for routers and 
     endpoints to process packets). Latency is typically measured in milliseconds and can vary significantly based on 
     network conditions and geographic distance.
     
     Throughput measures the amount of data transmitted successfully per unit time, typically expressed in bits per second 
     or bytes per second. High throughput is crucial for applications that transfer large amounts of data, such as video 
     streaming, file transfers, and database replication. Throughput is affected by bandwidth (the maximum capacity of the 
     network link) and by network congestion, packet loss, and protocol overhead.
     
     The bandwidth-delay product represents the amount of data that can be "in flight" in the network at any given time. 
     It's calculated by multiplying bandwidth by round-trip latency. Understanding this product is crucial for optimizing 
     protocols, especially for long-distance, high-bandwidth connections.
     
     Jitter refers to variation in latency over time. Some applications, particularly real-time systems like voice and 
     video calls, are sensitive to jitter. Buffering and smoothing techniques can mitigate jitter's effects but may 
     introduce additional latency."""),
    
    ("Chapter 5: Data Consistency Models",
     """Consistency models define the contract between the distributed system and applications regarding the visibility 
     and ordering of updates to shared data. Different consistency models offer different trade-offs between performance, 
     availability, and the guarantees provided to applications.
     
     Strong consistency (also called linearizability) provides the strongest guarantee: all operations appear to execute 
     atomically in some total order, and once a write completes, all subsequent reads will see that write or a later one. 
     Strong consistency makes distributed systems behave like single-machine systems, simplifying application logic. However, 
     it comes at a cost: achieving strong consistency typically requires coordination between nodes, which adds latency and 
     can reduce availability during network partitions.
     
     Sequential consistency is slightly weaker: operations appear to execute in some sequential order, and operations from 
     each individual process appear in program order within that sequence. However, different processes may observe operations 
     in different orders, as long as each process's view is consistent with program order.
     
     Causal consistency preserves the order of causally related operations. If operation A happens before operation B (meaning 
     B could have been influenced by A), then all processes observe A before B. Unrelated concurrent operations may be observed 
     in different orders by different processes. Causal consistency strikes a balance between strong consistency and performance.
     
     Eventual consistency is a weak model that guarantees only that if no new updates are made, eventually all replicas will 
     converge to the same state. This model allows replicas to diverge temporarily, enabling high availability and performance. 
     Many large-scale systems, including Amazon's Dynamo and Apache Cassandra, use eventual consistency for its scalability 
     benefits, though it places a burden on applications to handle conflicts and inconsistencies."""),
    
    ("Chapter 6: Replication Strategies",
     """Replication involves maintaining multiple copies of data across different nodes in a distributed system. It's a 
     fundamental technique for improving availability, fault tolerance, and performance, but it introduces challenges in 
     maintaining consistency across replicas.
     
     The primary motivation for replication is fault tolerance. If one replica fails, others can continue serving requests, 
     ensuring the system remains available. Replication also improves performance by allowing read requests to be served by 
     the nearest or least-loaded replica, reducing latency and distributing load.
     
     Primary-backup replication (also called master-slave) designates one replica as the primary, which handles all writes. 
     The primary propagates updates to backup replicas, which serve read requests. This approach simplifies consistency 
     management since there's a single source of truth for writes. However, the primary becomes a single point of failure 
     and a potential bottleneck for write operations.
     
     Multi-primary replication allows multiple replicas to accept writes concurrently. This improves write availability and 
     performance but introduces the challenge of conflict resolution when different replicas receive conflicting updates. 
     Conflict resolution strategies include last-write-wins (based on timestamps), application-specific merge logic, or 
     exposing conflicts to users for manual resolution.
     
     Quorum-based replication requires a majority of replicas to agree on operations. For a system with N replicas, write 
     operations must be acknowledged by W replicas, and read operations must contact R replicas, where R + W > N. This 
     ensures that reads will see the most recent write. Quorum systems provide tunable consistency and availability trade-offs.
     
     State machine replication treats each replica as a deterministic state machine. All replicas process the same sequence 
     of operations in the same order, ensuring they remain consistent. This approach requires a consensus protocol to agree 
     on the operation sequence."""),
    
    ("Chapter 7: Fault Tolerance and Recovery",
     """Fault tolerance is the ability of a system to continue operating correctly despite failures of some components. 
     In distributed systems, where failures are inevitable, fault tolerance is not optional—it's a fundamental requirement.
     
     Failures in distributed systems take many forms. Crash failures occur when a process stops executing, either due to 
     software bugs, hardware failures, or power loss. Omission failures happen when messages are lost in the network. 
     Byzantine failures are the most severe, where components exhibit arbitrary, potentially malicious behavior.
     
     Detecting failures in distributed systems is challenging. Unlike centralized systems, there's often no definitive way 
     to distinguish between a crashed process and a slow process or network delay. Timeout-based failure detectors assume 
     a process has failed if it doesn't respond within a specified time. However, this can lead to false positives if the 
     network is temporarily slow.
     
     Recovery strategies depend on the type of failure and system requirements. For crash failures, checkpointing allows 
     processes to periodically save their state, enabling them to restart from the last checkpoint rather than from scratch. 
     Logging records all operations, allowing replay to reconstruct state after a failure.
     
     Redundancy is key to fault tolerance. By replicating data and computation across multiple nodes, the system can continue 
     operating even if some nodes fail. However, redundancy alone isn't sufficient—the system needs coordination mechanisms 
     to ensure replicas remain consistent and to reconfigure the system when failures occur.
     
     Graceful degradation allows systems to continue operating with reduced functionality when failures occur, rather than 
     completely shutting down. For example, a social media platform might disable video uploads during a storage system 
     failure while keeping other features operational."""),
    
    ("Chapter 8: Consensus Algorithms",
     """Consensus is one of the most fundamental problems in distributed systems: how can a group of processes agree on a 
     single value, even in the presence of failures? Consensus is essential for many distributed system operations, including 
     leader election, atomic commitment, and state machine replication.
     
     The FLP impossibility result, proven by Fischer, Lynch, and Paterson, shows that in an asynchronous distributed system 
     with even one faulty process, there's no deterministic algorithm that guarantees consensus in bounded time. This 
     theoretical limitation doesn't mean consensus is impossible in practice—real systems use timeouts, randomization, or 
     assumptions about network behavior to achieve consensus.
     
     Paxos, proposed by Leslie Lamport, is one of the most well-known consensus algorithms. It operates in phases: prepare, 
     promise, propose, and accept. A process acting as a proposer suggests a value, and acceptors vote on it. Once a majority 
     of acceptors agree, consensus is reached. Paxos is provably correct and tolerates failures, but it's notoriously difficult 
     to understand and implement correctly.
     
     Raft was designed as a more understandable alternative to Paxos. It divides the consensus problem into leader election, 
     log replication, and safety. At any time, one server acts as the leader, receiving client requests and replicating them 
     to follower servers. Raft uses randomized timeouts for leader election and guarantees that committed entries won't be lost.
     
     Byzantine fault-tolerant consensus algorithms like PBFT (Practical Byzantine Fault Tolerance) can tolerate arbitrary 
     failures, including malicious behavior. These algorithms require more communication and can tolerate fewer failures 
     (at most f failures out of 3f+1 replicas) than crash-tolerant algorithms, but they're essential for systems that can't 
     trust all participants, such as blockchain networks.
     
     Consensus algorithms are typically implemented as part of coordination services like Apache ZooKeeper, etcd, or Consul, 
     which provide primitives for distributed coordination, configuration management, and leader election."""),
    
    ("Chapter 9: Distributed Databases",
     """Distributed databases store data across multiple machines while providing applications with a unified view and 
     query interface. They're essential for managing the massive data volumes generated by modern applications while 
     providing high availability and performance.
     
     Horizontal partitioning (sharding) divides data across nodes based on a partition key. For example, user data might 
     be partitioned by user ID, with each shard storing a subset of users. Sharding distributes load and storage requirements 
     but complicates queries that need to access multiple shards. Choosing the right partition key is crucial—poor choices 
     can lead to unbalanced load distribution (hot spots).
     
     The CAP theorem, proven by Eric Brewer, states that a distributed system can provide at most two of three guarantees: 
     Consistency (all nodes see the same data), Availability (every request receives a response), and Partition tolerance 
     (the system continues operating despite network partitions). Since network partitions are inevitable in distributed 
     systems, databases must choose between consistency and availability during partitions.
     
     CP systems (consistent and partition-tolerant) prioritize consistency over availability. During a partition, some nodes 
     may become unavailable to ensure that all available nodes return consistent data. Traditional relational databases and 
     systems like HBase fall into this category.
     
     AP systems (available and partition-tolerant) prioritize availability over consistency. All nodes remain available during 
     partitions, but they may return stale or conflicting data. Systems like Cassandra and Riak are AP systems, using eventual 
     consistency to reconcile divergent replicas after partitions heal.
     
     NewSQL databases aim to provide ACID guarantees (like traditional SQL databases) while scaling horizontally (like NoSQL 
     databases). Systems like Google Spanner, CockroachDB, and VoltDB use sophisticated protocols to provide strong consistency 
     across distributed nodes, though this typically comes with higher latency than eventual consistency systems."""),
    
    ("Chapter 10: Scalability Principles",
     """Scalability is the ability of a system to handle increasing load by adding resources. Designing for scalability 
     requires understanding the different dimensions of scale and the techniques to achieve each.
     
     Vertical scaling (scaling up) involves adding more resources to individual machines—more CPU, memory, or storage. 
     This approach is simple and doesn't require changes to application architecture. However, it has hard limits: there's 
     a maximum size for any single machine, and the cost typically increases non-linearly at higher scales.
     
     Horizontal scaling (scaling out) involves adding more machines to distribute load. This approach can scale nearly 
     indefinitely and uses commodity hardware, reducing costs. However, it requires application architecture changes to 
     distribute work across machines and handle the complexities of distributed computing.
     
     Load balancing distributes incoming requests across multiple servers to prevent any single server from becoming 
     overwhelmed. Load balancers can use various strategies: round-robin (cycling through servers), least connections 
     (sending requests to the server with fewest active connections), or based on server load. DNS-based load balancing 
     distributes traffic at the DNS level, while application-level load balancers (like HAProxy or NGINX) offer more 
     sophisticated routing.
     
     Caching stores frequently accessed data in fast-access storage to reduce load on backend systems. Caching can occur 
     at multiple levels: client-side (browser cache), CDN (content delivery network), application-level (memcached, Redis), 
     and database-level (query cache). Cache invalidation—keeping cached data consistent with the source—is notoriously 
     difficult, as Phil Karlton's famous quote suggests: "There are only two hard things in Computer Science: cache 
     invalidation and naming things."
     
     Asynchronous processing decouples components, allowing them to scale independently. Message queues like RabbitMQ, 
     Apache Kafka, and AWS SQS enable producers to submit work without waiting for consumers to process it, smoothing 
     traffic spikes and allowing consumers to scale based on queue depth."""),
    
    ("Chapter 11: Security in Distributed Systems",
     """Security in distributed systems is multifaceted, encompassing authentication (verifying identity), authorization 
     (controlling access), confidentiality (protecting data from unauthorized disclosure), integrity (ensuring data hasn't 
     been tampered with), and availability (ensuring services remain accessible).
     
     Authentication in distributed systems is challenging because credentials must be verified across network boundaries. 
     Password-based authentication is common but vulnerable to interception and brute-force attacks. Multi-factor 
     authentication (MFA) adds additional verification factors, significantly improving security. Public key infrastructure 
     (PKI) uses cryptographic certificates to verify identities without transmitting secrets over the network.
     
     Token-based authentication, exemplified by OAuth 2.0 and JWT (JSON Web Tokens), has become standard for web services. 
     After initial authentication, clients receive a token that they include in subsequent requests. Tokens can be stateless 
     (containing all necessary information, cryptographically signed) or stateful (requiring server-side session lookup).
     
     Authorization determines what authenticated users can access. Role-Based Access Control (RBAC) assigns permissions to 
     roles and users to roles, simplifying permission management. Attribute-Based Access Control (ABAC) makes decisions based 
     on attributes of users, resources, and environmental conditions, providing more fine-grained control.
     
     Encryption protects data confidentiality. Transport Layer Security (TLS) encrypts data in transit between clients and 
     servers, preventing eavesdropping and tampering. Encryption at rest protects stored data from unauthorized access if 
     physical storage is compromised. End-to-end encryption ensures only communicating endpoints can decrypt data, even if 
     intermediate servers are compromised.
     
     Distributed systems face unique security challenges. Network communication creates attack surfaces for interception and 
     tampering. Compromised nodes can launch insider attacks. Distributed denial-of-service (DDoS) attacks overwhelm systems 
     with traffic from many sources. Defense in depth—multiple layers of security controls—is essential for protecting 
     distributed systems."""),
    
    ("Chapter 12: Monitoring and Observability",
     """Observability is the ability to understand a system's internal state based on its external outputs. In distributed 
     systems, where failures can cascade across components and root causes may be far from symptoms, observability is crucial 
     for operations and debugging.
     
     The three pillars of observability are logs, metrics, and traces. Each provides different insights into system behavior, 
     and together they enable comprehensive understanding of distributed systems.
     
     Logs are timestamped records of events within a system. They capture details about requests, errors, state changes, and 
     other significant occurrences. Structured logging, where logs are formatted as key-value pairs or JSON rather than 
     free-form text, makes logs easier to query and analyze. Centralized logging systems like Elasticsearch, Splunk, or Loki 
     aggregate logs from all components, enabling cross-service analysis.
     
     Metrics are numerical measurements of system behavior over time. They include counters (monotonically increasing values 
     like request counts), gauges (point-in-time values like memory usage), and histograms (distributions of values like 
     request latencies). Time-series databases like Prometheus, InfluxDB, or TimescaleDB store metrics efficiently and support 
     powerful querying and alerting. Dashboards visualize metrics, helping operators understand system health at a glance.
     
     Distributed tracing tracks requests as they flow through multiple services. Each service adds spans to a trace, recording 
     timing information and metadata. Traces reveal the path requests take, identify bottlenecks, and help diagnose failures 
     that span multiple services. OpenTelemetry provides a standard for instrumenting applications for tracing, and systems 
     like Jaeger and Zipkin store and visualize traces.
     
     Service mesh technologies like Istio and Linkerd automatically instrument service-to-service communication, providing 
     observability without requiring application code changes. They collect metrics, generate traces, and can enforce policies 
     for security and traffic management."""),
    
    ("Chapter 13: Case Studies in Distributed Systems",
     """Real-world distributed systems demonstrate how theoretical concepts and design patterns are applied to solve practical 
     problems at scale. Examining these systems provides valuable insights for designing new distributed applications.
     
     Google's infrastructure exemplifies distributed systems engineering. The Google File System (GFS) pioneered large-scale 
     distributed storage, using commodity hardware and replication for fault tolerance. MapReduce provided a simple programming 
     model for processing massive datasets across thousands of machines. Bigtable demonstrated how to build a scalable NoSQL 
     database, while Spanner showed that globally distributed databases could provide strong consistency through novel use of 
     synchronized clocks.
     
     Apache Kafka has become the de facto standard for event streaming. It provides a distributed commit log that multiple 
     producers can write to and multiple consumers can read from, with strong ordering guarantees within partitions. Kafka's 
     architecture—partitioned topics, consumer groups, and replication—enables it to handle trillions of messages per day. 
     Organizations use Kafka for real-time analytics, event sourcing, log aggregation, and as the backbone of event-driven 
     architectures.
     
     Kubernetes has revolutionized container orchestration and application deployment. It manages containerized workloads across 
     clusters of machines, handling scheduling, scaling, load balancing, and failure recovery. Kubernetes's declarative model 
     allows operators to specify desired state, and controllers continuously work to make actual state match desired state. Its 
     extensibility through custom resources and operators has created a rich ecosystem of tools and platforms.
     
     Netflix's microservices architecture demonstrates patterns for building resilient distributed systems. Their Chaos Engineering 
     practices, including the famous Chaos Monkey that randomly terminates production instances, validate that systems can handle 
     failures. They've open-sourced many tools, including Hystrix for circuit breaking, Eureka for service discovery, and Zuul 
     for API gateway functionality.
     
     Blockchain systems like Bitcoin and Ethereum represent a different class of distributed systems—ones designed to operate 
     without trusted parties. They use consensus algorithms (Proof of Work, Proof of Stake) that tolerate Byzantine failures, 
     enabling participants who don't trust each other to agree on a shared ledger. These systems trade performance for security 
     and decentralization, processing far fewer transactions than centralized systems but operating without central authority."""),
    
    ("Chapter 14: Emerging Trends and Future Directions",
     """The field of distributed systems continues to evolve, driven by new use cases, technologies, and scale requirements. 
     Several trends are shaping the future of distributed computing.
     
     Edge computing brings computation and storage closer to data sources and end users. Instead of sending all data to 
     centralized cloud datacenters, processing occurs on edge devices or regional edge servers. This reduces latency (critical 
     for applications like autonomous vehicles and augmented reality), conserves bandwidth, and enables operation during network 
     disconnections. However, edge computing introduces challenges in resource management, security, and maintaining consistency 
     across geographically distributed edge nodes.
     
     Serverless computing abstracts away infrastructure management, allowing developers to focus on application logic. Functions 
     are executed in response to events, with the platform handling scaling, load balancing, and resource allocation. While 
     serverless simplifies development and can reduce costs for variable workloads, it introduces new challenges: cold starts 
     (latency when functions haven't been recently invoked), limited execution duration, and vendor lock-in. Research continues 
     on optimizing function execution, providing better programming models, and enabling function-to-function communication.
     
     WebAssembly (Wasm) is extending beyond browsers to become a portable, secure runtime for server-side applications. It 
     promises "write once, run anywhere" with near-native performance. In distributed systems, Wasm enables secure multi-tenancy 
     (isolating different customers' code), polyglot programming (using multiple languages in one application), and portable 
     functions (running the same code in cloud, edge, and browser).
     
     Machine learning is both a user of distributed systems and is changing how they're built. Training large models requires 
     distributed computation across many GPUs. Serving models at scale requires distributed inference systems. ML is also being 
     applied to distributed systems problems: predicting failures, optimizing resource allocation, detecting anomalies, and even 
     generating database indexes.
     
     Quantum computing may eventually impact distributed systems, though practical applications remain distant. Quantum networks 
     could enable secure communication through quantum key distribution. Quantum algorithms might solve certain distributed 
     coordination problems more efficiently. However, building reliable quantum systems poses immense engineering challenges."""),
    
    ("Chapter 15: Conclusion and Key Takeaways",
     """Distributed systems are fundamental to modern computing, powering the applications and services we use daily. While they 
     offer tremendous benefits—scalability, fault tolerance, resource sharing, and performance—they also introduce significant 
     complexity.
     
     The CAP theorem reminds us that we can't have perfect consistency, availability, and partition tolerance simultaneously. 
     System designers must make trade-offs based on application requirements. Understanding these trade-offs is crucial for 
     building appropriate solutions.
     
     There's no one-size-fits-all architecture for distributed systems. The right design depends on many factors: consistency 
     requirements, scale, latency sensitivity, failure modes, and operational constraints. Strong consistency simplifies application 
     logic but limits scalability. Eventual consistency enables massive scale but pushes complexity to applications. Choose based 
     on actual requirements, not assumptions.
     
     Failure is not exceptional—it's normal. In large-scale distributed systems, something is always failing. Design for failure 
     from the beginning: use timeouts, implement retries with exponential backoff, build circuit breakers, replicate data, and 
     plan for disaster recovery. Test failure scenarios regularly through chaos engineering.
     
     Observability is not optional. You can't fix what you can't see. Invest in comprehensive logging, metrics, and tracing from 
     the start. When (not if) problems occur, good observability dramatically reduces time to resolution.
     
     Security must be built in, not bolted on. Distributed systems have large attack surfaces. Use defense in depth: authenticate 
     and authorize all requests, encrypt sensitive data, minimize trust boundaries, and follow the principle of least privilege.
     
     The field of distributed systems continues to evolve. New technologies and paradigms emerge regularly. However, fundamental 
     principles remain relevant: understanding consistency models, designing for failure, managing trade-offs, and reasoning about 
     distributed state. These concepts will serve you well regardless of which specific technologies you use.
     
     Building distributed systems is challenging but rewarding. The systems you build can scale to serve millions of users, remain 
     available despite failures, and adapt to changing requirements. As computing continues to become more distributed—from cloud 
     to edge to IoT devices—expertise in distributed systems becomes increasingly valuable. Whether you're building microservices, 
     distributed databases, event streaming platforms, or entirely new categories of applications, the principles covered in this 
     book provide a foundation for success.""")
]

for title, content in chapters:
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Split content into paragraphs
    paragraphs = content.strip().split('\n     ')
    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), styles["BodyText"]))
            story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())

# Build the PDF
doc.build(story)

print(f"PDF generated successfully: {file_path}")
print(f"Total chapters: {len(chapters)}")
