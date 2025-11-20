#!/usr/bin/env python3
"""
Create a minimal FAISS RAG index with PlantUML examples.

This creates a basic index for testing. For production use, add more examples
or use your existing training data.
"""

import json
import os
import sys

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("\nInstall with:")
    print("  pip install faiss-cpu sentence-transformers numpy")
    sys.exit(1)


# Minimal but representative PlantUML examples
EXAMPLES = [
    # Class Diagrams
    {
        "type": "class",
        "description": "Basic class diagram with relationships",
        "plantuml": """@startuml
class User {
  +id: int
  +name: string
  +email: string
  +login()
  +logout()
}

class Database {
  +connect()
  +query()
  +close()
}

class Logger {
  +log(message)
  +error(message)
}

User --> Database : uses
User --> Logger : logs to
@enduml"""
    },
    {
        "type": "class",
        "description": "Class diagram with inheritance",
        "plantuml": """@startuml
abstract class Animal {
  +name: string
  +age: int
  +makeSound()
}

class Dog extends Animal {
  +breed: string
  +bark()
}

class Cat extends Animal {
  +color: string
  +meow()
}
@enduml"""
    },
    
    # Sequence Diagrams
    {
        "type": "sequence",
        "description": "API request sequence",
        "plantuml": """@startuml
actor User
participant API
participant Service
database DB

User -> API: HTTP Request
API -> Service: Process Request
Service -> DB: Query Data
DB --> Service: Result Set
Service --> API: Formatted Response
API --> User: JSON Response
@enduml"""
    },
    {
        "type": "sequence",
        "description": "Authentication flow",
        "plantuml": """@startuml
actor User
participant Frontend
participant AuthService
database UserDB

User -> Frontend: Login Request
Frontend -> AuthService: Validate Credentials
AuthService -> UserDB: Check User
UserDB --> AuthService: User Data
AuthService --> Frontend: Auth Token
Frontend --> User: Success
@enduml"""
    },
    
    # Activity Diagrams
    {
        "type": "activity",
        "description": "User registration flow",
        "plantuml": """@startuml
start
:User submits form;
if (Valid data?) then (yes)
  :Create account;
  :Send confirmation email;
  :Log event;
  stop
else (no)
  :Show error message;
  stop
endif
@enduml"""
    },
    {
        "type": "activity",
        "description": "Data processing workflow",
        "plantuml": """@startuml
start
:Receive data;
:Validate input;
if (Valid?) then (yes)
  :Process data;
  :Store results;
else (no)
  :Log error;
endif
:Send notification;
stop
@enduml"""
    },
    
    # State Diagrams
    {
        "type": "state",
        "description": "Order state machine",
        "plantuml": """@startuml
[*] --> Created
Created --> Pending : submit
Pending --> Processing : approve
Processing --> Completed : finish
Processing --> Cancelled : cancel
Completed --> [*]
Cancelled --> [*]
@enduml"""
    },
    {
        "type": "state",
        "description": "Connection state",
        "plantuml": """@startuml
[*] --> Disconnected
Disconnected --> Connecting : connect()
Connecting --> Connected : success
Connecting --> Disconnected : fail
Connected --> Disconnected : disconnect()
@enduml"""
    },
    
    # Component Diagrams
    {
        "type": "component",
        "description": "System components",
        "plantuml": """@startuml
[Web UI] as web
[API Server] as api
[Database] as db
[Cache] as cache

web --> api : HTTP
api --> db : SQL
api --> cache : Redis
@enduml"""
    },
    {
        "type": "component",
        "description": "Microservices architecture",
        "plantuml": """@startuml
[User Service] as user
[Order Service] as order
[Payment Service] as payment
[Message Queue] as queue

user --> queue : events
order --> queue : events
payment --> queue : events
@enduml"""
    },
    
    # Deployment Diagrams
    {
        "type": "deployment",
        "description": "Application deployment",
        "plantuml": """@startuml
node "Web Server" {
  [Web App]
}

node "App Server" {
  [API Service]
  [Worker Process]
}

node "Database Server" {
  database "PostgreSQL"
}

[Web App] --> [API Service]
[API Service] --> [PostgreSQL]
@enduml"""
    },
    {
        "type": "deployment",
        "description": "Cloud deployment",
        "plantuml": """@startuml
cloud "AWS" {
  node "Load Balancer" {
    [ALB]
  }
  
  node "App Servers" {
    [App Instance 1]
    [App Instance 2]
  }
  
  database "RDS" {
    [PostgreSQL]
  }
}

[ALB] --> [App Instance 1]
[ALB] --> [App Instance 2]
[App Instance 1] --> [PostgreSQL]
[App Instance 2] --> [PostgreSQL]
@enduml"""
    },
    
    # Use Case Diagrams
    {
        "type": "usecase",
        "description": "System use cases",
        "plantuml": """@startuml
left to right direction
actor User
actor Admin

rectangle System {
  User --> (Login)
  User --> (View Profile)
  User --> (Update Settings)
  
  Admin --> (Manage Users)
  Admin --> (View Logs)
  Admin --> (Login)
}
@enduml"""
    },
    {
        "type": "usecase",
        "description": "E-commerce use cases",
        "plantuml": """@startuml
actor Customer
actor Seller

rectangle Platform {
  Customer --> (Browse Products)
  Customer --> (Place Order)
  Customer --> (Track Shipment)
  
  Seller --> (List Products)
  Seller --> (Process Orders)
}
@enduml"""
    },
    
    # Object Diagrams
    {
        "type": "object",
        "description": "Runtime objects",
        "plantuml": """@startuml
object user1 {
  id = 101
  name = "John"
  email = "john@example.com"
}

object db1 {
  host = "localhost"
  port = 5432
  connected = true
}

user1 --> db1 : uses
@enduml"""
    },
    {
        "type": "object",
        "description": "Object instances",
        "plantuml": """@startuml
object request {
  method = "GET"
  path = "/api/users"
  status = "pending"
}

object response {
  status_code = 200
  content_type = "application/json"
}

request --> response : generates
@enduml"""
    },
]


def create_rag_index(output_dir="rag", embed_model="nomic-embed-text"):
    """Create FAISS index and metadata from examples."""
    
    print("=" * 70)
    print("Creating Minimal PlantUML RAG Index")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[1/4] Created output directory: {output_dir}/")
    
    # Load embedding model
    print(f"\n[2/4] Loading embedding model: {embed_model}")
    print("      (This may download the model on first run...)")
    try:
        model = SentenceTransformer(embed_model)
        print(f"      ✓ Model loaded (dimension: {model.get_sentence_embedding_dimension()})")
    except Exception as e:
        print(f"      ✗ Failed to load model: {e}")
        sys.exit(1)
    
    # Create embeddings
    print(f"\n[3/4] Creating embeddings for {len(EXAMPLES)} examples...")
    texts = []
    for ex in EXAMPLES:
        # Embed the type and description for better retrieval
        text = f"plantuml {ex['type']} diagram {ex['description']}"
        texts.append(text)
    
    try:
        embeddings = model.encode(texts, normalize_embeddings=True)
        embeddings = embeddings.astype('float32')
        print(f"      ✓ Created embeddings: {embeddings.shape}")
    except Exception as e:
        print(f"      ✗ Failed to create embeddings: {e}")
        sys.exit(1)
    
    # Build FAISS index
    print("\n[4/4] Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
    index.add(embeddings)
    
    # Save index
    index_path = os.path.join(output_dir, "faiss.index")
    faiss.write_index(index, index_path)
    print(f"      ✓ Saved index: {index_path}")
    
    # Save metadata
    meta_path = os.path.join(output_dir, "faiss_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(EXAMPLES, f, indent=2)
    print(f"      ✓ Saved metadata: {meta_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUCCESS: RAG Index Created")
    print("=" * 70)
    print(f"Location: {os.path.abspath(output_dir)}/")
    print(f"Files:")
    print(f"  - faiss.index     ({os.path.getsize(index_path):,} bytes)")
    print(f"  - faiss_meta.json ({os.path.getsize(meta_path):,} bytes)")
    print(f"\nIndex statistics:")
    print(f"  - Total examples:  {len(EXAMPLES)}")
    print(f"  - Vector dimension: {dimension}")
    print(f"  - Index type:      Inner Product (normalized)")
    
    # Show examples per type
    type_counts = {}
    for ex in EXAMPLES:
        type_counts[ex['type']] = type_counts.get(ex['type'], 0) + 1
    
    print(f"\nExamples per diagram type:")
    for dtype, count in sorted(type_counts.items()):
        print(f"  - {dtype.ljust(12)} {count}")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Test RAG retrieval:")
    print("   python -c \"from core.rag_retriever import RagRetriever; \\")
    print("              r = RagRetriever('rag/faiss.index', 'rag/faiss_meta.json'); \\")
    print("              print(r.search('class diagram', 2))\"")
    print("")
    print("2. Generate diagrams:")
    print("   python repo_to_diagrams_local_vllm.py \\")
    print("     --input ./your_repo \\")
    print("     --model openai/gpt-oss-120b \\")
    print("     --tp 4 \\")
    print("     --faiss-index rag/faiss.index \\")
    print("     --faiss-meta rag/faiss_meta.json")
    print("=" * 70)


def verify_index(index_path="rag/faiss.index", meta_path="rag/faiss_meta.json"):
    """Verify the created index works."""
    print("\n" + "=" * 70)
    print("Verifying Index...")
    print("=" * 70)
    
    try:
        # Load index
        index = faiss.read_index(index_path)
        print(f"✓ Index loaded: {index.ntotal} vectors")
        
        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"✓ Metadata loaded: {len(meta)} entries")
        
        # Quick search test
        model = SentenceTransformer("nomic-embed-text")
        query_vec = model.encode(["class diagram example"], normalize_embeddings=True).astype('float32')
        scores, indices = index.search(query_vec, 3)
        
        print("\nTest query: 'class diagram example'")
        print(f"Top results:")
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0 and idx < len(meta):
                ex = meta[idx]
                print(f"  {i+1}. [{ex['type']}] {ex['description']} (score: {score:.3f})")
        
        print("\n✓ Index verification successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create minimal PlantUML RAG index for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This creates a minimal but functional RAG index for PlantUML generation.

For production use:
  1. Add more examples (target: 100-500 per diagram type)
  2. Use diverse, high-quality PlantUML examples
  3. Or use your existing fine-tuning training data

The index includes 2 examples per diagram type (16 total).
        """
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="rag",
        help="Output directory for index files (default: ./rag)",
    )
    parser.add_argument(
        "--embed-model", "-m",
        default="nomic-embed-text",
        help="Embedding model to use (default: nomic-embed-text)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip index verification after creation",
    )
    
    args = parser.parse_args()
    
    # Create index
    create_rag_index(args.output_dir, args.embed_model)
    
    # Verify
    if not args.no_verify:
        index_path = os.path.join(args.output_dir, "faiss.index")
        meta_path = os.path.join(args.output_dir, "faiss_meta.json")
        verify_index(index_path, meta_path)


if __name__ == "__main__":
    main()
