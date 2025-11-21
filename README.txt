python util/build_faiss_rag.py \
  --corpus rag/plantuml_rag_corpus.jsonl \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

python util/test_faiss_query.py \
  "sequence diagram for async worker queues" \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json

python repo_to_diagrams.py \
  --input /home/mclark/Code/MachineLearning/Celebrity/celebrity-rec-rabbitmq \
  --output /home/mclark/Code/MachineLearning/Celebrity/celebrity-rec-rabbitmq/uml \
  --faiss-index rag/faiss.index \
  --faiss-meta rag/faiss_meta.json \
  --ollama-url http://172.32.1.250:11434 \
  --llm-model llama4:maverick
