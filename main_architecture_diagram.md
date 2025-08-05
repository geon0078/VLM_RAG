# Main.py Architecture - VLM RAG Pipeline

This mermaid diagram represents the architecture and data flow of the `main.py` VLM RAG pipeline.

```mermaid
flowchart TD
    %% Input Stage
    A[User Input:<br/>- Image Path<br/>- Question] --> B[Load Image<br/>PIL.Image.open]
    
    %% Model Initialization (Setup Phase)
    subgraph INIT ["🔧 Model Initialization"]
        I1[Ovis VLM Model<br/>AIDC-AI/Ovis2-8B]
        I2[SentenceTransformer<br/>BAAI/bge-m3]
        I3[ChromaDB Client<br/>korean_knowledge_base]
    end
    
    %% Stage 1: Image Description
    subgraph VLM ["🖼️ VLM Processing Stage"]
        B --> C[Preprocess Image<br/>+ Image Description Prompt<br/>'이 이미지를 보고 간단히 설명해 주세요']
        C --> D[Ovis VLM Generation<br/>max_new_tokens=256]
        D --> E[Image Description<br/>Generated Text]
    end
    
    %% Stage 2: Vector Retrieval
    subgraph RETRIEVAL ["🔍 Vector Retrieval Stage"]
        E --> F[Encode Description<br/>SentenceTransformer]
        F --> G[Query ChromaDB<br/>retrieve_documents(k=3)]
        G --> H[Retrieved Documents<br/>Top-K Context]
    end
    
    %% Stage 3: Final Answer Generation
    subgraph LLM ["🤖 LLM Answer Generation"]
        H --> I[Combine Context + Question<br/>PromptTemplate]
        I --> J[Final Prompt:<br/>문맥 + 질문 + 이미지]
        B --> J
        J --> K[Ovis VLM Final Generation<br/>max_new_tokens=512]
        K --> L[Final Answer]
    end
    
    %% Output
    L --> M[Output:<br/>- Image Description<br/>- Retrieved Context<br/>- Final Answer]
    
    %% Styling
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef vlmStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef retrievalStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef llmStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef outputStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef initStyle fill:#f5f5f5,stroke:#424242,stroke-width:2px
    
    class A,B inputStyle
    class C,D,E vlmStyle
    class F,G,H retrievalStyle
    class I,J,K llmStyle
    class L,M outputStyle
    class I1,I2,I3 initStyle
```

## Architecture Overview

### 🔧 **Initialization Phase**
1. **Ovis VLM Model**: Loads AIDC-AI/Ovis2-8B for vision-language tasks
2. **Embedding Model**: Loads BAAI/bge-m3 SentenceTransformer for text encoding
3. **ChromaDB**: Connects to persistent vector database with Korean knowledge base

### 🖼️ **VLM Processing Stage**
1. **Image Loading**: PIL loads the input image
2. **Image Description**: Ovis VLM generates Korean description of the image
3. **First Generation**: Uses image-only prompt to create textual representation

### 🔍 **Vector Retrieval Stage**
1. **Text Encoding**: SentenceTransformer encodes the image description
2. **Vector Search**: ChromaDB performs similarity search (top-k=3)
3. **Context Retrieval**: Returns relevant documents from knowledge base

### 🤖 **LLM Answer Generation**
1. **Prompt Construction**: Combines context, question, and image using PromptTemplate
2. **Final Generation**: Ovis VLM generates final answer using multimodal input
3. **Answer Output**: Returns grounded response based on retrieved context and image

## Key Features

- **Multimodal RAG**: Combines vision and text understanding
- **Korean Language**: Optimized for Korean subway domain knowledge
- **Direct ChromaDB**: Uses ChromaDB client without LangChain wrapper
- **Two-Stage VLM**: Image description → Final answer generation
- **Grounded Responses**: Answers must reference retrieved context or admit uncertainty

## Data Flow Summary

```
Image + Question → VLM Description → Vector Search → Context + Image + Question → Final Answer
```
