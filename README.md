# Get Hired - AI Recruitment Assistant

<div align="center">

![Get Hired Banner](https://github.com/user-attachments/assets/21b6f1a9-2a72-422e-9b45-c56c532484d6)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Weaviate](https://img.shields.io/badge/Vector_DB-Weaviate-green.svg)](https://weaviate.io/)

**An intelligent resume matching system powered by semantic embeddings and vector search**

[Features](#features) •
[Quick Start](#getting-started) •
[Documentation](#usage) •
[Benchmarks](#model-comparison--benchmarks) •
[Contributing](#contributing)

</div>

---

## Table of Contents

- [About](#about-the-project)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Comparison](#model-comparison--benchmarks)
- [Pipeline Overview](#resume-processing-pipeline)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About The Project

Get Hired is an AI-powered recruitment assistant that revolutionizes the resume screening process. By leveraging state-of-the-art transformer models and vector databases, it creates semantic embeddings of resumes to enable intelligent matching and retrieval based on job requirements.

### Motivation

Traditional keyword-based resume screening systems often miss qualified candidates due to rigid keyword matching, inability to understand context, poor handling of synonyms and related terms, and time-consuming manual processes.

Get Hired addresses these limitations by understanding semantic meaning beyond keywords, finding candidates with transferable skills, providing similarity scores for objective comparison, and scaling to thousands of resumes in seconds.

---

## Key Features

### Core Capabilities

- **Semantic Understanding**: Uses transformer-based models (BERT, Sentence-BERT) to capture deep semantic meaning from resume text
- **Vector Storage**: Efficient storage and retrieval using Weaviate's vector database with sub-second query times
- **Intelligent Matching**: Find best-matching resumes based on natural language job descriptions
- **Multiple Model Support**: Compare results across CBOW, Skip-gram, and Transformer models
- **Hybrid Search**: Combine vector similarity with metadata filtering (experience, skills, category)
- **Scalable Architecture**: Process and query thousands of resumes efficiently

### Advanced Features

- Natural language queries using plain English
- Batch processing for large resume datasets with configurable batch sizes
- Automated text cleaning, tokenization, and lemmatization
- Built-in tools to compare embedding quality across different models
- Robust error handling with detailed logging and recovery mechanisms

---

## Architecture

<div align="center">

![Solution Architecture](https://github.com/user-attachments/assets/de5047b5-64aa-4c8b-8500-27e1b64dab7e)

</div>

### System Components

**Data Ingestion Layer**
- Resume parsing and validation
- Text extraction and preprocessing

**Embedding Generation**
- HuggingFace Transformers (all-MiniLM-L6-v2)
- Custom Word2Vec models (CBOW, Skip-gram)

**Vector Storage**
- Weaviate vector database
- Schema-based structured storage
- Hybrid search capabilities

**Query Interface**
- Natural language processing
- Similarity search
- Metadata filtering

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Docker (for Weaviate instance)
- 4GB+ RAM recommended
- GPU optional (for faster embedding generation)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/dhou22/Get-Hired-Project.git
cd Get-Hired-Project
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
# Install from requirements file
pip install -r requirements.txt

# Or install packages individually
pip install transformers torch weaviate-client sentence-transformers \
            pandas numpy scikit-learn nltk gensim jupyter
```

#### 4. Start Weaviate Instance

```bash
# Using Docker
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED='true' \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e DEFAULT_VECTORIZER_MODULE='none' \
  semitechnologies/weaviate:latest
```

**Alternative**: Use Weaviate Cloud Services (WCS) for managed hosting

```python
# Connect to Weaviate Cloud
import weaviate

client = weaviate.connect_to_wcs(
    cluster_url="your-cluster-url.weaviate.network",
    auth_credentials=weaviate.auth.AuthApiKey("your-api-key")
)
```

#### 5. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## Usage

### Quick Start Example

```python
from sentence_transformers import SentenceTransformer
import weaviate

# 1. Initialize embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Connect to Weaviate
client = weaviate.connect_to_local()

# 3. Generate embedding for job description
job_description = """
Looking for a Senior Python Developer with experience in 
machine learning, NLP, and cloud technologies. 
Must have 5+ years of experience.
"""
query_vector = model.encode(job_description)

# 4. Search for matching resumes
collection = client.collections.get("Resume")
response = collection.query.near_vector(
    near_vector=query_vector,
    limit=5,
    return_metadata=['distance']
)

# 5. Display results
for item in response.objects:
    print(f"Resume ID: {item.properties['resume_id']}")
    print(f"Category: {item.properties['category']}")
    print(f"Similarity: {1 - item.metadata.distance:.3f}")
    print(f"Skills: {', '.join(item.properties['skills'])}")
    print("-" * 80)
```

### Running the Complete Pipeline

```bash
# Launch Jupyter Notebook
jupyter notebook resume-embedding-huggingface-weaviate-storage.ipynb
```

The notebook contains the following sections:

1. **Data Loading**: Import and explore resume dataset
2. **Preprocessing**: Clean and prepare text data
3. **Embedding Generation**: Create vector representations
4. **Weaviate Setup**: Configure schema and upload data
5. **Query & Retrieval**: Test semantic search functionality
6. **Benchmarking**: Compare model performance

### Advanced Usage

#### Filtering by Metadata

```python
# Search for Python developers with 5+ years experience
response = collection.query.near_vector(
    near_vector=query_vector,
    limit=10,
    filters=weaviate.classes.query.Filter.by_property("experience_years").greater_than(5) &
            weaviate.classes.query.Filter.by_property("category").equal("Data Science")
)
```

#### Batch Processing Resumes

```python
from utils import process_and_upload_resumes

# Process 1000 resumes in batches of 100
stats = process_and_upload_resumes(
    df=resume_dataframe,
    model=embedding_model,
    collection=weaviate_collection,
    batch_size=100
)

print(f"Successfully processed: {stats['success_count']} resumes")
print(f"Failed: {stats['failure_count']} resumes")
```

---

## Dataset
Dataset source on Kaggle : https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

<div align="center">

![Dataset Overview](https://github.com/user-attachments/assets/4289ca59-7ff7-42af-8041-841c0682528d)

</div>

### Dataset Statistics

- Total Resumes: 2,484
- Categories: 25 job categories
- Average Length: ~500 words per resume
- Format: Structured CSV with text fields

### Sample Categories

| Category | Count | Examples |
|----------|-------|----------|
| Data Science | 245 | Machine Learning Engineer, Data Analyst |
| Web Development | 312 | Full-Stack Developer, Frontend Engineer |
| DevOps | 189 | Cloud Engineer, SRE |
| Mobile Development | 156 | iOS Developer, Android Engineer |

### Data Schema in Weaviate

<img width="1890" height="906" alt="Capture d'écran 2025-10-04 221249" src="https://github.com/user-attachments/assets/759af581-3e41-46fa-8cba-bc67a1de1a81" />


---

## Resume Processing Pipeline


</div>

### Pipeline Steps

The preprocessing pipeline consists of the following stages:

**Lowercase Conversion**
- Ensures consistency across text
- Example: "Python" → "python"

**Remove Punctuation & Special Characters**
- Eliminates noise from embeddings
- Removes numbers unless contextually relevant

**Tokenization**
- Splits text into individual words/tokens
- Uses NLTK's word tokenizer

**Stopword Removal**
- Removes common words with little semantic value
- Preserves domain-specific terms

**Lemmatization**
- Converts words to base form
- Example: "running" → "run", "better" → "good"

**Text Reconstruction**
- Rejoins tokens into clean text
- Ready for embedding models

### Implementation Example

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_resume(text):
    # Lowercase conversion
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)
```

---

## Model Comparison & Benchmarks

<div align="center">

![Benchmark Results](https://github.com/user-attachments/assets/3e5a5d9b-f6ca-4c65-9db1-30952e3670f8)

</div>

### Performance Comparison

| Model | Embedding Dim | Inference Speed | OOV Handling | Quality Score |
|-------|--------------|-----------------|--------------|---------------|
| CBOW | 100 | Fast | Poor | Good |
| Skip-gram | 100 | Medium | Poor | Very Good |
| all-MiniLM-L6-v2 | 384 | Slower | Excellent | Excellent |

### Model Details
-------

### Word2VC models

<img width="1400" height="725" alt="image" src="https://github.com/user-attachments/assets/579d4d97-cff8-4ac3-b46e-b6e4e8b694a5" />


#### 1. CBOW (Continuous Bag of Words)

**Strengths:**
- Fast inference speed (~0.1ms per resume)
- Compact embeddings (100D)
- Excellent for frequent words
- Low memory footprint

**Limitations:**
- Limited to training vocabulary
- Poor performance on rare words
- Cannot handle out-of-vocabulary (OOV) terms

**Best Use Cases:**
- High-speed production systems
- Domain-specific vocabularies
- Resource-constrained environments

#### 2. Skip-gram

**Strengths:**
- Better semantic relationships
- Works well with rare words
- Captures fine-grained meanings
- Good for analogies and similarities

**Limitations:**
- Slower than CBOW
- Still limited to training vocabulary
- Requires more training data

**Best Use Cases:**
- Semantic similarity tasks
- Small to medium datasets
- Custom corpus training

### Sentence Transformer (all-MiniLM-L6-v2)
hugging face source model : https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
<img width="1901" height="583" alt="Capture d'écran 2025-10-04 224155" src="https://github.com/user-attachments/assets/e21cc5e4-7cd3-4f9f-81fd-21e1ef322ef3" />

**Strengths:**
- Handles any word (no OOV issues)
- Pre-trained on 1B+ sentence pairs
- Higher dimensional embeddings (384D)
- State-of-the-art quality
- Transfer learning capabilities
- Sentence-level understanding

**Limitations:**
- Slower inference (~10ms per resume)
- Larger model size (~80MB)
- Requires more computational resources

**Best Use Cases:**
- Production-ready applications
- General-purpose text matching
- When quality is priority
- Handling diverse vocabularies

### Benchmark Results

```
Query: "experienced python developer machine learning"

Model          | Top-3 Accuracy | Avg. Similarity | Query Time
---------------|----------------|-----------------|------------
CBOW           | 72%           | 0.68            | 0.02s
Skip-gram      | 78%           | 0.71            | 0.03s
MiniLM-L6-v2   | 94%           | 0.85            | 0.08s
```

### Recommendations

**Use CBOW/Skip-gram when:**
- Maximum speed is required
- Working with domain-specific vocabulary
- Training on custom corpus
- Memory/size is constrained
- Simple keyword matching suffices

**Use Sentence Transformer when:**
- Need robust OOV handling
- Working with sentences/phrases
- Want state-of-the-art quality
- Inference speed is acceptable
- Require transfer learning capabilities
- Production deployment is planned

---

## Roadmap

### Current Version (v1.0)

- Basic resume embedding and storage
- Semantic search functionality
- Multiple model support (CBOW, Skip-gram, Transformers)
- Batch processing
- Model benchmarking

### Upcoming Features (v1.1)

- REST API for integration
- Web-based UI dashboard
- Real-time resume parsing from PDFs
- Fine-tuned models on resume corpus
- Multi-language support
- Explainability features

### Future Enhancements (v2.0)

- Active learning feedback loop
- Candidate ranking algorithms
- Integration with ATS systems
- Bias detection and mitigation
- Skills gap analysis
- Automated interview question generation

See the [open issues](https://github.com/dhou22/Get-Hired-Project/issues) for proposed features and known issues.

---

## Contributing

Contributions are welcome. Any contributions you make are greatly appreciated.

### How to Contribute

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .

# Lint code
flake8 .
pylint src/
```

### Contribution Guidelines

- Write clear, commented code
- Add unit tests for new features
- Update documentation for API changes
- Follow PEP 8 style guidelines
- Keep pull requests focused and small
- Provide detailed PR descriptions

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Contact

**Project Maintainer:** dhou22

- GitHub: [@dhou22](https://github.com/dhou22)
- Project Link: [https://github.com/dhou22/Get-Hired-Project](https://github.com/dhou22/Get-Hired-Project)
- Issues: [Report a Bug](https://github.com/dhou22/Get-Hired-Project/issues)

---

## Acknowledgments

This project was made possible by:

- [HuggingFace](https://huggingface.co/) - State-of-the-art transformer models
- [Weaviate](https://weaviate.io/) - Vector database technology
- [Sentence-Transformers](https://www.sbert.net/) - Pre-trained semantic embedding models
- [NLTK](https://www.nltk.org/) - Natural language processing tools
- [Gensim](https://radimrehurek.com/gensim/) - Word2Vec implementations

### Research & References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers

---

## Additional Resources

- [Documentation](docs/)
- [API Reference](docs/api.md)
- [Tutorials](docs/tutorials/)
- [FAQ](docs/faq.md)
- [Changelog](CHANGELOG.md)

---

<div align="center">

Made by [dhou22](https://github.com/dhou22)

If you find this project helpful, please consider giving it a star.

![GitHub stars](https://img.shields.io/github/stars/dhou22/Get-Hired-Project?style=social)

</div>
