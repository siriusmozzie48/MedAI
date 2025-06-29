# Medical Information Assistant

A conversational AI assistant that provides medical information by combining local QA database search with real-time web crawling capabilities.

## Features

- Medical query analysis and symptom detection
- Local vector database search using FAISS
- Real-time web search and crawling with Firecrawl
- Structured medical information extraction
- Conversational interface with chat history
- Safety disclaimers and medical advice warnings

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection for web crawling features

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd medical-assistant
```

### 2. Create Virtual Environment

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

#### Get API Keys

**Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

**Firecrawl API Key:**
1. Visit [Firecrawl](https://www.firecrawl.dev/)
2. Sign up for an account
3. Navigate to your dashboard
4. Copy your API key

#### Add your API keys to the `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
Firecrawl_API_KEY=your_firecrawl_api_key_here
```

### 5. Configure Database Path

Open `QA_VectorStore.py` and update the `FOLDER_PATH` variable:

```python
FOLDER_PATH = "/path/to/your/project/Databases"
```

Replace `/path/to/your/project/Databases` with the actual path to your Databases folder.

**Example paths:**
- macOS/Linux: `/Users/yourusername/medical-assistant/Databases`
- Windows: `C:\Users\yourusername\medical-assistant\Databases`

## Usage

### First Run

Run the application for the first time to generate the vector database:

```bash
python main.py
```

The first run will:
- Process all CSV files in the Databases folder
- Create embeddings for the QA data
- Generate a FAISS vector database
- Save the database for future use

This process may take several minutes depending on your dataset size.

### Subsequent Runs

After the initial setup, subsequent runs will be much faster as they load the pre-built vector database:

```bash
python main.py
```

### Using the Assistant

1. Start the application with `python main.py`
2. Describe your symptoms or medical concerns
3. The assistant will analyze your query and provide relevant information
4. Ask follow-up questions for more detailed information
5. Type `exit`, `quit`, or `bye` to close the application

## Project Structure

```
medical-assistant/
├── main.py                     # Main application entry point
├── QA_VectorStore.py          # Vector database management
├── Retr_Ans_QA_VectorStore.py # Database retrieval functions
├── Web_VectorStore.py         # Web crawling integration
├── firecrawlSearch.py         # Firecrawl search functionality
├── crawlSchema.py             # Web content extraction schema
├── condition.py               # Medical condition detection
├── Databases/                 # CSV files directory
│   └── *.csv                  # QA data files
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── README.md                  # This file
```

## CSV Data Format

The system expects CSV files in the `Databases` folder with the following columns:
- `Question`: Medical questions
- `Answer`: Corresponding answers
- `qtype`: Question type classification

## Troubleshooting

### Common Issues

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using the correct Python environment

**API Key Errors:**
- Check that your `.env` file is in the project root
- Verify API keys are correct and active
- Ensure no extra spaces or quotes around API keys

**Database Path Issues:**
- Verify the `FOLDER_PATH` in `QA_VectorStore.py` points to your Databases folder
- Use absolute paths if relative paths cause issues
- Ensure CSV files exist in the specified directory

**Vector Database Issues:**
- Delete the `QA_db` folder and run the application again to regenerate
- Check that CSV files contain the required columns

### Performance Tips

- Use an SSD for faster vector database operations
- Ensure adequate RAM (8GB+ recommended) for large datasets
- Consider using GPU acceleration for faster embeddings

## Safety Notice

This application provides general medical information and is not a substitute for professional medical advice. Always consult qualified healthcare providers for medical concerns.

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please create an issue in the GitHub repository.
