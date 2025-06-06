# Research Q/A Bot

AI-powered research assistant for analyzing and searching scientific articles using LlamaIndex and OpenAI.

## Features

- **Intelligent Search**: Query your LlamaCloudIndex with natural language
- **Structured Analysis**: AI-powered extraction of key findings, facts, and insights
- **Source Citations**: Automatic citation generation with relevance scores
- **Research Gaps**: Identification of areas needing further investigation
- **Export Results**: Download analysis results as JSON
- **Search History**: Track recent queries and repeat searches

## Quick Start

### Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Create Streamlit Cloud account** at [share.streamlit.io](https://share.streamlit.io)

3. **Deploy the app**:
   - Click "New app" 
   - Connect your GitHub repository
   - Select this repository and branch `main`
   - Set main file path: `app.py`

4. **Configure Secrets**:
   - Go to app settings → Secrets
   - Add the following secrets:

```toml
OPENAI_API_KEY = "your-openai-api-key"
LLAMA_CLOUD_API_KEY = "your-llamacloud-api-key"
```

5. **Deploy** - Your app will be available at `https://your-app-name.streamlit.app`

### Local Development

```bash
# Clone repository
git clone <your-repo-url>
cd research_qa_bot

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"
export LLAMA_CLOUD_API_KEY="your-key"

# Run app
streamlit run app.py
```

## Configuration

Update your LlamaCloud index details in `config.py`:

```python
LLAMACLOUD_INDEX_NAME = "Your Index Name"
LLAMACLOUD_PROJECT_NAME = "Your Project" 
LLAMACLOUD_ORGANIZATION_ID = "your-org-id"
```

## Usage

1. Enter your research question in the text area
2. Select number of sources to analyze (3-10)
3. Click "Search & Analyze"
4. Review structured results with key findings and citations
5. Export results or explore suggested follow-up queries

## API Keys Required

- **OpenAI API Key**: For content analysis and structured outputs
- **LlamaCloud API Key**: For document search and retrieval

## Support

For issues or questions, check the system status in the sidebar or review the logs in Streamlit Cloud dashboard.
