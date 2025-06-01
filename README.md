If you are interested in trading but don't have much time to review indicators, this Stocklyzer can help you check your favorite stocks quickly. It leverages the power of technical indicators with Generative AI to produce factual analysis. Since the analysis relies on AI, the results may not be 100% accurate.

**Note:** This tool is for educational purposes only and is not recommended for making financial decisions.

[Stocklyzer App](https://stocklyzer.streamlit.app)

## 🚀 Features

- **Dual AI Support**: Choose between Groq (Llama models) and Google Gemini
- **Technical Analysis**: Comprehensive indicators and charts
- **Real-time Data**: Live market data from Yahoo Finance
- **Interactive UI**: Modern interface with clickable pills for selections

## ⚙️ Configuration

Create a `.streamlit/secrets.toml` file with your API keys and preferences:

```toml
# Required: At least one AI provider API key
GROQ_API_KEY = "your_groq_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"

# Optional: Customize AI provider names and models
GROQ_PROVIDER_NAME = "Groq"
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_PROVIDER_NAME = "Gemini"  
GEMINI_MODEL = "gemini-2.0-flash-exp"
```

### Available Models:

**Groq Models:**
- `llama-3.3-70b-versatile` (default)
- `llama-3.2-90b-text-preview`
- `mixtral-8x7b-32768`

**Gemini Models:**
- `gemini-2.0-flash-exp` (default)
- `gemini-1.5-pro`
- `gemini-1.5-flash`

---

## Buy Me a Coffee

If you find this app helpful, consider buying me a coffee! ☕️

[![Buy Me a Coffee](bmc-button.png)](https://buymeacoffee.com/nyeinchankoko)
