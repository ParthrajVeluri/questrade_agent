# questrade_agent
A trading agent for the Questrade platform which allows users to execute orders through voice commands in real-time.

## LangGraph prototype

Install dependencies, install Ollama, and download a local model:

```bash
pip install -r requirements.txt
ollama serve
ollama pull llama3.2
```

Company names are resolved to ticker symbols with `yfinance` when the
extractor does not already return a ticker.

Run the text-input prototype:

```bash
python agent.py
```

The default model is `llama3.2`. You can choose another Ollama model:

```bash
export OLLAMA_MODEL="llama3.1:8b"
export OLLAMA_BASE_URL="http://localhost:11434"
```

The graph extracts flexible natural-language commands, validates required
fields, and asks for confirmation. It currently submits only a paper order.
To submit a confirmed command from Python:

```python
from agent import process_command

prompt = process_command("Can you get me ten Apple shares?")
print(prompt)
print(process_command("Can you get me ten Apple shares?", confirmed=True))
```

Do not connect the `execute` node to live trading until risk limits, account
selection, duplicate-order protection, and paper-trading tests are in place.
