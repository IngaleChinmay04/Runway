# Run this to see the REAL error
import sys
print(f"Python Executable: {sys.executable}")

print("Attempting to import agno...")
import agno
print(f"Agno version: {agno.__version__}")

print("Attempting to import Gemini model...")
from agno.models.google import Gemini
print("Gemini imported successfully!")

print("Attempting to import Agent...")
from agno.agent import Agent
print("Agent imported successfully!")
import mcp
