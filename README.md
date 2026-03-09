# AI Weekly Planning Assistant

Personal AI agent that generates a weekly plan based on calendar events, weather, and local activities.

## Features

- Fetches upcoming events from Google Calendar
- Computes free time slots
- Fetches weather forecast
- Scrapes weekly events from The Lausanne Guide
- Uses Claude to recommend activities
- Sends a weekly email digest
- Creates Trello tasks via Zapier MCP

## Architecture

Data sources:
- Google Calendar API
- Open-Meteo Weather API
- Lausanne Guide website

Agent orchestration:
- LangGraph workflow

Action layer:
- Email via Gmail API
- Trello tasks via Zapier MCP

## Stack

Python  
LangChain / LangGraph  
Claude Sonnet  
Zapier MCP  
Google APIs

## Example output

Weekly digest email + Trello activity board.
