---
title: Atlas Trip Planner
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Multi-agent trip planner with live flights, hotels and weather
---

# Atlas — agentic trip planner

Six specialist agents research live flights, real hotel rates, weather and
attractions, then reconcile the result against your budget.

Ask for a trip in plain language:

> long weekend in Lisbon from Chicago in October, two of us, around $3500

Watch the agents work in parallel, then read the plan. Every price traces back
to a tool call — the model is never trusted with a number.

Source and design notes: https://github.com/SwayamDesai/agentic-trip-planner

**Note:** this runs on free-tier language models with a shared daily budget. If
the agents report a rate limit, the quota is spent for the day rather than
anything being broken.
