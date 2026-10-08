"""Lightweight GenAI assistant (CLI) with tool calling.

Usage:
    python genai/assistant.py "What is the on-time rate for shipments from Shanghai to Rotterdam?"
    python genai/assistant.py --interactive

LLM provider: OpenAI (gpt-4o-mini by default) -- see README "Tech Choices"
for the cost-per-query rationale. If OPENAI_API_KEY is not set, the
assistant falls back to a small deterministic intent router ("offline
demo mode") so the tool-calling architecture can still be exercised and
tested without network access or an API key. This fallback is clearly
logged and labelled; it is not meant to simulate a real LLM's language
understanding.

Guardrails implemented (see README "GenAI guardrails" section):
  1. Off-domain refusal: the system prompt restricts the assistant to
     supply-chain/shipment questions; a lightweight keyword gate also
     short-circuits obviously-unrelated questions before calling the LLM.
  2. Tool-call cap: at most MAX_TOOL_CALLS tool invocations per question,
     to bound cost and prevent runaway loops.
  3. Citation requirement: the system prompt instructs the model to name
     which tool produced each factual claim; the offline router always
     labels its source tool explicitly.

Every call is logged as one JSON line to genai/logs/assistant_log.jsonl:
  prompt sent, tools called (with arguments), tool responses, and the
  final model output. This is treated as a production observability
  requirement, not an afterthought.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import TOOL_DEFINITIONS, Tools  # noqa: E402

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "assistant_log.jsonl"

MODEL_NAME = os.environ.get("GENAI_MODEL", "gpt-4o-mini")
MAX_TOOL_CALLS = 5

SYSTEM_PROMPT = """You are a supply-chain operations assistant for a container shipping
company. You answer questions ONLY about shipments, routes, ports, delays,
and delay-risk predictions, using the tools provided. You must not answer
questions outside this domain (politics, general trivia, coding help,
etc.) -- politely refuse those and explain you are scoped to supply-chain
data.

Rules:
- Always use a tool to look up factual data; never invent numbers.
- When you state a factual claim (a number, a rate, a prediction), say
  which tool produced it, e.g. "(via get_route_stats)".
- If a tool returns an error (e.g. route/shipment not found), tell the
  user plainly rather than guessing.
- Be concise.
"""

OFF_DOMAIN_KEYWORDS = [
    "politic", "weather", "recipe", "joke", "poem", "stock price",
    "sports score", "movie", "celebrity",
]


def _log_call(record: dict[str, Any]) -> None:
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _looks_off_domain(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in OFF_DOMAIN_KEYWORDS)


class Assistant:
    def __init__(self, api_base_url: str = "http://localhost:8000") -> None:
        self.tools = Tools(api_base_url)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except Exception as exc:  # pragma: no cover
                logging.warning("Could not init OpenAI client (%s); using offline mode", exc)

    @property
    def online(self) -> bool:
        return self._client is not None

    def ask(self, question: str) -> str:
        if _looks_off_domain(question):
            answer = (
                "I'm scoped to supply-chain shipment/route/delay questions and "
                "can't help with that. Try asking about a route, a shipment id, "
                "or a delay-risk prediction."
            )
            _log_call(
                {
                    "mode": "guardrail_refusal",
                    "question": question,
                    "tools_called": [],
                    "final_output": answer,
                }
            )
            return answer

        if self.online:
            return self._ask_llm(question)
        return self._ask_offline(question)

    # ------------------------------------------------------------------
    # Real LLM path (OpenAI tool calling)
    # ------------------------------------------------------------------
    def _ask_llm(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tools_called: list[dict[str, Any]] = []

        for _ in range(MAX_TOOL_CALLS):
            response = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                final = msg.content or ""
                _log_call(
                    {
                        "mode": "llm",
                        "model": MODEL_NAME,
                        "question": question,
                        "messages_sent": messages,
                        "tools_called": tools_called,
                        "final_output": final,
                    }
                )
                return final

            messages.append(msg.model_dump())
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments or "{}")
                result = self.tools.call(tool_call.function.name, args)
                tools_called.append(
                    {"name": tool_call.function.name, "arguments": args, "result": result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

        final = "I hit the tool-call limit for this question. Please narrow your request."
        _log_call(
            {
                "mode": "llm_capped",
                "model": MODEL_NAME,
                "question": question,
                "tools_called": tools_called,
                "final_output": final,
            }
        )
        return final

    # ------------------------------------------------------------------
    # Offline fallback: deterministic intent routing, same tools.
    # Used only when no OPENAI_API_KEY is configured.
    # ------------------------------------------------------------------
    def _ask_offline(self, question: str) -> str:
        tools_called: list[dict[str, Any]] = []
        q = question.strip()

        shipment_match = re.search(r"SHP-\d+", q, re.IGNORECASE)
        port_pair_match = re.search(r"\b([A-Z]{5})\b.*?\b([A-Z]{5})\b", q)

        if shipment_match:
            shipment_id = shipment_match.group(0).upper()
            result = self.tools.call("predict_delay", {"shipment_id": shipment_id})
            tools_called.append({"name": "predict_delay", "arguments": {"shipment_id": shipment_id}, "result": result})
            if "error" in result:
                answer = f"Couldn't find shipment {shipment_id}: {result['error']}"
            else:
                answer = (
                    f"Delay risk for {shipment_id}: {result['risk_label']} "
                    f"(probability={result['delay_risk_probability']}, "
                    f"model={result['model_name']}) (via predict_delay)"
                )
        elif port_pair_match:
            origin, destination = port_pair_match.group(1), port_pair_match.group(2)
            result = self.tools.call("get_route_stats", {"origin": origin, "destination": destination})
            tools_called.append({"name": "get_route_stats", "arguments": {"origin": origin, "destination": destination}, "result": result})
            if "error" in result:
                answer = f"No data for route {origin} -> {destination}: {result['error']}"
            else:
                answer = (
                    f"Route {result['route_key']}: on-time rate="
                    f"{result.get('on_time_rate')}, avg delay hours="
                    f"{result.get('avg_delay_hours')}, shipments={result['shipment_count']} "
                    f"(via get_route_stats)"
                )
        else:
            result = self.tools.call("query_shipments", {"limit": 5})
            tools_called.append({"name": "query_shipments", "arguments": {"limit": 5}, "result": result})
            answer = (
                f"I'm in offline demo mode (no OPENAI_API_KEY set), so I can only "
                f"do simple keyword routing. Here are {len(result.get('items', []))} "
                f"sample shipments (via query_shipments). Try including a shipment id "
                f"(SHP-xxxxx) or 'from <ORIGIN> to <DEST>' with 5-letter port codes."
            )

        _log_call(
            {
                "mode": "offline_fallback",
                "question": question,
                "tools_called": tools_called,
                "final_output": answer,
            }
        )
        return answer


def main() -> int:
    parser = argparse.ArgumentParser(description="Supply-chain GenAI assistant (CLI).")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--api-base-url", default=os.environ.get("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    assistant = Assistant(api_base_url=args.api_base_url)
    mode = "OpenAI (" + MODEL_NAME + ")" if assistant.online else "offline demo mode (no OPENAI_API_KEY)"
    print(f"[assistant mode: {mode}]")

    if args.interactive:
        print("Type 'exit' to quit.")
        while True:
            try:
                q = input("> ")
            except EOFError:
                break
            if q.strip().lower() in {"exit", "quit"}:
                break
            print(assistant.ask(q))
        return 0

    if not args.question:
        parser.print_help()
        return 1

    print(assistant.ask(args.question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
