from __future__ import annotations

import os
import time

from typing import Literal, TypedDict, Optional
from dotenv import load_dotenv
from decimal import Decimal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field, field_validator

from tools.name_to_ticker import get_ticker_from_name


load_dotenv()


DEBUG = os.getenv("DEBUG", "true").lower() in {"1", "true", "yes", "on"}


def debug(message: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {message}")


class OrderDraft(BaseModel):
    """Fields extracted from natural-language input."""

    side: str | None = Field(
        default=None,
        description=(
            "Transaction direction. MUST be 'buy' if the user wants to "
            "purchase or buy shares. MUST be 'sell' if the user wants "
            "to sell shares. This field is required when either action "
            "is explicitly stated."
        ),
    )

    symbol: str | None = Field(
        default=None,
        description="Ticker symbol, such as AAPL.",
    )

    company_name: str | None = Field(
        default=None,
        description="Company name when the user names a company instead of its ticker.",
    )

    quantity: Decimal  | None = Field(
        default=None,
        description="Number of shares. May be fractional, e.g. 12.6.",
    )

    order_type: Literal["market", "limit"] | None = None
    
    limit_price: Optional[float] = Field(
        default=None, 
        description="The specified price boundary. Leave null for market orders."
    )

    @field_validator("symbol", "company_name", mode="before")
    @classmethod
    def normalize_missing_text(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value


class AgentState(TypedDict, total=False):
    command: str
    confirmed: bool
    draft: dict[str, object]
    missing: list[str]
    response: str
    error: str


def build_graph():
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    debug(f"Building graph with model={model_name!r}, base_url={base_url!r}")
    model = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0,
        format="json",
        keep_alive="1h",
    )
    extractor = model.with_structured_output(OrderDraft, method="json_schema")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a stock order extraction system.

                    Extract the following:
                    - side
                    - symbol
                    - company_name
                    - quantity
                    - order_type
                    - limit_price

                    SIDE RULES:
                    - buy, buying, purchase, purchasing = buy
                    - sell, selling = sell

                    TICKER RULES:
                    Convert company names into their publicly traded ticker symbols.

                    Examples:
                    - Apple = AAPL
                    - Microsoft = MSFT
                    - NVIDIA = NVDA
                    - Amazon = AMZN
                    - Tesla = TSLA
                    - Google = GOOGL
                    - Meta = META

                    IMPORTANT:
                    - If the company is clearly identifiable, return its ticker.
                    - Do not return null just because the user provided a company name
                    instead of a ticker.
                    - If the company is ambiguous, return null.
                    - If the user names a company, return its name in company_name.

                    ORDER RULES:
                    - Extract the number of shares.
                    - If the user does not state a share quantity, return
                      quantity as null. 
                      Examples:
                      - "buy Apple at 50"
                      means quantity null and limit_price 50.
                    - If a price is specified, the order must be limit.
                    - If no price is specified, use market.
                    - Never invent missing information.
                    - User can sepcify the number of shares in words or numbers, and may include fractions.
                    Examples:
                    - "buy 10 shares" = 10
                    - "buy ten shares" = 10
                    - "buy 10.5 shares" = 10.5
                    - "buy ten and a half shares" = 10.5
                    - "buy 10 1/2 shares" = 10.5
                    - "buy 10 point two five shares" = 10.25
                    - "buy ten point twenty five shares" = 10.25
                    - "sell one fifty Mayville" = 150 shares
                    - "buy Apple at 50" = quantity null
                    
                    PRICE RULES:
                    - If the user specifies a price, extract it as limit_price.
                    - The user may specify a price in various ways
                    - The price may be specified in words or numbers, and may include currency symbols or words like "dollars" or "cents".
                    - Treat a number phrase following "at" or "for" as the
                      limit price, even when the sentence contains a typo or
                      informal wording such as "I went to sell".
                    - If the price is specified in words, convert it to a number.
                    Examples:
                    - "at $150" = 150.00
                    - "for 150 dollars" = 150.00
                    - "at 150.50" = 150.50
                    - "at one fifty" = 150.00
                    - "at one hundred fifty and fifty cents" = 150.50
                    - "at one hundred fifty point five" = 150.50
                    - "at one five o" = 150.00
                    - "sell fifteen AudioEye at five thirty one" = limit_price 531.00

                    
                    GENERAL RULES:
                    - If the user query is not a stock order, return null for all fields.
                    - If the user query is ambiguous, return null for all fields.
                    - If the user query is unclear, return null for all fields.
                    - If the user query is incomplete, return null for all fields.
                    - If the user query is not in English, return null for all fields.  """,
            ),
            ("human", "{command}"),
        ]
    )

    def extract(state: AgentState) -> AgentState:        
        debug(f"[extract] command={state['command']!r}")
        debug("[extract] invoking Ollama structured output")
        draft = (prompt | extractor).invoke({"command": state["command"]})
        previous = state.get("draft")
        if previous:
            existing_data = OrderDraft.model_validate(previous).model_dump()
            new_data = draft.model_dump()
            for field, value in new_data.items():
                if value is not None:
                    existing_data[field] = value
                    
            draft = OrderDraft.model_validate(existing_data)

        if draft.company_name:
            debug(f"[extract] resolving company={draft.company_name!r}")
            ticker = get_ticker_from_name(draft.company_name)
            if ticker:
                draft.symbol = ticker.upper()
                debug(f"[extract] resolved ticker={draft.symbol!r}")
            else:
                draft.symbol = None
                debug("[extract] company lookup returned no ticker")

        debug(f"[extract] draft={draft.model_dump()}")
        return {"draft": draft.model_dump(mode="json")}

    def validate(state: AgentState) -> AgentState:
        draft = OrderDraft.model_validate(state["draft"])
        missing: list[str] = []
        
        debug(f"[validate] input={draft.model_dump()}")
        if draft.order_type is None:
            draft.order_type = (
                "limit" if draft.limit_price is not None else "market"
            )

        if draft.side is None:
            missing.append("whether to buy or sell")

        if not draft.symbol:
            missing.append("the ticker symbol")

        if draft.quantity is None:
            missing.append("the number of shares")
        elif draft.quantity <= 0:
            return {
                "missing": [],
                "error": "Share quantity must be greater than zero.",
            }

        if draft.order_type == "limit":
            if draft.limit_price is None or draft.limit_price <= 0:
                missing.append("a positive limit price")

        debug(f"[validate] missing={missing!r}")

        return {
            "missing": missing,
            "error": "",
        }

    def clarify(state: AgentState) -> AgentState:
            if state.get("error"):
                response = state["error"]
            else:
                response = (
                    "Please provide "
                    + ", ".join(state["missing"])
                    + "."
                )

            debug(f"[clarify] response={response!r}")

            return {"response": response}

    def confirm(state: AgentState) -> AgentState:
        draft = OrderDraft.model_validate(state["draft"])
        debug(f"[confirm] confirmed={state.get('confirmed', False)!r}")
        price = (
            f" at ${draft.limit_price:.2f}"
            if draft.order_type == "limit"
            else ""
        )
        if not state.get("confirmed", False):
            debug("[confirm] waiting for user confirmation")
            return {
                "response": (
                    f"Please confirm: {draft.side} {draft.quantity} shares of "
                    f"{draft.symbol.upper()} as a {draft.order_type} order{price}."
                )
            }
        return {}

    def execute(state: AgentState) -> AgentState:
        draft = OrderDraft.model_validate(state["draft"])
        debug(f"[execute] paper order={draft.model_dump()}")
        # Replace this node with the Questrade order API after paper-trading tests.
        return {
            "response": (
                f"Paper order submitted: {draft.side} {draft.quantity} shares "
                f"of {draft.symbol.upper()}."
            ),
            "draft": {},
            "confirmed": False,
            "missing": [],
            "error": "",
        }
        
    def route_initial(state: AgentState) -> str:
        # If confirming an existing order, skip extraction.
        if state.get("confirmed") and state.get("draft"):
            return "confirm"
        return "extract"
        
    def route_after_confirmation(state: AgentState) -> str:
        route = "execute" if state.get("confirmed", False) else "await_confirmation"
        debug(f"[route_after_confirmation] route={route!r}")
        return route
    
    def route_after_validation(state: AgentState) -> str:
        route = "clarify" if state.get("missing") or state.get("error") else "confirm"        
        debug(f"[route_after_validation] route={route!r}")
        return route

    graph = StateGraph(AgentState)
    graph.add_node("extract", extract)
    graph.add_node("validate", validate)
    graph.add_node("clarify", clarify)
    graph.add_node("confirm", confirm)
    graph.add_node("execute", execute)
    
    graph.add_conditional_edges(
        START,
        route_initial,
        {
            "extract": "extract",
            "confirm": "confirm",
        }
    )
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {"clarify": "clarify", "confirm": "confirm"},
    )
    graph.add_edge("clarify", END)
    graph.add_conditional_edges(
        "confirm",
        route_after_confirmation,
        {"execute": "execute", "await_confirmation": END},
    )
    graph.add_edge("execute", END)
    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)


def process_command(
    graph,
    command: str = "",
    confirmed: bool = False,
) -> dict:

    config = {
        "configurable": {
            "thread_id": "stock-trading-session"
        }
    }

    result = graph.invoke(
        {
            "command": command,
            "confirmed": confirmed,
        },
        config=config,
    )

    debug(f"[process_command] final_state={result}")

    return result


if __name__ == "__main__":

    graph = build_graph()

    config = {
        "configurable": {
            "thread_id": "stock-trading-session"
        }
    }

    print("\nStock Trading Agent")
    print("Enter 'quit' or 'exit' to close the agent.\n")

    while True:
        try:
            command = input("\nOrder command: ").strip()

            if command.lower() in {"quit", "exit"}:
                print("Closing trading agent.")
                break

            if not command:
                continue

            start = time.time()

            result = graph.invoke(
                {
                    "command": command,
                    "confirmed": False,
                },
                config=config,
            )

            print("\n" + result["response"])

            # If all fields are available, request confirmation
            if result["response"].startswith("Please confirm:"):

                answer = input(
                    "\nConfirm order? (yes/no): "
                ).strip().lower()

                if answer in {"yes", "y"}:

                    result = graph.invoke(
                        {"confirmed": True},
                        config=config,
                    )

                    print("\n" + result["response"])

                elif answer in {"no", "n"}:

                    # Reset the stored state for the next order
                    graph.update_state(
                        config,
                        {
                            "draft": {},
                            "missing": [],
                            "confirmed": False,
                            "error": "",
                        },
                    )

                    print("\nOrder cancelled.")

                else:
                    print("\nInvalid confirmation.")

            debug(
                f"[main] processing time: "
                f"{time.time() - start:.2f} seconds"
            )

        except KeyboardInterrupt:
            print("\nClosing trading agent.")
            break

        except Exception as e:
            print(f"\nAn error occurred: {e}")