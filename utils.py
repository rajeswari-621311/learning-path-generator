import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from prompt import user_goal_prompt
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, Any, Callable

from composio import Composio
from composio_langchain import LangchainProvider

load_dotenv()

cfg = RunnableConfig(recursion_limit=100)

COMPOSIO_USER_ID = os.getenv("COMPOSIO_USER_ID", "default")
REQUIRED_TOOLKITS = ("youtube", "googledrive")
TOOLKIT_LABELS = {
    "youtube": "YouTube",
    "googledrive": "Google Drive",
}


def initialize_model(google_api_key: str, model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=google_api_key
    )


def get_composio_client() -> Composio:
    return Composio(provider=LangchainProvider())


def _toolkit_slug(account: Any) -> str:
    toolkit = getattr(account, "toolkit", None)
    if isinstance(toolkit, str):
        return toolkit.lower()
    slug = getattr(toolkit, "slug", None) or getattr(account, "toolkit_slug", None)
    return str(slug or "").lower()


def get_connected_toolkits(user_id: str = COMPOSIO_USER_ID) -> dict[str, str]:
    """Return {toolkit_slug: status} for accounts belonging to this user."""
    client = get_composio_client()
    accounts = client.connected_accounts.list(user_ids=[user_id])
    statuses: dict[str, str] = {}
    for item in accounts.items or []:
        slug = _toolkit_slug(item)
        status = str(getattr(item, "status", "") or "").upper()
        if not slug:
            continue
        # Prefer ACTIVE if multiple accounts exist for the same toolkit.
        if slug not in statuses or status == "ACTIVE":
            statuses[slug] = status
    return statuses


def missing_required_toolkits(user_id: str = COMPOSIO_USER_ID) -> list[str]:
    connected = get_connected_toolkits(user_id)
    return [slug for slug in REQUIRED_TOOLKITS if connected.get(slug) != "ACTIVE"]


def create_connect_link(toolkit: str, user_id: str = COMPOSIO_USER_ID) -> str:
    """Create a Composio Connect Link so the user can authorize a toolkit."""
    client = get_composio_client()
    session = client.create(user_id=user_id, toolkits=[toolkit])
    request = session.authorize(toolkit)
    redirect_url = getattr(request, "redirect_url", None)
    if not redirect_url:
        raise RuntimeError(f"Composio did not return a connect URL for {toolkit}.")
    return redirect_url


async def setup_agent_with_tools(
    google_api_key: str,
    model: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Any:
    """
    Set up the agent with Composio tools (YouTube, Drive).
    """
    try:
        if progress_callback:
            progress_callback("Setting up agent with Composio tools... ✅")

        composio_client = get_composio_client()

        missing = missing_required_toolkits()
        if missing:
            labels = ", ".join(TOOLKIT_LABELS.get(slug, slug) for slug in missing)
            raise RuntimeError(
                f"No connected Composio account for: {labels}. "
                "Use the Connect buttons in the sidebar, finish Google authorization, "
                "then click Generate Learning Path again."
            )

        if progress_callback:
            progress_callback("Getting available tools from Composio... ✅")

        tools = composio_client.tools.get(
            user_id=COMPOSIO_USER_ID,
            toolkits=list(REQUIRED_TOOLKITS),
        )

        if progress_callback:
            progress_callback("Creating AI agent... ✅")

        mcp_orch_model = initialize_model(google_api_key, model)
        agent = create_react_agent(mcp_orch_model, tools)

        if progress_callback:
            progress_callback("Setup complete! Starting to generate learning path... ✅")

        return agent
    except Exception as e:
        print(f"Error in setup_agent_with_tools: {str(e)}")
        raise


def run_agent_sync(
    google_api_key: str,
    user_goal: str = "",
    model: str = "gemini-3.5-flash-lite",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Synchronous wrapper for running the agent with a chosen Gemini model."""
    async def _run():
        try:
            agent = await setup_agent_with_tools(
                google_api_key=google_api_key,
                model=model,
                progress_callback=progress_callback
            )

            learning_path_prompt = "User Goal: " + user_goal + "\n" + user_goal_prompt

            if progress_callback:
                progress_callback("Generating your learning path...")

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=learning_path_prompt)]},
                config=cfg
            )

            if progress_callback:
                progress_callback("Learning path generation complete!")

            return result
        except Exception as e:
            print(f"Error in _run: {str(e)}")
            raise

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()
