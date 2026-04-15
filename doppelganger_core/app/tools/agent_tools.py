"""OpenAI Agents SDK tool registration for the AI doppelganger."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.services import internal_documents
from app.tools import file_tools, gmail_client


def build_agent_tools(function_tool: Callable[..., Any]) -> list[Any]:
    """Build the current set of custom tools available to the doppelganger."""

    @function_tool(
        name_override="send_gmail",
        description_override=(
            "Send a plain-text Gmail message on behalf of the user. "
            "Use this only when the user explicitly wants to send an email."
        ),
    )
    def send_gmail(
        to: list[str],
        subject: str,
        body_text: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        thread_id: str | None = None,
        from_email: str | None = None,
    ) -> dict[str, Any]:
        """Send a Gmail message with variable To, Cc, and Bcc recipients."""
        email = gmail_client.OutboundEmail(
            to=to,
            cc=cc or [],
            bcc=bcc or [],
            subject=subject,
            body_text=body_text,
            thread_id=thread_id,
            from_email=from_email,
        )
        result = gmail_client.send_gmail_message(email)
        return {
            "status": "sent",
            "id": result.get("id"),
            "threadId": result.get("threadId"),
            "labelIds": result.get("labelIds"),
        }

    @function_tool(
        name_override="read_gmail",
        description_override=(
            "Read recent Gmail messages for the user. "
            "Use this when the user explicitly wants to inspect, summarize, or reference email."
        ),
    )
    def read_gmail(
        query: str | None = None,
        max_results: int = 5,
    ) -> dict[str, Any]:
        """Read recent Gmail messages with an optional Gmail search query."""
        messages = gmail_client.read_gmail_messages(
            query=query,
            max_results=max_results,
        )
        return {
            "status": "ok",
            "messages": messages,
            "count": len(messages),
        }

    @function_tool(
        name_override="search_internal_documents",
        description_override=(
            "Search the user's internal notes and documents by semantic similarity. "
            "Use this when you deliberately want to inspect private notes, documents, "
            "or prior written material beyond the automatic lightweight retrieval."
        ),
    )
    def search_internal_documents(
        query: str,
        max_results: int = 3,
    ) -> dict[str, Any]:
        """Search the internal documents pgvector store with an explicit query."""
        documents = internal_documents.search_internal_documents_for_query(
            query,
            limit=max_results,
        )
        return {
            "status": "ok",
            "documents": documents,
            "count": len(documents),
        }

    @function_tool(
        name_override="read_file",
        description_override=(
            "Read a project text file. Use this especially for inspecting mind files like "
            "mind/SOUL.md and mind/DIRECTIVES.md before revising them. "
            "Access is limited to safe text files within the doppelganger_core project."
        ),
    )
    def read_file(
        path: str,
        max_chars: int = file_tools.MAX_READ_CHARS,
    ) -> dict[str, Any]:
        """Read a project text file with bounded output."""
        return file_tools.read_file(path, max_chars=max_chars)

    @function_tool(
        name_override="get_file_info",
        description_override=(
            "Get metadata for a project text file, including line count and a content hash. "
            "Use this before making targeted edits so you can reason about file size and "
            "optionally verify the file has not changed."
        ),
    )
    def get_file_info(path: str) -> dict[str, Any]:
        """Return basic metadata for a project text file."""
        return file_tools.get_file_info(path)

    @function_tool(
        name_override="read_file_window",
        description_override=(
            "Read a specific inclusive line window from a project text file. "
            "Use this for large files when you want to inspect them in overlapping sections "
            "instead of reading the whole file at once."
        ),
    )
    def read_file_window(
        path: str,
        start_line: int,
        end_line: int,
    ) -> dict[str, Any]:
        """Read an inclusive line window from a project text file."""
        return file_tools.read_file_window(path, start_line, end_line)

    @function_tool(
        name_override="search_in_file",
        description_override=(
            "Search a project text file for matching text and return nearby snippets with line numbers. "
            "Use this to locate the exact region that should be revised before calling replace_in_file."
        ),
    )
    def search_in_file(
        path: str,
        query: str,
        case_sensitive: bool = False,
        context_lines: int = file_tools.DEFAULT_SEARCH_CONTEXT_LINES,
        max_matches: int = file_tools.DEFAULT_SEARCH_MAX_MATCHES,
    ) -> dict[str, Any]:
        """Search a project text file and return matching line snippets."""
        return file_tools.search_in_file(
            path,
            query,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            max_matches=max_matches,
        )

    @function_tool(
        name_override="write_file",
        description_override=(
            "Write or append a project text file. Use this especially for revising "
            "mind/SOUL.md and mind/DIRECTIVES.md when the user explicitly wants to "
            "adjust the doppelganger's behavior. "
            "Access is limited to safe text files within the doppelganger_core project."
        ),
    )
    def write_file(
        path: str,
        content: str,
        append: bool = False,
    ) -> dict[str, Any]:
        """Write or append text content to a project file."""
        return file_tools.write_file(path, content, append=append)

    @function_tool(
        name_override="replace_in_file",
        description_override=(
            "Replace one exact text block inside a project text file. "
            "Use this for targeted edits after you have identified the precise section to change. "
            "Optionally pass an expected file hash from get_file_info or read_file_window to avoid stale edits."
        ),
    )
    def replace_in_file(
        path: str,
        old_text: str,
        new_text: str,
        expected_hash: str | None = None,
    ) -> dict[str, Any]:
        """Replace one exact text block in a project text file."""
        return file_tools.replace_in_file(
            path,
            old_text,
            new_text,
            expected_hash=expected_hash,
        )

    return [
        send_gmail,
        read_gmail,
        search_internal_documents,
        read_file,
        get_file_info,
        read_file_window,
        search_in_file,
        write_file,
        replace_in_file,
    ]
