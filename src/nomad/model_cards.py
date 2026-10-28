from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from huggingface_hub import hf_hub_download
from pydantic import Field
from thefuzz import fuzz, process

from nomad.truststore import configure_network_clients

MODEL_CARD_TOOL_NAME = "get_model_card"
LOGGER = logging.getLogger(__name__)


class ModelCardLocator:
    """Locate README files for registered SciFM tools."""

    def __init__(self) -> None:
        self._sources: dict[str, str | Path] = {}

    def register(self, tool_name: str, source: str | Path) -> None:
        """Associate a tool name with its resolved local path or repo id."""
        self._sources[tool_name] = source

    def registered_tools(self) -> Iterable[str]:
        return self._sources.keys()

    def read_model_card(self, tool_name: str) -> str:
        """Return the text contents of the README for the given tool."""
        source = self._sources.get(tool_name)
        if source is None:
            raise KeyError(self._unknown_tool_message(tool_name))

        card_path = self._resolve_local_card(source)
        if card_path is not None:
            return card_path.read_text(encoding="utf-8")

        remote_card = self._download_model_card(source)
        if remote_card is not None:
            return remote_card.read_text(encoding="utf-8")

        raise FileNotFoundError(f"No Model Card for `{tool_name}`")

    def _resolve_local_card(self, source: str | Path) -> Path | None:
        directory = source if isinstance(source, Path) else Path(source).expanduser()
        if not directory.is_dir():
            return None

        readme = directory / "README.md"
        if readme.is_file():
            return readme
        return None

    def _download_model_card(self, repo_id: str | Path) -> Path | None:
        if isinstance(repo_id, Path):
            return None

        configure_network_clients()
        try:
            downloaded = hf_hub_download(repo_id=repo_id, filename="README.md")
        except Exception as exc:  # pragma: no cover - exercised via tests with mocking
            LOGGER.exception("Error downloading README.md for %s", repo_id)
            raise FileNotFoundError(
                f"Unable to download README.md for repository '{repo_id}'."
            ) from exc

        card_path = Path(downloaded)
        if not card_path.is_file():
            return None
        return card_path

    def _unknown_tool_message(self, tool_name: str) -> str:
        message = f"No model registered with the name '{tool_name}'."
        suggestions = self._suggest_tool_names(tool_name)
        if not suggestions:
            return message
        matches = ", ".join(f"'{candidate}'" for candidate in suggestions)
        return f"{message} Did you mean: {matches}?"

    def _suggest_tool_names(self, tool_name: str) -> list[str]:
        registered = list(self._sources)
        if not registered:
            return []

        normalized_to_name = {candidate.lower(): candidate for candidate in registered}
        matches = process.extract(
            tool_name.lower(),
            list(normalized_to_name),
            scorer=fuzz.WRatio,
            limit=3,
        )
        return [normalized_to_name[match] for match, score in matches if score >= 60]

    def unregister(self, tool_name: str) -> None:
        """Remove a tool from the locator."""
        self._sources.pop(tool_name, None)


def register_model_card_tool(server: FastMCP, locator: ModelCardLocator) -> None:
    """Register the built-in model card retrieval tool with the server."""

    @server.tool(
        name=MODEL_CARD_TOOL_NAME,
        description="Return the Markdown model card for a registered SciFM tool.",
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True,
        },
    )
    def get_model_card(
        tool_name: Annotated[
            str,
            Field(description="Name of the SciFM tool to fetch the Model Card for"),
        ],
    ) -> str:
        return locator.read_model_card(tool_name)
