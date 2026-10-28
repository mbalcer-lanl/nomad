from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import blake2b
from os import environ
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from filelock import FileLock, Timeout
from oras.auth.utils import get_basic_auth
from oras.provider import Registry

from nomad.copycow import copy_cow
from nomad.truststore import TruststoreHTTPAdapter, configure_network_clients

CACHE_LOCK_TIMEOUT_SECONDS = 900
SUPPORTED_SCHEMES = {"file", "hf", "oras", "git+https", "git+ssh"}

LOGGER = logging.getLogger(__name__)

_GIT_CERTIFICATE_ENVIRONMENT_VARIABLES = frozenset(
    {
        "CURL_CA_BUNDLE",
        "GIT_SSL_CAINFO",
        "GIT_SSL_CAPATH",
        "GIT_SSL_NO_VERIFY",
        "GIT_PROXY_SSL_CAINFO",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
    }
)
_GIT_TLS_CONFIG_SUFFIXES = (
    ".proxysslcainfo",
    ".sslcainfo",
    ".sslcapath",
    ".sslverify",
)


def looks_like_digest(value: str) -> bool:
    """Return True when ``value`` looks like a bare or algorithm-prefixed digest."""
    if ":" in value:
        algorithm, digest = value.lower().split(":", 1)
        if not (
            algorithm.startswith("sha") and algorithm.removeprefix("sha").isdigit()
        ):
            return False
    else:
        digest = value.lower()

    return len(digest) in {40, 64, 96, 128} and all(
        char in "0123456789abcdef" for char in digest
    )


def get_cache_root() -> Path:
    """Return the cache root, honoring XDG conventions or defaulting to ~/.cache."""
    if cache := environ.get("NOMAD_CACHE"):
        cache = Path(cache)
    elif cache := environ.get("XDG_CACHE_HOME"):
        cache = Path(cache) / "nomad"
    else:
        cache = Path.home().joinpath(".cache", "nomad")
    return cache


def _auth_target_key(target_ref: str | None) -> str | None:
    if target_ref is None:
        return None
    target = target_ref.removeprefix("oras://").split("@", 1)[0]
    if "/" not in target:
        return target.split(":", 1)[0]
    registry, remainder = target.split("/", 1)
    repository = remainder.rsplit(":", 1)[0]
    return f"{registry}/{repository}".rstrip("/")


def _auth_host(auth_key: str) -> str:
    return auth_key.removeprefix("https://").removeprefix("http://").split("/", 1)[0]


def _matching_auth_items(auths: dict, target_key: str | None) -> list[tuple[str, dict]]:
    if target_key is None:
        return [(key, value) for key, value in auths.items() if isinstance(value, dict)]

    target_host = _auth_host(target_key)
    prefix_matches: list[tuple[str, dict]] = []
    host_matches: list[tuple[str, dict]] = []
    domain_matches: list[tuple[str, dict]] = []
    for key, value in auths.items():
        if not isinstance(value, dict):
            continue
        normalized_key = (
            key.removeprefix("https://").removeprefix("http://").rstrip("/")
        )
        if normalized_key == target_key or target_key.startswith(f"{normalized_key}/"):
            prefix_matches.append((key, value))
        elif "/" not in normalized_key and _auth_host(normalized_key) == target_host:
            host_matches.append((key, value))
        elif _auth_host(normalized_key) == target_host:
            domain_matches.append((key, value))

    matches = prefix_matches or host_matches or domain_matches
    if domain_matches and matches is domain_matches:
        LOGGER.warning(
            "Using repository-scoped auth entry for '%s' as a domain-level fallback "
            "for '%s'",
            sorted(domain_matches, key=lambda item: len(item[0]), reverse=True)[0][0],
            target_key,
        )
    return sorted(matches, key=lambda item: len(item[0]), reverse=True)


def _read_auth_file(auth_file: Path) -> dict:
    try:
        return json.loads(auth_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid auth file '{auth_file}': {exc}") from exc


def _auth_entry_basic_auth(entry: dict) -> str | None:
    encoded_auth = entry.get("auth")
    if encoded_auth:
        return str(encoded_auth)

    username = entry.get("username")
    password = entry.get("password")
    if username is not None and password is not None:
        return get_basic_auth(str(username), str(password))
    return None


def _matching_basic_auth(auth_file: Path, target_ref: str | None) -> str | None:
    payload = _read_auth_file(auth_file)
    auths = payload.get("auths")
    if not isinstance(auths, dict):
        raise ValueError(f"Auth file '{auth_file}' must contain an 'auths' object")

    target_key = _auth_target_key(target_ref)
    for key, entry in _matching_auth_items(auths, target_key):
        basic_auth = _auth_entry_basic_auth(entry)
        if basic_auth is not None:
            LOGGER.debug(
                "Using auth entry '%s' from %s for ORAS container '%s'",
                key,
                str(auth_file),
                target_ref,
            )
            return basic_auth
    return None


def _get_docker_config_path(target_ref: str | None = None) -> str | None:
    """Return the registry auth config path, honoring container tool env vars."""
    config_paths: list[Path] = []

    if nomad_auth_file := environ.get("NOMAD_AUTH_FILE"):
        auth_path = Path(nomad_auth_file).expanduser()
        if not auth_path.exists():
            raise FileNotFoundError(
                f"NOMAD_AUTH_FILE is set but does not exist: {auth_path}"
            )
        return str(auth_path)
    if docker_config := environ.get("DOCKER_CONFIG"):
        config_path = Path(docker_config).expanduser()
        if config_path.is_dir() or config_path.suffix == "":
            config_path = config_path / "config.json"
        config_paths.append(config_path)
    config_paths.append(Path.home() / ".docker" / "config.json")

    for config_path in config_paths:
        if config_path.exists():
            return str(config_path)
    return None


CACHE_ROOT = get_cache_root() / "hub-v1"


class OrasRegistry(Registry):
    """ORAS registry client that refreshes stale bearer tokens after scope changes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session.mount("https://", TruststoreHTTPAdapter())

    def load_auth_config(self, target, *, config_path: str | None = None):
        container = self.get_container(target)
        config_path = config_path or _get_docker_config_path(str(container))
        self.auth.load_configs(
            container, configs=[config_path] if config_path else None
        )
        if config_path is not None:
            basic_auth = _matching_basic_auth(Path(config_path), str(container))
            if basic_auth is not None:
                self.auth._basic_auth = basic_auth
        return container

    def push(
        self,
        target: str,
        *args,
        config_path: str | None = None,
        do_chunked: bool = True,
        **kwargs,
    ):
        self.load_auth_config(target, config_path=config_path)
        return super().push(
            target, *args, config_path=None, do_chunked=do_chunked, **kwargs
        )

    def pull(self, target: str, *args, config_path: str | None = None, **kwargs):
        self.load_auth_config(target, config_path=config_path)
        return super().pull(target, *args, config_path=None, **kwargs)

    def request(
        self,
        target: str,
        url: str,
        method: str = "GET",
        *,
        config_path: str | None = None,
        data=None,
        headers: dict | None = None,
        json=None,
        stream: bool = False,
    ):
        self.load_auth_config(target, config_path=config_path)
        return self.do_request(url, method, data, headers, json, stream)

    def do_request(
        self,
        url: str,
        method: str = "GET",
        data=None,
        headers: dict | None = None,
        json=None,
        stream: bool = False,
    ):
        had_token = bool(getattr(self.auth, "token", None))
        response = super().do_request(url, method, data, headers, json, stream)
        if (
            response.status_code in {401, 403}
            and had_token
            and response.headers.get("Www-Authenticate")
        ):
            self.auth.token = None
            response = super().do_request(url, method, data, headers, json, stream)
        return response


def run(
    cmd: list[str],
    cwd: Path | None = None,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    """Execute a subprocess, exiting the interpreter if the command fails."""
    kwargs = {
        "cwd": cwd,
        "text": True,
        "check": True,
    }
    child_env = dict(environ)
    if cmd and Path(cmd[0]).name == "git":
        child_env.update(env or {})
        # Git cannot consume a Python SSLContext. Let its native TLS backend
        # use the host trust store without inherited CA overrides.
        for name in _GIT_CERTIFICATE_ENVIRONMENT_VARIABLES:
            child_env.pop(name, None)
        for name in tuple(child_env):
            if name in {"GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"} or name.startswith(
                ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
            ):
                child_env.pop(name, None)
    else:
        child_env.update(env or {})
    if env is not None or (cmd and Path(cmd[0]).name == "git"):
        kwargs["env"] = child_env
    if capture:
        kwargs["capture_output"] = True
    else:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.PIPE

    try:
        result = subprocess.run(cmd, **kwargs)
    except FileNotFoundError:
        logging.error("%s not found on PATH", cmd[0])
        raise
    except subprocess.CalledProcessError as e:
        logging.error("%s\n%s", str(e), e.stderr)
        raise
    return result.stdout.strip() if capture else ""


@contextmanager
def _sanitized_git_environment(*, cwd: Path | None):
    result = subprocess.run(
        ["git", "config", "--show-scope", "--null", "--list", "--includes"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    fields = iter(result.stdout.split("\0"))
    entries = []
    for scope, entry in zip(fields, fields, strict=False):
        key, separator, value = entry.partition("\n")
        if (
            separator
            and scope not in {"local", "worktree"}
            and not key.lower().startswith(("include.", "includeif."))
            and not key.lower().endswith(_GIT_TLS_CONFIG_SUFFIXES)
        ):
            entries.append((key, value))

    with tempfile.TemporaryDirectory(prefix="nomad-git-config-") as directory:
        config = Path(directory) / "config"
        for key, value in entries:
            subprocess.run(
                ["git", "config", "--file", str(config), "--add", key, value],
                check=True,
            )
        yield {
            "GIT_CONFIG_SYSTEM": str(config),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "0",
        }


def run_git_network(
    args: list[str], *, url: str, cwd: Path | None = None, capture: bool = False
) -> str:
    may_use_https = url.lower().startswith("https://") or args[:2] == ["lfs", "pull"]
    if not may_use_https:
        return run(["git", *args], cwd=cwd, capture=capture)
    with _sanitized_git_environment(cwd=cwd) as env:
        return run(["git", *args], cwd=cwd, capture=capture, env=env)


def check_dependencies() -> None:
    """Ensure required git tooling is available, exiting with a helpful message."""
    if shutil.which("git") is None:
        sys.exit("git not found in PATH")

    if subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
        sys.exit("git-lfs not installed or not initialized")


@dataclass(frozen=True)
class RepoSpec:
    """Normalized model-weight source URI with download and cache behavior."""

    scheme: str
    location: str
    reference: str | None = None
    subpath: str | None = None

    @classmethod
    def can_parse(cls, spec: str, *, base_dir: Path | None = None) -> bool:
        """Return True when the input can be normalized into a RepoSpec."""
        try:
            cls.parse(spec, base_dir=base_dir)
        except ValueError:
            return False
        return True

    @classmethod
    def parse(cls, spec: str, *, base_dir: Path | None = None) -> RepoSpec:
        """Normalize a model source string into a RepoSpec.

        Missing schemes are treated as local files when the path exists and as
        Hugging Face model IDs otherwise.
        """
        spec = cls.normalize(spec, base_dir=base_dir)
        parts, subpath = cls._split_spec(spec)

        if not parts.scheme:
            raise ValueError(f"Unable to normalize model source URI '{spec}'")

        if parts.scheme not in SUPPORTED_SCHEMES:
            raise ValueError(
                f"Unsupported model source scheme '{parts.scheme}' in '{spec}'"
            )

        if parts.scheme == "file":
            file_path = unquote(parts.path)
            if not file_path and parts.netloc:
                file_path = parts.netloc
            return FileRepoSpec(
                scheme="file",
                location=str(cls._resolve_local_path(file_path, base_dir=base_dir)),
                subpath=subpath,
            )

        if parts.scheme == "hf":
            repo = f"{parts.netloc}{parts.path}".strip("/")
            repo, reference = cls._split_reference(repo)
            if not repo:
                raise ValueError(f"Hugging Face URI requires a repo ID: '{spec}'")
            return HuggingFaceRepoSpec(
                scheme="hf",
                location=repo,
                reference=reference,
                subpath=subpath,
            )

        if parts.scheme == "oras":
            artifact_ref = f"{parts.netloc}{parts.path}"
            artifact_ref, reference = cls._split_oras_digest(artifact_ref)
            if not parts.netloc or not parts.path.strip("/"):
                raise ValueError(f"ORAS URI requires an artifact ref: '{spec}'")
            return OrasRepoSpec(
                scheme="oras",
                location=artifact_ref,
                reference=reference,
                subpath=subpath,
            )

        repo_path, reference = cls._split_reference(parts.path)
        repo_location = urlunsplit(
            (parts.scheme.removeprefix("git+"), parts.netloc, repo_path, "", "")
        )
        if not parts.netloc or not repo_path.strip("/"):
            raise ValueError(f"Git URI requires a repository path: '{spec}'")
        return GitRepoSpec(
            scheme=parts.scheme,
            location=repo_location,
            reference=reference,
            subpath=subpath,
        )

    @classmethod
    def normalize(cls, spec: str, *, base_dir: Path | None = None) -> str:
        """Return a canonical model source URI for supported shorthand inputs."""
        if not spec:
            raise ValueError("Model source URI cannot be empty")

        parts, subpath = cls._split_spec(spec)
        normalized: str

        if not parts.scheme:
            if cls._looks_like_scp_url(parts.path):
                normalized = cls._normalize_scp_url(parts.path)
            else:
                # Check if the original spec is an absolute path (starts with / or ~)
                is_absolute_spec = parts.path.startswith("/") or parts.path.startswith("~")
                local_path = cls._resolve_local_path(parts.path, base_dir=base_dir)
                # If spec is absolute path, treat as local file even if it doesn't exist yet.
                # Relative paths that don't exist are treated as HF repo IDs.
                if is_absolute_spec or local_path.exists():
                    normalized = local_path.as_uri()
                else:
                    normalized = f"hf://{parts.path}"
        elif parts.scheme in {"http", "https"}:
            if not parts.netloc or not parts.path.strip("/"):
                raise ValueError(f"Git URI requires a repository path: '{spec}'")
            normalized = urlunsplit(("git+https", parts.netloc, parts.path, "", ""))
        elif parts.scheme == "ssh":
            if not parts.netloc or not parts.path.strip("/"):
                raise ValueError(f"Git URI requires a repository path: '{spec}'")
            normalized = urlunsplit(("git+ssh", parts.netloc, parts.path, "", ""))
        else:
            normalized = urlunsplit(
                (parts.scheme, parts.netloc, parts.path, parts.query, "")
            )

        if subpath:
            normalized = f"{normalized}#{cls._quote_uri_piece(subpath)}"
        return normalized

    @staticmethod
    def _resolve_local_path(path: str, *, base_dir: Path | None = None) -> Path:
        local_path = Path(path).expanduser()
        if not local_path.is_absolute() and base_dir is not None:
            local_path = base_dir / local_path
        return local_path.resolve()

    @staticmethod
    def _normalize_subpath(fragment: str) -> str | None:
        subpath = unquote(fragment).strip("/") if fragment else None
        return subpath or None

    @classmethod
    def _split_spec(cls, spec: str):
        parts = urlsplit(spec)
        return parts._replace(fragment=""), cls._normalize_subpath(parts.fragment)

    @staticmethod
    def _split_reference(value: str) -> tuple[str, str | None]:
        if not value or "@" not in value:
            return value, None
        before, after = value.split("@", 1)
        if not before:
            return value, None
        return before, after or None

    @staticmethod
    def _looks_like_scp_url(value: str) -> bool:
        if "://" in value:
            return False
        user_host, sep, remainder = value.partition(":")
        return bool(sep and "@" in user_host and remainder)

    @classmethod
    def _normalize_scp_url(cls, value: str) -> str:
        user_host, _, path_and_ref = value.partition(":")
        repo_path, reference = cls._split_reference(path_and_ref)
        if repo_path.startswith("~/"):
            normalized_path = "/~/" + repo_path[2:]
        elif repo_path.startswith("/"):
            normalized_path = repo_path
        elif repo_path:
            normalized_path = "/" + repo_path
        else:
            normalized_path = "/"
        return urlunsplit(
            (
                "git+ssh",
                user_host,
                cls._render_ref(normalized_path, reference),
                "",
                "",
            )
        )

    @staticmethod
    def _split_oras_digest(artifact_ref: str) -> tuple[str, str | None]:
        ref_without_digest, sep, digest = artifact_ref.rpartition("@")
        if not sep:
            return artifact_ref, None
        if "/" in ref_without_digest and looks_like_digest(digest):
            return ref_without_digest, digest
        raise ValueError(f"Invalid ORAS digest reference '{digest}'")

    @staticmethod
    def _oras_repository_ref(artifact_ref: str) -> str:
        ref_without_digest, digest = RepoSpec._split_oras_digest(artifact_ref)
        if digest is not None:
            return ref_without_digest

        last_slash = ref_without_digest.rfind("/")
        last_colon = ref_without_digest.rfind(":")
        if last_colon > last_slash:
            return ref_without_digest[:last_colon]
        return ref_without_digest

    @property
    def is_remote(self) -> bool:
        """Return True for sources that can be left remote in exported configs."""
        raise ValueError(f"Unsupported model source scheme '{self.scheme}'")

    def name(self) -> str:
        """Return a human-readable name suitable for exported model directories."""
        raise ValueError(f"Unsupported model source scheme '{self.scheme}'")

    @staticmethod
    def _quote_uri_piece(value: str) -> str:
        return quote(value, safe="/:@")

    @staticmethod
    def _render_ref(value: str, reference: str | None) -> str:
        if reference:
            return f"{value}@{reference}"
        return value

    def _with_uri_subpath(self, spec: str) -> str:
        if self.subpath:
            return f"{spec}#{self._quote_uri_piece(self.subpath)}"
        return spec

    def uri(self, *, base_dir: Path | None = None) -> str:
        """Render this source as a canonical URI without resolving remote refs."""
        raise ValueError(f"Unsupported model source scheme '{self.scheme}'")

    def resolved(self) -> RepoSpec:
        """Return this source with mutable remote refs resolved where applicable."""
        return self

    @classmethod
    def lock(cls, spec: str, *, base_dir: Path | None = None) -> str:
        """Return ``spec`` pinned to its current remote revision."""
        return cls.parse(spec, base_dir=base_dir).resolved().uri()

    def as_https(self) -> RepoSpec:
        """Return a copy that uses HTTPS transport when the scheme supports it."""
        raise ValueError(f"Unsupported model source scheme '{self.scheme}'")

    @classmethod
    def to_https_spec(cls, spec: str, *, base_dir: Path | None = None) -> str:
        """Return ``spec`` rewritten to HTTPS transport when possible."""
        return cls.parse(spec, base_dir=base_dir).as_https().uri()

    def pull(self) -> Path:
        """Return a local path containing this source, using the cache if needed."""
        raise ValueError(f"Unsupported model source scheme '{self.scheme}'")

    def push(self, source: Path) -> RepoSpec:
        """Push ``source`` to this destination and return the exported spec."""
        raise ValueError(f"Unsupported model destination scheme '{self.scheme}'")

    def _with_subpath(self, path: Path) -> Path:
        if self.subpath is not None:
            return path.joinpath(self.subpath)
        return path

    @staticmethod
    def _safe_cache_name(value: str, *, fallback: str) -> str:
        safe_name = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value
        ).strip("-")
        return safe_name or fallback

    @staticmethod
    def _hash(value: str) -> str:
        return blake2b(value.encode(), digest_size=8).hexdigest()

    def _cache_key(self, identity: str, *, fallback: str) -> str:
        safe_name = RepoSpec._safe_cache_name(self.name(), fallback=fallback)
        return f"{safe_name}--{RepoSpec._hash(identity)}"

    def repo_key(self) -> str:
        """Return a stable cache key based on the source identity."""
        return self._cache_key(self.location.rstrip("/"), fallback="repo")

    def cache_digest(self) -> str:
        """Return the resolved cache version for this source."""
        if self.reference:
            return self.reference
        raise ValueError(f"Cannot cache unversioned model source '{self.uri()}'")

    def cache_dir(self) -> Path:
        """Return the cache directory for the resolved source."""
        return CACHE_ROOT / self.scheme / self.repo_key() / self.cache_digest()


class FileRepoSpec(RepoSpec):
    """Local filesystem model-weight source."""

    @property
    def is_remote(self) -> bool:
        return False

    def name(self) -> str:
        if self.subpath:
            return Path(self.subpath).name
        return Path(self.location).name

    def as_https(self) -> RepoSpec:
        return self

    def uri(self, *, base_dir: Path | None = None) -> str:
        path = Path(self.location)
        if base_dir is not None:
            try:
                path = path.resolve().relative_to(Path(base_dir).expanduser().resolve())
            except ValueError:
                pass

        if path.is_absolute():
            spec = path.as_uri()
        else:
            spec = f"file:{self._quote_uri_piece(path.as_posix())}"
        return self._with_uri_subpath(spec)

    def pull(self) -> Path:
        return self._with_subpath(Path(self.location))

    def push(self, source: Path) -> RepoSpec:
        destination = self._with_subpath(Path(self.location))
        if not destination.exists():
            copy_cow(source, destination, symlinks=True)
        return self


class HuggingFaceRepoSpec(RepoSpec):
    """Hugging Face model repository source."""

    @property
    def is_remote(self) -> bool:
        return True

    def name(self) -> str:
        if self.subpath:
            return Path(self.subpath).name
        return self.location.rsplit("/", 1)[-1]

    def as_https(self) -> RepoSpec:
        return self

    def uri(self, *, base_dir: Path | None = None) -> str:
        spec = f"hf://{self._render_ref(self.location, self.reference)}"
        return self._with_uri_subpath(spec)

    def resolved(self) -> RepoSpec:
        return HuggingFaceRepoSpec(
            scheme="hf",
            location=self.location,
            reference=self.cache_digest(),
            subpath=self.subpath,
        )

    def cache_digest(self) -> str:
        """Return the resolved Hugging Face revision used as the cache version."""
        if self.reference and looks_like_digest(self.reference):
            return self.reference

        configure_network_clients()
        from huggingface_hub import HfApi

        model_info_kwargs = {"repo_id": self.location}
        if self.reference:
            model_info_kwargs["revision"] = self.reference
        revision = getattr(HfApi().model_info(**model_info_kwargs), "sha", None)
        if not revision:
            sys.exit(f"Could not resolve Hugging Face model '{self.location}'")
        return revision

    def pull(self) -> Path:
        """Download or reuse a Hugging Face snapshot and return its local path."""
        configure_network_clients()
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                repo_id=self.location,
                revision=self.cache_digest(),
            )
        )
        return self._with_subpath(path)


class GitRepoSpec(RepoSpec):
    """Git/Git LFS repository model-weight source."""

    @property
    def is_remote(self) -> bool:
        return True

    def name(self) -> str:
        if self.subpath:
            return Path(self.subpath).name
        parsed = urlsplit(self.location)
        repo_name = parsed.path.rstrip("/").rsplit("/", 1)[-1] or "repo"
        return repo_name.removesuffix(".git")

    def as_https(self) -> RepoSpec:
        if self.scheme != "git+ssh":
            return self

        parsed = urlsplit(self.location)
        host = parsed.hostname
        if host is None:
            raise ValueError(f"Cannot rewrite repo URL '{self.location}' to HTTPS")

        return GitRepoSpec(
            scheme="git+https",
            location=urlunsplit(("https", host, parsed.path, "", "")),
            reference=self.reference,
            subpath=self.subpath,
        )

    def uri(self, *, base_dir: Path | None = None) -> str:
        parsed = urlsplit(self.location)
        repo_path = self._render_ref(parsed.path, self.reference)
        spec = urlunsplit((self.scheme, parsed.netloc, repo_path, "", ""))
        return self._with_uri_subpath(spec)

    def resolved(self) -> RepoSpec:
        return GitRepoSpec(
            scheme=self.scheme,
            location=self.location,
            reference=self.cache_digest(),
            subpath=self.subpath,
        )

    def _cache_identity(self) -> str:
        parsed = urlsplit(self.location)
        host = (parsed.hostname or parsed.netloc).lower()
        return f"{host}{parsed.path.rstrip('/')}"

    def repo_key(self) -> str:
        """Return a stable cache key based on the repository or artifact URL."""
        return self._cache_key(self._cache_identity(), fallback="repo")

    def cache_digest(self) -> str:
        """Return the resolved commit used as this source's cache version."""
        if self.reference and looks_like_digest(self.reference):
            return self.reference

        ref = self.reference or "HEAD"
        with tempfile.TemporaryDirectory(prefix="nomad-git-ls-remote-") as directory:
            output = run_git_network(
                ["ls-remote", self.location, ref],
                url=self.location,
                cwd=Path(directory),
                capture=True,
            )
        if not output:
            sys.exit(f"Could not resolve ref '{ref}'")
        return output.split()[0]

    def _lfs_pull_cmd(self) -> list[str]:
        cmd = ["git", "lfs", "pull"]
        if self.subpath:
            cmd.extend(["--include", f"{self.subpath},{self.subpath}/**"])
        return cmd

    def pull(self) -> Path:
        """Clone or reuse a cached git repository and return its path."""
        resolved = self.resolved()
        commit = resolved.reference
        if commit is None:  # pragma: no cover - resolved always pins git refs
            sys.exit(f"Could not resolve ref '{self.reference or 'HEAD'}'")
        cache_dir = resolved.cache_dir()

        if cache_dir.exists():
            return self._with_subpath(cache_dir)

        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_dir.with_suffix(".lock")
        try:
            lock = FileLock(lock_path, timeout=CACHE_LOCK_TIMEOUT_SECONDS)
            with lock:
                if cache_dir.exists():
                    return self._with_subpath(cache_dir)

                cache_dir_tmp = cache_dir.with_name("tmp--" + cache_dir.name)
                if cache_dir_tmp.is_dir():
                    shutil.rmtree(cache_dir_tmp)

                try:
                    cache_dir_tmp.mkdir(parents=True, exist_ok=True)
                    run(["git", "init", "-q"], cwd=cache_dir_tmp)
                    run(
                        ["git", "remote", "add", "origin", self.location],
                        cwd=cache_dir_tmp,
                    )
                    run_git_network(
                        ["fetch", "--depth", "1", self.location, commit],
                        url=self.location,
                        cwd=cache_dir_tmp,
                    )
                    run(
                        ["git", "checkout", commit],
                        cwd=cache_dir_tmp,
                        env={"GIT_LFS_SKIP_SMUDGE": "1"},
                    )
                    try:
                        run_git_network(
                            self._lfs_pull_cmd()[1:],
                            url=self.location,
                            cwd=cache_dir_tmp,
                        )
                    except subprocess.CalledProcessError as e:
                        if self.scheme == "git+ssh":
                            LOGGER.error(
                                "Hint: support for authentication when pulling Git LFS from git+ssh repos is limited. Try using git+https."
                            )
                        raise RuntimeError(f"Git LFS failed for {self.uri()}") from e
                    cache_dir_tmp.rename(cache_dir)
                except BaseException:
                    shutil.rmtree(cache_dir_tmp, ignore_errors=True)
                    raise
        except Timeout as exc:
            raise TimeoutError(
                f"Timed out waiting for cache lock: {lock_path}"
            ) from exc

        return self._with_subpath(cache_dir)


class OrasRepoSpec(RepoSpec):
    """ORAS artifact model-weight source."""

    @property
    def is_remote(self) -> bool:
        return True

    def name(self) -> str:
        if self.subpath:
            return Path(self.subpath).name
        return self._oras_repository_ref(self.location).rsplit("/", 1)[-1]

    def as_https(self) -> RepoSpec:
        return self

    def uri(self, *, base_dir: Path | None = None) -> str:
        spec = f"oras://{self._render_ref(self.location, self.reference)}"
        return self._with_uri_subpath(spec)

    @staticmethod
    def _registry():
        return OrasRegistry()

    def resolved(self) -> RepoSpec:
        return OrasRepoSpec(
            scheme="oras",
            location=RepoSpec._oras_repository_ref(self.location),
            reference=self._digest(),
            subpath=self.subpath,
        )

    def repo_key(self) -> str:
        """Return a stable cache key based on the artifact repository URL."""
        identity = RepoSpec._oras_repository_ref(self.location).rstrip("/")
        return self._cache_key(identity, fallback="artifact")

    def _digest(self) -> str:
        """Return the resolved ORAS manifest digest."""
        if self.reference and looks_like_digest(self.reference):
            return self.reference

        import oras.defaults

        registry = self._registry()
        container = registry.get_container(self.location)
        headers = {
            "Accept": ", ".join(oras.defaults.default_manifest_accepted_media_types)
        }
        try:
            response = registry.request(
                self.location,
                f"{registry.prefix}://{container.manifest_url()}",
                "HEAD",
                headers=headers,
            )
            if response.status_code not in {200, 201, 202}:
                reason = getattr(response, "reason", "unexpected response")
                raise ValueError(f"{response.status_code} {reason}")
        except Exception as exc:
            sys.exit(f"Could not resolve ORAS artifact '{self.location}': {exc}")

        digest = response.headers.get("Docker-Content-Digest")
        if not digest:
            sys.exit(f"Could not resolve ORAS artifact '{self.location}'")
        return digest

    @staticmethod
    def _digest_cache_name(digest: str) -> str:
        return RepoSpec._safe_cache_name(digest, fallback="digest")

    def cache_digest(self) -> str:
        """Return the resolved digest used as this source's cache version."""
        return self._digest_cache_name(self._digest())

    def pull(self) -> Path:
        """Pull or reuse a cached ORAS artifact and return its path."""
        resolved = self.resolved()
        repository_ref = resolved.location
        digest = resolved.reference
        if digest is None:  # pragma: no cover - resolved always pins ORAS refs
            sys.exit(f"Could not resolve ORAS artifact '{self.location}'")
        cache_dir = resolved.cache_dir()

        if cache_dir.exists():
            return self._with_subpath(cache_dir)

        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_dir.with_suffix(".lock")
        try:
            lock = FileLock(lock_path, timeout=CACHE_LOCK_TIMEOUT_SECONDS)
            with lock:
                if cache_dir.exists():
                    return self._with_subpath(cache_dir)

                cache_dir_tmp = cache_dir.with_name("tmp--" + cache_dir.name)
                if cache_dir_tmp.is_dir():
                    shutil.rmtree(cache_dir_tmp)
                cache_dir_tmp.mkdir(parents=True, exist_ok=True)

                try:
                    resolved._registry().pull(
                        f"{repository_ref}@{digest}",
                        outdir=str(cache_dir_tmp),
                    )
                    cache_dir_tmp.rename(cache_dir)
                except BaseException:
                    shutil.rmtree(cache_dir_tmp, ignore_errors=True)
                    raise
        except Timeout as exc:
            raise TimeoutError(
                f"Timed out waiting for cache lock: {lock_path}"
            ) from exc

        return self._with_subpath(cache_dir)

    def push(self, source: Path) -> RepoSpec:
        """Push a local model directory and return a digest-pinned ORAS spec."""
        if not source.is_dir():
            raise ValueError(f"ORAS model export requires a directory: {source}")

        target = self._render_ref(self.location, self.reference)
        files = [str(path) for path in sorted(source.iterdir())]
        try:
            response = self._registry().push(
                target,
                disable_path_validation=True,
                files=files,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not push model '{source.name}' to ORAS artifact "
                f"'{target}': {exc}"
            ) from exc

        digest = response.headers.get("Docker-Content-Digest")
        if not digest:
            raise ValueError(f"Could not resolve pushed ORAS artifact '{target}'")

        return OrasRepoSpec(
            scheme="oras",
            location=RepoSpec._oras_repository_ref(target),
            reference=digest,
            subpath=None,
        )
