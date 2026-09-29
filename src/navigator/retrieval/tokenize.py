"""Identifier-level tokenization.

Standard text tokenizers destroy the most useful signal in source code. A query
mentioning "parse config" should reach `parseConfigFile` and `parse_config`,
and a whitespace tokenizer reaches neither. This module splits on camelCase and
snake_case boundaries and keeps both the parts and the original identifier, so
an exact symbol match still outranks a partial one.
"""

from __future__ import annotations

import re

# Identifiers, including dotted attribute paths, which are split further below.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# camelCase and PascalCase boundaries, including acronym runs such as HTTPServer.
_CAMEL = re.compile(r".+?(?:(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to",
    "in", "on", "for", "with", "and", "or", "not", "it", "this", "that",
    "i", "we", "you", "does", "do", "did", "how", "what", "where", "which",
    "when", "why", "can", "should", "would", "if", "then", "else", "from",
    "by", "as", "at", "but", "so", "than", "into", "about",
}

# Python keywords and boilerplate carry no discriminative weight in a code index.
CODE_STOPWORDS = {
    "def", "class", "return", "self", "import", "none", "true", "false",
    "pass", "raise", "try", "except", "finally", "with", "as", "lambda",
    "yield", "assert", "del", "global", "nonlocal", "await", "async",
}

ALL_STOPWORDS = STOPWORDS | CODE_STOPWORDS


def split_identifier(identifier: str) -> list[str]:
    """Break one identifier into its parts, lowercased.

    `parseConfigFile` -> [parseconfigfile, parse, config, file]
    `parse_config`    -> [parse_config, parse, config]

    The intact identifier is emitted first so an exact match scores on both the
    whole symbol and each of its parts, which is what makes exact hits win.
    """
    out: list[str] = []
    lowered = identifier.lower()
    if lowered:
        out.append(lowered)

    for piece in identifier.split("_"):
        if not piece:
            continue
        for match in _CAMEL.finditer(piece):
            part = match.group(0).lower()
            if part and part != lowered:
                out.append(part)

    return list(dict.fromkeys(out))


def tokenize(text: str, drop_stopwords: bool = True,
             min_length: int = 2) -> list[str]:
    """Tokenize a query or a chunk into identifier-level terms."""
    tokens: list[str] = []
    for match in _IDENT.finditer(text):
        for part in split_identifier(match.group(0)):
            if len(part) < min_length:
                continue
            if drop_stopwords and part in ALL_STOPWORDS:
                continue
            tokens.append(part)
    return tokens


def tokenize_query(text: str) -> list[str]:
    """Queries keep short tokens: `id`, `os` and `db` are real search terms."""
    return tokenize(text, drop_stopwords=True, min_length=2)
