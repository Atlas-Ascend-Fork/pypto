# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Span tracking for preserving source location information during parsing."""

from __future__ import annotations

__all__ = ["SpanTracker"]


import ast
from collections.abc import Sequence

from pypto.pypto_impl import ir


class SpanTracker:
    """Tracks source locations from AST nodes to IR spans."""

    def __init__(self, source_file: str, source_lines: Sequence[str], line_offset: int = 0, col_offset: int = 0):
        """Initialize span tracker.

        Args:
            source_file: Path to the source file
            source_lines: List of source code lines (dedented for parsing)
            line_offset: Line number offset to add to AST line numbers (for dedented code)
            col_offset: Column offset to add to AST column numbers (for dedented code)
        """
        self.source_file = source_file
        self.source_lines = source_lines
        self.line_offset = line_offset
        self.col_offset = col_offset

    def get_span(self, ast_node: ast.AST | None) -> ir.Span:
        """Extract span from AST node.

        Args:
            ast_node: AST node with line/column information

        Returns:
            IR span corresponding to the AST node location
        """
        if ast_node is None or not hasattr(ast_node, "lineno"):
            return ir.Span.unknown()

        return ir.Span(
            self.source_file,
            getattr(ast_node, "lineno", 0) + self.line_offset,
            # ir.Span columns are 1-based and inclusive; the AST gives a 0-based
            # start, so it moves over by one. ``end_col_offset`` is 0-based and
            # exclusive, which is already the 1-based inclusive end.
            getattr(ast_node, "col_offset", 0) + self.col_offset + 1,
            getattr(ast_node, "end_lineno", 0) + self.line_offset,
            getattr(ast_node, "end_col_offset", 0) + self.col_offset,
        )

    def keyword_span(self, keywords: dict, name: str, fallback: ir.Span) -> ir.Span:
        """Span of the value written for keyword *name* in a parsed call.

        An error that says which argument is wrong should point at that argument
        rather than at the whole call. *fallback* covers the case where the
        keyword was not written at all -- there is nothing to point at then, and
        the call itself is the location.

        Args:
            keywords: ``{name: ast.keyword}``, as the op parsers build it
            name: keyword whose value the error is about
            fallback: span to use when *name* was not written

        Returns:
            Span of the keyword's value, or *fallback*
        """
        written = keywords.get(name)
        return self.get_span(written.value) if written is not None else fallback

    def call_argument_span(self, call: "ast.Call", name: "str | int", fallback: ir.Span) -> ir.Span:
        """Span of the argument *name* as written in *call*, or *fallback*.

        *name* is a keyword name, or an int index for a positional argument. The
        fallback covers an argument that was not written at all -- there is
        nothing to point at then, and the call itself is the location.
        """
        if isinstance(name, int):
            return self.get_span(call.args[name]) if name < len(call.args) else fallback
        written = next((kw.value for kw in call.keywords if kw.arg == name), None)
        return self.get_span(written) if written is not None else fallback

    def call_keyword_name_span(self, call: "ast.Call", name: str, fallback: ir.Span) -> ir.Span:
        """Span of the keyword *name* itself in *call*, not of the value it was given.

        A keyword the callee never declared, or one that repeats a positional
        argument, is wrong in its name; the value beside it may be perfectly good.
        Pointing at the value would mark the one part of the argument that is not
        at fault.
        """
        written = next((kw for kw in call.keywords if kw.arg == name), None)
        if written is None or not hasattr(written, "col_offset"):
            return fallback
        start = written.col_offset + self.col_offset + 1
        return ir.Span(
            self.source_file,
            written.lineno + self.line_offset,
            start,
            written.lineno + self.line_offset,
            start + len(name) - 1,
        )

    def argument_spans(self, call: "ast.Call") -> dict:
        """Where each argument of *call* was written, keyed for ``argument_span``.

        Keywords are keyed by name and positional arguments by their index, which
        is what a builder knows about the value it is rejecting. Published by the
        op dispatcher so builders -- which see values, not syntax -- can still
        point at the argument rather than at the whole call.
        """
        spans: dict = {}
        for index, node in enumerate(call.args):
            spans[index] = self.get_span(node)
        for keyword in call.keywords:
            if keyword.arg is not None:
                spans[keyword.arg] = self.get_span(keyword.value)
        return spans

    def get_multiline_span(self, start_node: ast.AST, end_node: ast.AST) -> ir.Span:
        """Get span covering multiple lines.

        Args:
            start_node: AST node at the start
            end_node: AST node at the end

        Returns:
            IR span covering the range from start to end
        """
        if not hasattr(start_node, "lineno") or not hasattr(end_node, "lineno"):
            return ir.Span.unknown()

        return ir.Span(
            self.source_file,
            getattr(start_node, "lineno", 0) + self.line_offset,
            # ir.Span columns are 1-based and inclusive; the AST gives a 0-based
            # start, so it moves over by one. ``end_col_offset`` is 0-based and
            # exclusive, which is already the 1-based inclusive end.
            getattr(start_node, "col_offset", 0) + self.col_offset + 1,
            getattr(end_node, "end_lineno", 0) + self.line_offset,
            getattr(end_node, "end_col_offset", 0) + self.col_offset,
        )

    def get_source_text(self, ast_node: ast.AST | ir.Span | None) -> str:
        """Recover source text from an AST node or an existing IR span."""
        if ast_node is None:
            return ""
        if isinstance(ast_node, ir.Span):
            start = ast_node.begin_line - self.line_offset - 1
            end = ast_node.end_line - self.line_offset - 1
            column = ast_node.begin_column - self.col_offset - 1
            end_column = ast_node.end_column - self.col_offset
            if ast_node.filename != self.source_file or not (0 <= start <= end < len(self.source_lines)):
                return ""
            if column < 0 or end_column < 0:
                return ""
            lines = [line.encode("utf-8") for line in self.source_lines[start:end + 1]]
            lines[-1] = lines[-1][:end_column]
            lines[0] = lines[0][column:]
            return b"\n".join(lines).decode("utf-8")

        source = "\n".join(self.source_lines)
        try:
            text = ast.get_source_segment(source, ast_node)
        except Exception:
            text = None
        if text is not None:
            return text

        try:
            return ast.unparse(ast_node)
        except Exception:
            return ""
