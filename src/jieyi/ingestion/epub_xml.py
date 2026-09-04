from __future__ import annotations

import re
from html.entities import html5
from xml.etree import ElementTree as ET
from xml.parsers import expat

_NAMED_ENTITY = re.compile(rb"&([A-Za-z][A-Za-z0-9]+);")
# Legacy EPUB 2 documents commonly declare one of these standard external
# doctypes. The identifiers are allowlisted exactly; the referenced resources
# are never fetched or parsed.
_STANDARD_EXTERNAL_DOCTYPES = {
    (
        "ncx",
        "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd",
        "-//NISO//DTD ncx 2005-1//EN",
    ),
    (
        "html",
        "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd",
        "-//W3C//DTD XHTML 1.1//EN",
    ),
    (
        "html",
        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd",
        "-//W3C//DTD XHTML 1.0 Strict//EN",
    ),
    (
        "html",
        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd",
        "-//W3C//DTD XHTML 1.0 Transitional//EN",
    ),
    (
        "html",
        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-frameset.dtd",
        "-//W3C//DTD XHTML 1.0 Frameset//EN",
    ),
    (
        "html",
        "http://www.w3.org/TR/xhtml-basic/xhtml-basic10.dtd",
        "-//W3C//DTD XHTML Basic 1.0//EN",
    ),
    (
        "html",
        "http://www.w3.org/TR/xhtml-basic/xhtml-basic11.dtd",
        "-//W3C//DTD XHTML Basic 1.1//EN",
    ),
}


def parse_xml_resource(data: bytes, label: str) -> ET.Element:
    """Accept standard EPUB doctypes without fetching or evaluating their DTDs."""
    def replace_html_entity(match: re.Match[bytes]) -> bytes:
        name = match.group(1).decode("ascii")
        if name in {"amp", "apos", "gt", "lt", "quot"}:
            return match.group(0)
        value = html5.get(f"{name};")
        if value is None:
            return match.group(0)
        # Numeric references preserve XML escaping and the document's byte encoding.
        return "".join(f"&#{ord(char)};" for char in value).encode("ascii")

    data = _NAMED_ENTITY.sub(replace_html_entity, data)
    validator = expat.ParserCreate()

    def reject_declaration(*_args: object) -> None:
        raise ValueError(f"DTD or entity declarations are not allowed in {label}")

    def check_doctype(
        name: str, system_id: str | None, public_id: str | None, has_internal_subset: int
    ) -> None:
        if has_internal_subset:
            reject_declaration()
        if (system_id is not None or public_id is not None) and (
            name,
            system_id,
            public_id,
        ) not in _STANDARD_EXTERNAL_DOCTYPES:
            reject_declaration()

    # Use XML events rather than a byte search: UTF-16 must not bypass this check,
    # and declaration-like text in comments or CDATA is just text. Reject subsets
    # at their opening, before any entity expansion or DTD processing can occur.
    validator.StartDoctypeDeclHandler = check_doctype
    validator.EntityDeclHandler = reject_declaration
    validator.ExternalEntityRefHandler = reject_declaration
    validator.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    try:
        validator.Parse(data, True)
        # ElementTree does not load external DTDs. Only validated input reaches it.
        return ET.fromstring(data)
    except (expat.ExpatError, ET.ParseError) as exc:
        raise ValueError(f"Invalid XML in {label}: {exc}") from exc
