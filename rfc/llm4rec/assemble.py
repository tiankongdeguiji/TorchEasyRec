# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assemble rfc/llm4rec/index.html from fragments/ and generate the TOC.

The RFC is maintained as per-section HTML fragments under ``fragments/``
(``_head.html`` + ``sec-NN.html`` in order + ``_tail.html``). Edit the
fragments, then run this script to regenerate ``index.html`` — never edit
``index.html`` directly.
"""

import re
import sys
from pathlib import Path

FRAG = Path(__file__).resolve().parent / "fragments"
OUT = Path(__file__).resolve().parent / "index.html"


def build_toc(body: str) -> str:
    """Build a nested TOC list from the h2/h3 headings in the document body.

    Args:
        body (str): concatenated section + tail HTML.

    Returns:
        str: a properly nested ``<ul>`` tree (each h3 sub-list lives inside
            its parent h2 ``<li>``).
    """
    items = []
    for m in re.finditer(r'<h([23]) id="([^"]+)">(.*?)</h\1>', body, re.S):
        level, hid, text = int(m.group(1)), m.group(2), m.group(3)
        text = re.sub(r"<[^>]+>", "", text).strip()
        items.append((level, hid, text))
    out, depth = ["<ul>"], 2
    for level, hid, text in items:
        if level > depth:
            # open a sub-list inside the still-open parent <li>
            out[-1] = out[-1][: -len("</li>")]
            out.append("<ul>")
            depth = level
        elif level < depth:
            out.append("</ul></li>")
            depth = level
        out.append(f'<li><a href="#{hid}">{text}</a></li>')
    if depth > 2:
        out.append("</ul></li>")
    out.append("</ul>")
    return "\n".join(out)


def main() -> None:
    """Concatenate head + sections + tail, inject the TOC, write index.html."""
    head = (FRAG / "_head.html").read_text()
    tail = (FRAG / "_tail.html").read_text()
    secs = sorted(FRAG.glob("sec-*.html"))
    if not secs:
        sys.exit("no section fragments found")
    print("assembling:", ", ".join(p.name for p in secs))
    body = "\n".join(p.read_text().strip() for p in secs)
    toc = build_toc(body + tail)
    html = head.replace("<!--TOC-->", toc).replace("<!--SECTIONS-->", body) + tail
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
