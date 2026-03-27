import json, base64, re
from pathlib import Path

nb_path = Path(r"c:\Users\Milos\PythonProjects\speech_recognation\new_res\eda-analysis.ipynb")
out_dir = nb_path.parent / "_nb_audit"
img_dir = out_dir / "images"
out_dir.mkdir(exist_ok=True)
img_dir.mkdir(exist_ok=True)

nb = json.loads(nb_path.read_text(encoding="utf-8"))
lines = []
img_count = 0

for i, cell in enumerate(nb.get("cells", []), start=1):
    if cell.get("cell_type") != "code":
        continue
    outputs = cell.get("outputs", [])
    lines.append(f"\n=== Cell {i} ===")
    src = "".join(cell.get("source", []))
    src_short = src.strip().splitlines()[:2]
    lines.append("source_head: " + " | ".join(src_short))
    lines.append(f"outputs: {len(outputs)}")

    for j, out in enumerate(outputs, start=1):
        otype = out.get("output_type", "")
        lines.append(f"-- output {j}: {otype}")

        if otype == "stream":
            txt = out.get("text", "")
            if isinstance(txt, list):
                txt = "".join(txt)
            txt = txt.strip()
            lines.append("stream:\n" + txt[:2500])
            continue

        data = out.get("data", {})
        if not isinstance(data, dict):
            continue
        mimes = sorted(data.keys())
        lines.append("mimes: " + ", ".join(mimes))

        if "image/png" in data:
            img_count += 1
            b64 = data["image/png"]
            if isinstance(b64, list):
                b64 = "".join(b64)
            img_path = img_dir / f"cell_{i:02d}_out_{j:02d}.png"
            img_path.write_bytes(base64.b64decode(b64))
            lines.append(f"saved_image: {img_path.name}")

        if "text/plain" in data:
            t = data["text/plain"]
            if isinstance(t, list):
                t = "".join(t)
            lines.append("text_plain:\n" + str(t).strip()[:2500])

        if "text/markdown" in data:
            t = data["text/markdown"]
            if isinstance(t, list):
                t = "".join(t)
            lines.append("markdown:\n" + str(t).strip()[:2500])

        if "text/html" in data:
            h = data["text/html"]
            if isinstance(h, list):
                h = "".join(h)
            h_one = re.sub(r"\s+", " ", str(h))
            lines.append("html_head:\n" + h_one[:1200])

report = out_dir / "audit_report.txt"
report.write_text("\n".join(lines), encoding="utf-8")
print(f"Audit written: {report}")
print(f"Extracted images: {img_count}")
