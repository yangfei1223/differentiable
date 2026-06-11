import re
from pathlib import Path

files = [
    Path(r"C:\Users\yangfei\Code\differentiable\docs\reports\05_Piano_PBR_Multi.md"),
    Path(r"C:\Users\yangfei\Code\differentiable\README.md"),
]

def video_to_link(m):
    src = m.group(1)
    name = Path(src).stem
    return f"[▶ {name}]({src})"

for f in files:
    content = f.read_text(encoding="utf-8")
    # Replace <video src="..." .../> with link
    new_content = re.sub(r'<video src="(.*?)"[^>]*/>', video_to_link, content)
    if new_content != content:
        f.write_text(new_content, encoding="utf-8")
        print(f"{f.name}: fixed")
    else:
        print(f"{f.name}: no change")
