from pathlib import Path

reports = Path(r"C:\Users\yangfei\Code\differentiable\docs\reports")
for f in sorted(reports.glob("*.md")):
    content = f.read_text(encoding="utf-8")
    if '../resource/' in content:
        content = content.replace('src="../resource/', 'src="../../resource/')
        f.write_text(content, encoding="utf-8")
        print(f"{f.name}: fixed")
    else:
        print(f"{f.name}: already correct")
