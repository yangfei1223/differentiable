import re
from pathlib import Path

reports_dir = Path(r"C:\Users\yangfei\Code\differentiable\docs\reports")

for f in sorted(reports_dir.glob("*.md")):
    content = f.read_text(encoding="utf-8")
    
    # Replace video blocks: <p align="center"> ... </p> containing <video> tags
    # Match the whole video section
    def replace_video_block(m):
        lines = m.group(0).split('\n')
        links = []
        for line in lines:
            src_match = re.search(r'src="([^"]+)"', line)
            if src_match:
                # Get filename from path for display
                path = src_match.group(1)
                name = Path(path).stem
                links.append(f'[▶ {name}]({path})')
        
        if links:
            return '<p align="center">' + ' &nbsp; '.join(links) + '</p>'
        return m.group(0)
    
    pattern = r'<p align="center">\s*(?:\s*<video[^>]+/>\s*)+(?:\s*<video[^>]+/>\s*)+\s*</p>'
    new_content = re.sub(pattern, replace_video_block, content, flags=re.DOTALL)
    
    if new_content != content:
        f.write_text(new_content, encoding="utf-8")
        print(f"{f.name}: fixed")
    else:
        print(f"{f.name}: no change")
