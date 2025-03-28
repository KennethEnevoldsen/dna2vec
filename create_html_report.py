import markdown

# File paths
md_input_path = "/home/yigit/codebase/dna2vec/Results/analysis/plots_for_report/dna2vec_final_visual_report.md"
html_output_path = "/home/yigit/codebase/dna2vec/Results/analysis/plots_for_report/dna2vec_report.html"

# Load markdown content
with open(md_input_path, "r", encoding="utf-8") as f:
    md_text = f.read()

# Convert to HTML
html_body = markdown.markdown(md_text, extensions=["tables"])

# Basic CSS styling for readability
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DNA2Vec Topk Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            background-color: #fafafa;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #1a237e;
        }}
        img {{
            display: block;
            margin: 20px auto;
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
            padding: 5px;
            background: #fff;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table, th, td {{
            border: 1px solid #aaa;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: center;
        }}
        th {{
            background-color: #e0e0e0;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

# Write to HTML file
with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_template)

html_output_path
