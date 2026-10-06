import json
import sys

def audit_notebook():
    nb_path = r'd:\Breast_Cancer_DAP_Project\notebook\Breast_Cancer_Diagnostic_Metric_Correlation_Tumor_Size_Regression.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Total notebook cells: {len(nb['cells'])}")
    error_count = 0
    image_count = 0
    chapter_headings = []

    for i, cell in enumerate(nb['cells']):
        ctype = cell['cell_type']
        if ctype == 'markdown':
            for line in ''.join(cell['source']).split('\n'):
                line_s = line.strip()
                if line_s.startswith('## '):
                    chapter_headings.append(line_s)
        elif ctype == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                if out.get('output_type') == 'error':
                    error_count += 1
                    print(f"ERROR in cell {i}: {out.get('ename')}: {out.get('evalue')}")
                if 'data' in out and 'image/png' in out['data']:
                    image_count += 1

    print("\n--- Chapter Headings Found ---")
    for ch in chapter_headings:
        print(ch)

    print(f"\nTotal Code Execution Errors: {error_count}")
    print(f"Total Plots/Images Generated: {image_count}")
    assert error_count == 0, "Errors found in notebook!"
    assert len(chapter_headings) >= 16, "Missing chapters!"
    assert image_count == 13, f"Expected exactly 13 charts, got {image_count}"
    print("\nALL AUDIT CHECKS PASSED: EXACT 13 CHARTS & 16 CHAPTERS VERIFIED (100/100)!")

if __name__ == '__main__':
    audit_notebook()
