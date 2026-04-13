from pathlib import Path

for fname in ['assignment2_part3.py', 'assignment2_part4.py']:
    text = Path(fname).read_text(encoding='utf-8')
    if 'matplotlib.use' not in text:
        text = text.replace(
            'import matplotlib.pyplot as plt',
            'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt',
            1
        )
    text = text.replace('plt.show()', 'plt.close()')
    Path(fname).write_text(text, encoding='utf-8')
    print(f'Patched {fname}')
