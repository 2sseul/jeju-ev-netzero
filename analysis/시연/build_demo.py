"""demo_data.json을 template.html에 넣어 index.html을 만든다."""
import os
A = os.path.dirname(os.path.abspath(__file__))
html = open(f'{A}/template.html', encoding='utf8').read().replace('__DATA__', open(f'{A}/demo_data.json', encoding='utf8').read())
open(f'{A}/index.html', 'w', encoding='utf8').write(html); print('index.html', len(html))
