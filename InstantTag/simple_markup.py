import re
from util import blocks


f = open('test_output.html', 'w')
# print('<html><head><title>...</title><body>')
f.write('<html><head><title>...</title><body>')


title = True
for block in blocks('test_input.txt'):
    block = re.sub(r'\*(.+?)\*', r'<em>\l</em>', block)
    if title:
        # print('<h1>')
        f.write('<h1>')
        # print(block)
        f.write(block)
        # print('</h1>')
        f.write('</h1>')
        title = False
    else:
        # print('<p>')
        f.write('<p>')
        # print(block)
        f.write(block)
        # print('</p>')
        f.write('</p>')
# print('</body></html>')
f.write('</body></html>')
