def lines(file):
    for line in file:
        yield line
    yield '\n'


def blocks(file):
    lines = open(file, 'r').read()
    block = []
    for line in lines:
        if line.strip():
            block.append(line)
        elif block:
            yield ''.join(block).strip()
            block = []
