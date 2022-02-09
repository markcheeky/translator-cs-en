def chunkify(iterable, size, drop_incomplete=False):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk and not drop_incomplete:
        yield chunk


def unchunkify(iterable):
    for chunk in iterable:
        for item in chunk:
            yield item
