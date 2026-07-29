def chunk_text(text, chunk_size=700):

    paragraphs = text.split("\n")

    chunks = []

    current = ""

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        if len(current) + len(para) <= chunk_size:
            current += para + "\n"

        else:
            chunks.append(current.strip())
            current = para + "\n"

    if current:
        chunks.append(current.strip())

    return chunks