import re
import json

def count_tokens(text: str) -> int:
    return len(text.split())

def split_text(text: str, max_tokens: int = 500, chunk_overlap: int = 50) -> list[str]:
    sentences = re.split(r'[.!?]', text)
    chunk = []
    current_chunk = ''
    current_tokens = 0
    overlap_text = ''
    overlap_tokens = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = count_tokens(sent)
        if current_tokens + sent_tokens >= max_tokens and current_chunk:
            chunk.append(current_chunk.strip())
            words = current_chunk.split()
            if len(words) > chunk_overlap: 
                overlap_text = ' '.join(words[-chunk_overlap:])
                overlap_tokens = chunk_overlap
            else:
                overlap_text = current_chunk
                overlap_tokens = len(words)
            current_chunk = overlap_text + ' ' + sent
            current_tokens = overlap_tokens + sent_tokens
        else:
            current_chunk += sent + '. '
            current_tokens += sent_tokens
    if current_chunk:
        chunk.append(current_chunk.strip())
    return chunk

def chunking_text(lines: list[str], max_tokens: int = 500, chunk_overlap: int = 50) -> list[dict]:
    chunks = []
    chuong, muc, dieu = None, None, None
    buffer = []
    collecting = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not chuong and not re.match(r'Chương\s+[IVXLC]+', line):
            buffer.append(line)
            continue
        elif buffer and not chuong:
            text = ' '.join(buffer).strip()
            if text:
                if count_tokens(text) > max_tokens:
                    split_chunks = split_text(text, max_tokens, chunk_overlap)
                    for text_chunk in split_chunks:
                        chunks.append({
                            'chuong': '',
                            'muc': '',
                            'dieu': '',
                            'noidung': text_chunk.strip()
                        })
                else:
                    chunks.append({
                        'chuong': '',
                        'muc': '',
                        'dieu': '',
                        'noidung': text.strip()
                    })
            buffer = []

        if re.match(r'Chương\s+[IVXLC]+', line):
            chuong = line.strip()
            muc = ''
            chunks.append({
                'chuong': chuong,
                'muc': '',
                'dieu': '',
                'noidung': chuong
            })
            continue

        if re.match(r'Mục\s+\d+', line):
            muc = line.strip()
            chunks.append({
                'chuong': chuong,
                'muc': muc,
                'dieu': '',
                'noidung': muc
            })
            continue

        if re.match(r'Điều\s+\d+\.', line):
            if dieu and buffer:
                text = ' '.join(buffer).strip(  )
                if count_tokens(text) > max_tokens:
                    split_chunks = split_text(text, max_tokens, chunk_overlap)
                    for text_chunk in split_chunks:
                        chunks.append({
                            'chuong': chuong,
                            'muc': muc,
                            'dieu': dieu,
                            'noidung': text_chunk.strip()
                        })
                else:
                    chunks.append({
                        'chuong': chuong,
                        'muc': muc,
                        'dieu': dieu,
                        'noidung': text.strip()
                    })
            buffer = []
            dieu = line.strip()
            collecting = True
            continue

        if collecting:
            buffer.append(line)

    if dieu and buffer:
        text = ' '.join(buffer).strip()
        if count_tokens(text) > max_tokens:
            split_chunks = split_text(text, max_tokens, chunk_overlap)
            for text_chunk in split_chunks:
                chunks.append({
                    'chuong': chuong,
                    'muc': muc,
                    'dieu': dieu,
                    'noidung': text_chunk.strip()
                })
        else:
            chunks.append({
                'chuong': chuong,
                'muc': muc,
                'dieu': dieu,
                'noidung': text.strip()
            })

    return chunks

if __name__ == '__main__':
    file_path = 'data/Legan_new.txt'
    output_path = 'data/Chunk.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunks = chunking_text(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to: {output_path}")