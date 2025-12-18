#!/usr/bin/env python3
"""
Scrape RTS article: download audio and extract read-aloud text.

Usage:
  python scripts/scrape_rts.py <url>

Saves audio (if found) and a text file with title, date, source, lead and body paragraphs.
"""
import argparse
import os
import re
import sys
from urllib.parse import urlparse, urljoin, quote_plus, parse_qs
from datetime import datetime, date

import requests
from bs4 import BeautifulSoup, Comment


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def get_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def extract_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    # Title: take text before first '|'
    title = soup.find('title')
    title_text = ''
    if title:
        full = title.get_text(strip=True)
        title_text = full.split('|', 1)[0].strip()

    # Date: normalize to 'недеља, 14.12.2025, 12:15' (date + first time)
    story_date = ''
    sd = soup.select_one('.storyDate')
    if sd:
        raw = sd.get_text(' ', strip=True)
        # Try to find pattern: <weekday>, DD.MM.YYYY, HH:MM
        m = re.search(r'([^,]+,\s*\d{1,2}\.\d{1,2}\.\d{4}),\s*(\d{1,2}:\d{2})', raw)
        if m:
            story_date = f"{m.group(1)}, {m.group(2)}"
        else:
            # fallback: collapse whitespace
            story_date = ' '.join(raw.split())

    # Source
    story_source = ''
    ss = soup.select_one('.storySource')
    if ss:
        story_source = ss.get_text(' ', strip=True)

    # Lead
    lead = ''
    ld = soup.select_one('.lead.storyMainLead')
    if ld:
        lead = ld.get_text(' ', strip=True)

    # Collect main body paragraphs from article container only
    container = soup.select_one('.storyInfo') or soup.select_one('article') or soup
    def is_header_or_meta(text: str) -> bool:
        if not text:
            return True
        # common meta lines
        if 'Извор' in text or text.startswith('Извор:'):
            return True
        # time ranges like 12:15 -> 12:43
        if '->' in text and re.search(r'\d{1,2}:\d{2}', text):
            return True
        # short or placeholder
        if len(text) < 4:
            return True
        return False

    body_paragraphs = []

    # Prefer document order inside the body parent: include <p>, section titles and their short bodies
    body_comment = None
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        if 'BODY' in c.upper():
            body_comment = c
            break

    if body_comment:
        # prefer collecting <p> tags that come after the BODY comment
        parent = body_comment.parent
        # try to find a higher container that may include nearby short-story-holder elements
        container_for_flow = parent
        ancestor = parent
        found = None
        while ancestor is not None:
            if ancestor.find('div', class_='short-story-holder') or ancestor.find('h4', class_='short-story-title') or ancestor.name == 'article':
                found = ancestor
                break
            ancestor = ancestor.parent
        if found is not None:
            container_for_flow = found

        # Collect paragraphs and short-story-title sections from within the chosen container in DOM order.
        for elem in container_for_flow.descendants:
            if not getattr(elem, 'name', None):
                continue
            # short-story-holder or inline short-story-title
            if elem.name != 'p' and elem.find_all and elem.find('h4', class_='short-story-title'):
                title_el = elem.find('h4', class_='short-story-title')
                stitle = title_el.get_text(' ', strip=True) if title_el else ''
                if stitle:
                    body_paragraphs.append(stitle)
                body_div = elem.find('div', class_='short-story-body')
                if body_div:
                    for p in body_div.find_all('p'):
                        text = p.get_text(' ', strip=True)
                        if is_header_or_meta(text):
                            continue
                        body_paragraphs.append(text)
                else:
                    for p in elem.find_all('p'):
                        text = p.get_text(' ', strip=True)
                        if is_header_or_meta(text):
                            continue
                        body_paragraphs.append(text)
                continue

            if elem.name == 'p':
                p = elem
                # ensure paragraph belongs to container_for_flow (defensive)
                if not (container_for_flow in p.parents):
                    continue
                text = p.get_text(' ', strip=True)
                if ld and p is ld:
                    continue
                if is_header_or_meta(text):
                    continue
                if title_text and text == title_text:
                    continue
                if story_date and text.startswith(story_date.split(',')[0]):
                    continue
                body_paragraphs.append(text)
        

    else:
        for p in container.find_all('p'):
            text = p.get_text(' ', strip=True)
            # skip lead (keep separately)
            if ld and p is ld:
                continue
            if is_header_or_meta(text):
                continue
            # skip repeated title or source lines
            if title_text and text == title_text:
                continue
            if story_date and text.startswith(story_date.split(',')[0]):
                continue
            body_paragraphs.append(text)

    return {
        'title': title_text,
        'date': story_date,
        'source': story_source,
        'lead': lead,
        'paragraphs': body_paragraphs,
    }


def find_audio_url(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')

    # 1) <audio src=...> or <audio><source src=...>
    for audio in soup.find_all('audio'):
        src = audio.get('src')
        if src:
            return src
        src_tag = audio.find('source')
        if src_tag and src_tag.get('src'):
            return src_tag.get('src')

    # 2) look for tags with data-audio or data-src attributes
    for tag in soup.find_all(attrs=True):
        for attr in ('data-audio', 'data-src', 'data-file', 'data-media'):
            if tag.has_attr(attr):
                return tag.get(attr)

    # 3) JSON-like patterns inside scripts
    m = re.search(r'"(?:audio|file|mp3|url)"\s*[:=]\s*"(https?://[^"\']+\.(?:mp3|m4a|ogg|wav|aac))"', html, re.I)
    if m:
        return m.group(1)

    # 4) generic regex for typical audio file extensions
    m2 = re.search(r'https?://[^\s"\']+?\.(?:mp3|m4a|ogg|wav|aac)\b', html, re.I)
    if m2:
        return m2.group(0)

    return ''


def download_file(url: str, out_path: str, referer: str = None) -> None:
    headers = HEADERS.copy()
    if referer:
        headers['Referer'] = referer
    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def safe_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path)
    if not name:
        name = 'rts_audio'
    return name


def save_text(info: dict, out_path: str) -> None:
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        if info.get('title'):
            f.write(info['title'].strip() + '\n')
        if info.get('date'):
            f.write(info['date'].strip() + '\n')
        # Do not write source (Извор) per user request
        f.write('\n')
        if info.get('lead'):
            f.write(info['lead'].strip() + '\n\n')
        for p in info.get('paragraphs', []):
            f.write(p.strip() + '\n\n')


def article_basename_from_url(url: str) -> str:
    p = urlparse(url)
    name = os.path.basename(p.path)
    if not name:
        name = 'article'
    # remove query and extension
    name = name.split('?')[0]
    name = name.replace('.html', '')
    return re.sub(r'[^a-zA-Z0-9_\-\.]+', '_', name)


def crawl_search(term: str, out_dir: str):
    """Crawl RTS search results for `term`, follow pagination, download each article."""
    os.makedirs(out_dir, exist_ok=True)
    visited = set()
    session = requests.Session()
    session.headers.update(HEADERS)

    page_idx = 0
    while True:
        if page_idx == 0:
            search_url = f'https://www.rts.rs/pretraga.html?lang=sr&searchText={quote_plus(term)}'
        else:
            search_url = f'https://www.rts.rs/pretraga.html?lang=sr&searchText={quote_plus(term)}&position={page_idx}'

        print(f'Fetching search page: {search_url}')
        try:
            resp = session.get(search_url, timeout=20)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            print('Failed to fetch search page:', e)
            break

        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for a in soup.select('h2.title a'):
            href = a.get('href')
            if not href:
                continue
            full = urljoin('https://www.rts.rs/', href)
            links.append(full)

        new_found = 0
        for link in links:
            if link in visited:
                continue
            visited.add(link)
            new_found += 1
            print('Processing article:', link)
            try:
                ah = get_html(link)
            except Exception as e:
                print('  Failed to fetch article:', e)
                continue
            # parse article and skip short-story-title articles
            a_soup = BeautifulSoup(ah, 'html.parser')
            if a_soup.find('h4', class_='short-story-title'):
                print('  Skipping article (contains short-story-title)')
                continue

            info = extract_text(ah)
            # skip articles older than 2024-01-01 if date is available
            date_str = info.get('date', '') or ''
            mdate = re.search(r"(\d{1,2}\.\d{1,2}\.\d{4})", date_str)
            if mdate:
                try:
                    art_date = datetime.strptime(mdate.group(1), '%d.%m.%Y').date()
                    if art_date < date(2024, 1, 1):
                        print('  Skipping article (older than 2024-01-01)', art_date)
                        continue
                except Exception:
                    pass
            audio_url = find_audio_url(ah)

            # skip if article lacks audio or lacks extracted text
            has_text = bool(info.get('lead') or (info.get('paragraphs') and len(info.get('paragraphs'))>0) or info.get('title'))
            if not audio_url or not has_text:
                print('  Skipping article (missing audio or text)')
                continue

            base = article_basename_from_url(link)
            text_out = os.path.join(out_dir, base + '.txt')
            save_text(info, text_out)
            print('  Saved text ->', text_out)

            if audio_url:
                if audio_url.startswith('//'):
                    audio_url = 'https:' + audio_url
                if audio_url.startswith('/'):
                    audio_url = urljoin(link, audio_url)
                fname = safe_filename_from_url(audio_url)
                audio_out = os.path.join(out_dir, base + os.path.splitext(fname)[1])
                try:
                    download_file(audio_url, audio_out, referer=link)
                    print('  Saved audio ->', audio_out)
                except Exception as e:
                    print('  Failed to download audio:', e)

        # find next page link
        next_a = soup.select_one('nav[aria-label="Pagination"] .next a') or soup.select_one('li.next a')
        if next_a and next_a.get('href'):
            # extract position param; if missing, increment
            href = next_a.get('href')
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            if 'position' in qs and qs['position']:
                try:
                    page_idx = int(qs['position'][0])
                except Exception:
                    page_idx += 1
            else:
                page_idx += 1
            # if no new links on this page and we've looped, break
            if new_found == 0:
                print('No new articles found on page; stopping.')
                break
            # continue to next page
            continue
        else:
            print('No next page found; finished crawling.')
            break


def main():
    parser = argparse.ArgumentParser(description='Download audio and text from an RTS article')
    parser.add_argument('url', nargs='?', help='Article URL')
    parser.add_argument('--search', help='Search term to crawl (site search)')
    default_text_out = os.path.join(os.path.dirname(__file__), 'rts_text.txt')
    default_audio_out = os.path.join(os.path.dirname(__file__), 'rts_audio')
    parser.add_argument('--audio-out', help='Output audio filename', default=default_audio_out)
    parser.add_argument('--text-out', help='Output text filename', default=default_text_out)
    parser.add_argument('--out-dir', help='Output directory for search results', default=os.path.join(os.path.dirname(__file__), 'rts_results'))
    args = parser.parse_args()

    if args.search:
        safe_term = re.sub(r'[^a-z0-9]+', '_', args.search.lower())
        out_dir = os.path.join(args.out_dir, safe_term)
        crawl_search(args.search, out_dir)
        return

    if not args.url:
        print('No URL provided. Use positional URL or --search for crawling.')
        return

    url = args.url
    print(f'Fetching page: {url}')
    html = get_html(url)

    print('Extracting text...')
    info = extract_text(html)
    # skip if article older than 2024-01-01 when date is present
    date_str = info.get('date', '') or ''
    mdate = re.search(r"(\d{1,2}\.\d{1,2}\.\d{4})", date_str)
    if mdate:
        try:
            art_date = datetime.strptime(mdate.group(1), '%d.%m.%Y').date()
            if art_date < date(2024, 1, 1):
                print(f'Article date {art_date} is older than 2024-01-01 — skipping.')
                return
        except Exception:
            pass
    save_text(info, args.text_out)
    print(f'Saved text -> {args.text_out}')

    print('Searching for audio URL...')
    audio_url = find_audio_url(html)
    if not audio_url:
        print('Audio not found in page source.')
        return

    # make absolute if needed
    if audio_url.startswith('//'):
        audio_url = 'https:' + audio_url
    if audio_url.startswith('/'):
        parsed = urlparse(url)
        audio_url = f'{parsed.scheme}://{parsed.netloc}{audio_url}'

    fname = safe_filename_from_url(audio_url)
    # allow override base name
    if args.audio_out and not args.audio_out.lower().endswith(('.mp3', '.m4a', '.ogg', '.wav', '.aac')):
        out_audio = args.audio_out + os.path.splitext(fname)[1]
    else:
        out_audio = args.audio_out if args.audio_out else fname

    print(f'Downloading audio: {audio_url} -> {out_audio}')
    try:
        download_file(audio_url, out_audio, referer=url)
        print(f'Saved audio -> {out_audio}')
    except Exception as e:
        print('Failed to download audio:', e)


if __name__ == '__main__':
    main()
