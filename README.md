# RTS scraper

Jednostavna Python skripta koja preuzima audio (ako postoji) i izvlači tekst iz RTS vesti.

Quick start:

```bash
python scripts/scrape_rts.py <URL> --text-out article.txt --audio-out audio
```

Primer (default URL je članak koji ste dali):

```bash
python scripts/scrape_rts.py
```

Napomene:
- Skripta traži audio fajl u tagovima `<audio>` i kroz regularne izraze u izvoru stranice.
- Ako audio URL počinje sa `//` ili `/`, skripta će ga pretvoriti u apsolutni URL.
- Ako nije pronađen audio, fajl neće biti preuzet ali će tekst biti sačuvan.

Instalacija zavisnosti:

```bash
pip install -r requirements.txt
```
