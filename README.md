Sistem za automatsko prepoznavanje srpskog govora

Autori: Miloš Milanović, Bojan Živanić

--------------------------------------------------------------------------------

Sažetak
--------
Ovaj repozitorijum sadrži implementaciju i eksperimentalnu infrastrukturu za
razvoj sistema automatskog prepoznavanja govora (ASR) za srpski jezik, sa
posebnim fokusom na domen vesti i informativnih emisija (RTS). Projekat koristi
pristup transfer learning-a: fine-tuning modernih pre‑treniranih modela (npr.
Whisper, Wav2Vec2, HuBERT) na domenskom korpusu audio–transkripcija. Glavni
ciljevi su: izgradnja reproduktivnog pipelinesa, duboka EDA analiza uticaja
kvaliteta podataka, i rigorozna, uparena evaluacija baseline vs finetuned
modela.


Sadržaj
-------
- Uvod i motivacija
- Ciljevi i doprinosi
- Pregled relevantne literature
- Skup podataka i QA
- Metodologija i modeli
- Eksperimentalni protokol i metrike
- Rezultati i zaključci iz EDA
- Instrukcije za reprodukciju
- Struktura repozitorijuma
- Dalji pravci i reference

1. Uvod i motivacija
--------------------
Potreba za automatskom transkripcijom vesti i informativnih emisija proizlazi
iz praktičnih zahteva za pristupačnost (titlovanje), arhiviranjem i semantičkom
analitikom medijskih sadržaja. U Srbiji, ovaj problem je posebno relevantan zbog
niskog pokrivenja kvalitetnim real‑time titlovima. Cilj projekta je prilagoditi
state‑of‑the‑art modele za srpski jezik i empirijski pokazati poboljšanje.

2. Ciljevi i doprinosi
---------------------
- Konstrukcija reproduktivnog eksperimentalnog protokola za fine‑tuning ASR
  modela na srpskom jeziku.
- Detaljna EDA analiza uticaja akustičkih i tekstualnih karakteristika na greške
  modela (korelacije: ZCR/SNR vs fonetske greške, taxonomy grešaka, OOV analize).
- Rigorozna, uparena evaluacija baseline i finetuned modela sa statističkim
  testiranjem signifikantnosti razlika.

3. Relevantna literatura
------------------------
- Whisper — veliki multilingvalni encoder‑decoder model treniran na stotinama
  hiljada sati audio materijala; dobar za zero‑/few‑shot i domen‑specifičan
  fine‑tuning.

4. Skup podataka
----------------
Glavni korpus: RTS audio‑transkripcioni parovi (MP3 + .txt). Podaci su raznoliki po temi i
akustičkim uslovima; transkripti sadrže mešavinu latinice i ćirilice.

Quality kontrola: EDA identifikuje "suspicious" chunkove i preporučuje QA
pragove (aligned_word_ratio, max_chars_per_second) koje se primenjuju pre treninga.

5. Metodologija
---------------
Transfer learning: fine‑tuning pre‑treniranih modela sa sledećim strategijama:

- Whisper: opcija zamrzavanja encoder‑a i treniranja decoder‑a radi smanjenja
  memorijskih zahteva i bržeg konvergiranja na domenskom jeziku.

Preprocessing: log‑mel spektrogram, tekstualna normalizacija (konzistentno
pisanje, normalizacija brojeva i interpunkcije) i QA filtriranje.

6. Eksperimentalni protokol i metrike
-------------------------------------
- Podela: trening/validacija/test ≈ 80/10/10 (strogi article‑level holdout
  moguć kroz `tools/create_holdout.py`).
- Metrike: primarni WER, sekundarni CER. Statističko testiranje (npr.
  Mann‑Whitney) za potvrdu signifikantnosti poboljšanja.

7. Rezultati — sažetak iz EDA notebooka
-------------------------------------------------------
Glavne nalaze sažima `notebooks/Kaggle_ASR_EDA_Definitive.ipynb` (EDA notebook):

- Fine‑tuning dovodi do statistički značajnog smanjenja WER‑a u uparenim
  poređenjima na identičnim test primerima.
- Kvalitet podataka (QA filtriranje) je od presudnog značaja: loši parovi
  narušavaju trening i evaluaciju; preporučuje se striktna QA procedura.
- Akustički režimi (high ZCR + low SNR) su povezani sa specifičnim fonetskim
  greškama (frikativne supstitucije) — ciljane augmentacije su preporučene.

Za numeričke vrednosti i tabelarni digest sledite redosled:
1. Pokrenite `notebooks/Kaggle_Full_ASR_Run.ipynb` (full run) da generišete
   finetuned run artefakte (npr. `new_res/asr_full_run_v1`).
2. Pokrenite `notebooks/Kaggle_ASR_EDA_Definitive.ipynb` (EDA notebook) — on
   očekuje dataset i output iz full‑run‑a i automatski generiše `digest` i
   `summary_table_df` sa preciznim brojevima.

8. Reprodukcija — osnovni koraci
--------------------------------
1) Okruženje:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix
source .venv/bin/activate
pip install -r requirements.txt
```

2) Kreiranje holdout‑a (primer):

```bash
python tools/create_holdout.py --raw_dir data/raw --out_dir data/holdout --holdout_ratio 0.15
```

3) Treniranje (primer Whisper):

```bash
python cli/train.py --data_dir data/raw --output_dir models/whisper --model_name openai/whisper-base --model_type whisper
```

4) Evaluacija:

```bash
python cli/eval_model.py --data_dir data/raw --model_dir models/whisper/final --model_type whisper --split test
```

5) EDA i digest:

Preporučeni redosled za reprodukciju EDA digest‑a:

1. Ako pokrećete puni eksperiment na Kaggle‑u (ili lokalno reproducirate run),
  prvo pokrenite `notebooks/Kaggle_Full_ASR_Run.ipynb` da biste generisali
  artefakte finetuned run‑a (npr. `new_res/asr_full_run_v1`).
2. Zatim otvorite i pokrenite `notebooks/Kaggle_ASR_EDA_Definitive.ipynb` —
  prva ćelija rešava kanonične putanje (KAGGLE_INPUT / KAGGLE_WORKING ili
  lokalne putanje) i pristupa artefaktima koje je proizveo full run.

6) Pokretanje na Kaggle platformi (opcionalno)
---------------------------------------------
Ovaj projekat je spreman za izvođenje i na Kaggle platformi. Tipični workflow za
pokretanje na Kaggle‑u zahteva da se (a) kod zapakuje u jedan bundle koji se
uploaduje kao dataset ili se doda kao "code" input, i (b) da se doda dataset sa
audio/transkript fajlovima u Notebook session. Kako biste to uradili lokalno:

```bash
# Iz repozitorijuma, kreira se Kaggle bundle zip (skripta u tools/)
python tools/build_kaggle.py --bundle_name kaggle_bundle --zip

# Output:
#  - dist/kaggle_bundle/   (folder sa pakovanim kodom)
#  - dist/kaggle_bundle.zip (zip za upload na Kaggle)
```

Na Kaggle‑u kreirajte novi Notebook i u sekciji „Add data“ dodajte:
- `dist/kaggle_bundle.zip` (ili raspakovan sadržaj kao dataset),
- Vaš dataset sa audio/transkript parovima (npr. `speech-recognation-raw`).

Preporučeni workflow na Kaggle‑u:

1. Učitajte `dist/kaggle_bundle.zip` (ili raspakovan kod) i dataset sa audio
  fajlovima u istu Kaggle session.
2. Otvorite i pokrenite `notebooks/Kaggle_Full_ASR_Run.ipynb` kao primarni
  entrypoint za full fine‑tuning / evaluaciju na Kaggle infrastrukturi.
3. Nakon što full run završi, u istoj session‑u pokrenite
  `notebooks/Kaggle_ASR_EDA_Definitive.ipynb` da biste iz generisanih artefakata
  (npr. `new_res/asr_full_run_v1`) proizveli EDA, digest i tabele sa numeričkim
  rezultatima.

Notebook `notebooks/Kaggle_ASR_EDA_Definitive.ipynb` je dizajniran da
automatski pronađe kanonične putanje u `KAGGLE_INPUT` i `KAGGLE_WORKING`; to
olakšava direktno pokretanje EDA‑e u istom Kaggle session‑u nakon full run‑a.

Napomena: ako koristite alternativni tooling ili želite da bundle bude kreiran
sa drugačijom strukturom, možete prilagoditi `tools/build_kaggle.py` ili
iskoristiti alatke u `tools/` za repacking. Preporučuje se da uploadujete zip
iz `dist/` da biste izbegli dupliciranje i kako biste imali stabilnu verziju za
Kaggle runs.

9. Struktura repozitorijuma
---------------------------
- `src/` — core implementation (data loaders, model adapters, inference).
- `cli/` — entrypoint skripte (train, eval, transcribe, compare).
- `tools/` — pomoćne skripte (EDA, packing, scraping, pipeline orchestration).
- `notebooks/` — interaktivna analiza i automatski sažeci.
- `data/`, `models/`, `logs/`, `dist/`, `new_res/` — artefakti i eksperimenti.
- `docs/` — specifikacije i zahtevi (polazna dokumentacija projekta).

10. Dalji rad i preporuke
-------------------------
- Fokus na ciljanu augmentaciju i dodatno prikupljanje primera u akustičkim
  režimima gde model pokazuje sistemske greške.
- Istraživanje semi‑supervised strategija (lokalni pretraining Wav2Vec2) za
  maksimalnu upotrebu neoznačenih audio‑podataka.

Kontakt i autorska prava
------------------------
Za pitanja i saradnju kontaktirajte autore navedene na vrhu README‑a. Kod i
skripte su namenjeni istraživačkoj upotrebi; proverite dodatne licence za
spoljna pretrained modela (npr. Whisper licencu) pre komercijalne upotrebe.

