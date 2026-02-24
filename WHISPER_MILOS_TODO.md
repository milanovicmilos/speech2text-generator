# Whisper-only TODO (Miloš)

Ovaj plan pokriva **kompletan deo jednog studenta** za predmetni projekat, fokusiran isključivo na Whisper liniju rada (bez implementacije koleginog modela).

## 0) Scope i granice odgovornosti

- Tvoj deo: kompletna Whisper implementacija, eksperimentalna procedura, validacija, analiza grešaka, dokumentacija i odbrambeni materijal.
- Kolegin deo: implementacija i evaluacija alternativnog modela (npr. Wav2Vec2/HuBERT) i/ili dodatni eksperimentalni pravac.
- Integracija na kraju: zajednička tabela poređenja modela i zajednički zaključak.

---

## 1) Kod i infrastruktura (Whisper pipeline hardening)

### 1.1 Trening reproducibilnost
- [x] Uskladiti `seed` kroz CLI/config (bez hardkodovanja).
- [x] Uskladiti `gradient_accumulation_steps`, `warmup_steps` i ostale argumente da stvarno ulaze u trening.
- [x] Sačuvati kompletan `run_config.yaml` uz svaki trening run.

**Done kriterijum:** isti komandni poziv sa istim seed-om daje konzistentan trend metrika i identičan split podataka.

### 1.2 Korektna evaluacija u treningu
- [x] Uključiti generativnu evaluaciju (`predict_with_generate`) umesto sirovog `argmax` preko logits.
- [x] Verifikovati da `WER` i `CER` imaju smislen opseg i trend.
- [x] Sačuvati `trainer_state.json` + tekstualni summary najboljeg checkpoint-a.

**Done kriterijum:** `best_metric` i epoch eval metrika su interpretabilne i dosledne između logova i summary-ja.

### 1.3 Data split bez leakage-a
- [x] Uvesti group-aware split po izvornom dokumentu (chunk-ovi istog izvora ne smeju prelaziti između train/val/test).
- [x] Logovati broj grupa i raspodelu po split-u.

**Done kriterijum:** nijedna grupa (source stem) nije prisutna u više splitova.

### 1.4 Zavisnosti i skripte
- [x] Ažurirati `requirements.txt` da pokriva realno korišćene pakete.
- [x] Srediti pomoćne skripte (`gpu_train.sh` i sl.) da odgovaraju aktuelnom CLI interfejsu.

**Done kriterijum:** čista instalacija + pokretanje osnovnih komandi bez missing dependency grešaka.

---

## 2) Eksperimenti koje moraš da imaš za odbranu

## 2.1 Baseline u okviru Whisper rada (minimalno)
- [x] Definisati **Whisper baseline** (npr. `openai/whisper-base` bez LoRA, bez unfreeze, default decode).
- [x] Definisati **improved Whisper** (npr. LoRA ili unfreeze + tuned decode parametri).
- [x] Trenirati/evaluirati oba pod istim splitom.
- [x] Izvesti formalno poređenje baseline vs fine-tuned pod istim decode parametrima.

**Done kriterijum:** tabela `baseline vs improved` sa WER/CER i kratkim objašnjenjem dobitka.

## 2.2 Ablation za tvoj deo
- [x] Bar 2 kontrolisana poređenja (npr. LoRA on/off, unfreeze on/off, chunked vs raw, decode settings A/B).
- [x] Evidentirati samo promenu jednog faktora po eksperimentu.

**Done kriterijum:** postoji jasan odgovor „šta je donelo poboljšanje, a šta nije“.

## 2.3 Stabilna finalna evaluacija
- [x] Fiksirati finalni model i finalni decoding setup.
- [x] Pokrenuti evaluaciju na test splitu i sačuvati rezultate.
- [x] Sačuvati 20+ primera predikcija (ref/pred) za kvalitativnu analizu.

**Done kriterijum:** finalni broj + dokaz (log, json/csv, primeri predikcija).

---

## 3) Analiza grešaka (obavezno za višu ocenu)

- [x] Kategorizovati greške: brojevi, imena, entiteti, duge rečenice, šum, dijalekat/izgovor, ponavljanja.
- [x] Izvući najmanje 3 konkretna pattern-a i predlog mitigacije.
- [x] Prikazati kako su tuning/normalizacija uticali na specifične tipove grešaka.

**Done kriterijum:** sekcija „Error Analysis“ sa primerima i zaključcima, ne samo numerikom.

---

## 4) Artefakti koje moraš predati/poneti

- [x] Final model path + best checkpoint path.
- [x] `trainer_state.json`, eval log, config run-a.
- [x] Tabela eksperimenata (CSV/MD).
- [x] Spisak komandi za reprodukciju (u ovom TODO dokumentu + CLI help).
- [x] EDA statistika skupa podataka i split-integrity izveštaj.

**Done kriterijum:** drugi član tima može da reprodukuje tvoj finalni rezultat iz README/komandi.

---

## 5) Podela rada (da kolegi ostane čist prostor)

### Tvoje (Miloš)
- Whisper trening/eval pipeline
- Svi Whisper eksperimenti + error analysis
- Reproducibility i dokumentacija tvog dela

### Kolega
- Drugi model (Wav2Vec2 ili HuBERT)
- Njegovi eksperimenti + metrika
- Integracija u zajedničku tabelu poređenja

---

## 6) Operativni redosled (izvršenje)

1. Stabilizacija koda i zavisnosti
2. Reproducibilan split i trening
3. Baseline + improved Whisper
4. Ablation eksperimenti
5. Final test + error analysis
6. Paketovanje artefakata za izveštaj/odbranu

---

## 7) Minimalni acceptance checklist (pred odbranu)

- [x] Imam baseline i improved Whisper sa istim splitom.
- [x] Imam validne WER/CER rezultate i logove.
- [x] Nemam data leakage između splitova.
- [x] Imam kvalitativnu analizu grešaka sa primerima.
- [x] Imam jasan opis šta je moj doprinos, a šta kolegin.
- [x] Mogu da reprodukujem rezultat jednom komandom po fazi.
