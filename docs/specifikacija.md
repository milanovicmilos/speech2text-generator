Predlog projekta iz SIAP-a

Ovaj dokument sadrži kratak opis teme projekta, definiciju problema, motivaciju za odabranu temu, pregled relevantne literature, opis skupa podataka, predloženu metodologiju i metod evaluacije. Na kraju dokumenta se nalazi i plan realizacije projekta.

Tema projekta je sistem za automatsko prepoznavanje srpskog govora (Automatic Speech Recognition - ASR) korišćenjem modernih modela dubokog učenja. Cilj je primeniti i evaluirati različite pristupe za transkripciju audio snimaka na srpskom jeziku, sa fokusom na domen vesti i informativnih emisija (dnevnik, reportaže, podkasti) sa Radio Televizije Srbije.

1. Definicija problema

Porebno je razviti sistem za automatsko prepoznavanje govora koji će primati audio fajl na srpskom jeziku i generisati odgovarajuću tekstualnu transkripciju. Glavni cilj je prilagođavanje (fine-tuning) postojećih pretrenirani modela za prepoznavanje govora na srpski jezik, posebno na domen vesti i informativnih emisija sa Radio Televizije Srbije (dnevnik, reportaže, podkasti). Razmatraćemo primenu nekoliko savremenih modela (Whisper, Wav2Vec2, HuBERT) i uporediti njihove performanse na našem skupu podataka.

2. Motivacija problema rešavanog u projektu

Automatska transkripcija audio sadržaja sa vesti i informativnih emisija ima važne praktične primene, posebno za osobe sa oštećenim sluhom. Sistem bi mogao automatski generisati titlove za TV dnevnik, informativne emisije i podkaste, čineći ovaj sadržaj pristupačnijim širokoj publici. Ovo je od posebnog značaja jer trenutno većina TV programa u Srbiji nema kvalitetne titlove u realnom vremenu.

Pored pristupačnosti, tekstualne transkripcije omogućavaju pretraživanje kroz velike arhive vesti, što je od značaja za novinare, istraživače, i analitičare koji žele brzo da pronađu specifične informacije iz prošlih emisija. Takođe, automatska transkripcija štedi vreme i novac jer eliminiše potrebu za ručnim prepisivanjem audio materijala.

Za srpski jezik postoji ograničen broj dostupnih ASR sistema visokog kvaliteta u poređenju sa engleskim ili drugim široko rasprostranjenim jezicima. Srpski spada u low-resource jezike, što znači da je količina javno dostupnih audio podataka sa transkripcijama znatno manja. Razvoj funkcionalnih ASR sistema za srpski jezik može doprineti smanjenju ovog jaza i omogućiti lokalnim medijskim kućama pristup modernim tehnologijama prepoznavanja govora.

3. Relevantna literatura

[1] Alec Radford, Jong Wook Kim, Tao Xu, et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision"
https://arxiv.org/abs/2212.04356

Tema rada: Rad predstavlja Whisper, model za automatsko prepoznavanje govora treniran na 680,000 sati multilingvalnog i multitask audio materijala prikupljenog sa interneta. Model je dizajniran kao encoder-decoder Transformer arhitektura koja može vršiti transkripciju, identifikaciju jezika, i prevođenje govora.

Podaci: Skup podataka obuhvata 680,000 sati audio zapisa na 117 različitih jezika, prikupljenih sa interneta (YouTube video titlovi, podkasti, javno dostupni audio materijal). Dataset je raznovrstan po domenima - razgovori, konferencije, podkasti, audio knjige.

Metodologija: Whisper koristi encoder-decoder Transformer arhitekturu. Audio signal se konvertuje u log-mel spektrogram sa 80 filterskih kanala, encoder procesira audio features i kreira latentne reprezentacije, dok decoder autoregressivno generiše tekst. Trening se vrši korišćenjem cross-entropy loss funkcije.

Evaluacija: Model je evaluiran na višem poznatim benchmark dataset-ovima kao što su LibriSpeech, Common Voice i Fleurs, korišćenjem Word Error Rate (WER) metrike koja meri procenat grešaka u transkripciji. Whisper-base model postiže WER od oko 10-15% na clean audio za engleski jezik.

Rezultati: Whisper modeli pokazuju dobru robusnost na šum i različite akustičke uslove zahvaljujući raznolikosti podataka tokom treniranja. Model se može koristiti bez dodatnog treniranja (zero-shot transfer), ali fine-tuning na specifičnom domenu i jeziku značajno poboljšava performanse.

Zaključak: U našem projektu ćemo koristiti Whisper kao jedan od glavnih kandidata za ASR sistem. Planiramo da primenimo fine-tuning pristup da prilagodimo pretreniran model na specifičnosti srpskog jezika i domena vesti i informativnih emisija. Očekujemo da će multilingvalna priroda Whisper modela omogućiti brže učenje na srpskom jeziku.

[2] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
https://arxiv.org/abs/2006.11477

Tema rada: Rad predstavlja Wav2Vec2, model koji koristi self-supervised learning za učenje reprezentacija govora iz neoznačenih audio podataka. Model se prvo pretrenira na velikim količinama unlabeled audio materijala, a zatim se fine-tunuje na manjim količinama labeled podataka za ASR zadatke.

Podaci: Za pretraining je korišćen LibriSpeech dataset sa 960 sati čitanog govora na engleskom jeziku (bez labela). Fine-tuning je vršen na značajno manjim labeled podsetima - eksperimenti su rađeni sa 100 sati, 10 sati, pa čak i 1 sat labeled podataka.

Metodologija: Wav2Vec2 koristi contrastive learning pristup - audio se enkoduje u latentne reprezentacije, deo reprezentacija se maskira, i model uči da razlikuje prave maskirane vrednosti od distraktora. Nakon pretreninga, dodaje se CTC (Connectionist Temporal Classification) head za fine-tuning na transkripciju govora.

Evaluacija: Model je evaluiran na LibriSpeech test skupovima korišćenjem WER metrike. Sa potpunim labeled dataset-om (960 sati), Wav2Vec2 postiže WER od 1.8% na test-clean skupu. Čak i sa samo 10 minuta labeled podataka, model postiže upotrebljive rezultate.

Rezultati: Ključni rezultat rada je da Wav2Vec2 može postići odlične ASR performanse čak i sa vrlo malom količinom labeled podataka. To je posebno važno za low-resource jezike gde je labeled audio tešk i skup za prikupljanje. Model pokazuje da se sa dovoljno unlabeled audio materijala može naučiti efikasna reprezentacija govora.

Zaključak: Wav2Vec2 će biti razmatran kao alternativa Whisper modelu u našem projektu. Njegova sposobnost da radi sa malom količinom labeled podataka je ključna prednost za srpski jezik. U projektu ćemo isprobati fine-tuning Wav2Vec2 modela na našem skupu audio snimaka vesti sa RTS-a i uporediti ga sa Whisper pristupom.

[3] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, et al. (2021). "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"
https://arxiv.org/abs/2106.07447

Tema rada: HuBERT (Hidden-Unit BERT) je model za self-supervised learning audio reprezentacija koji koristi masked prediction pristup inspirisan BERT modelom iz oblasti obrade prirodnog jezika. Model uči da predviđa maskirane delove audio signala bez potrebe za labelovanim podacima tokom pretreniranja.

Podaci: Za pretraining je korišćen LibriSpeech dataset sa 960 sati neoznačenog audio materijala. Kao i kod Wav2Vec2, fine-tuning se vrši na manjim labeled podsetima LibriSpeech dataseta.

Metodologija: HuBERT prvo konvertuje audio u continuous features, zatim primenjuje maskiranje (slično BERT-u), i trenira Transformer encoder da predviđa discrete klaster labele za maskirane regione. Model se iterativno poboljšava kroz više faza - u svakoj fazi se vrši novo klasterovanje naučenih reprezentacija, i model se ponovo trenira sa novim klaster labelama.

Evaluacija: HuBERT je evaluiran na LibriSpeech i drugim ASR benchmark dataset-ovima korišćenjem WER metrike. Model postiže rezultate konkurentne sa Wav2Vec2 modelom, a na nekim zadacima i bolje performanse.

Rezultati: HuBERT pokazuje da jednostavniji masked prediction pristup može biti jednako efikasan kao složeniji contrastive learning iz Wav2Vec2. Model pokazuje dobre performanse u low-resource scenarijima i može se efikasno fine-tunovati na nove jezike i domene sa ograničenom količinom podataka.

Zaključak: HuBERT će biti treća opcija koju ćemo razmotriti u projektu. Planiram da uporedim sva tri modela (Whisper, Wav2Vec2, HuBERT) tokom realizacije projekta kako bih utvrdio koji najbolje funkcioniše na srpskom jeziku i domenu vesti i informativnih emisija. Cilj je empirijski evaluirati koji pristup daje najbolje rezultate na našem specifičnom skupu podataka.

4. Skup podataka

Za ovaj projekat ćemo koristiti skup podataka koji je prikupljen sa javno dostupnih izvora - audio snimci vesti i informativnih emisija sa RTS (Radio Televizija Srbije). Svaki audio fajl je u MP3 formatu i ima pridruženu tekstualnu transkripciju u .txt formatu. Podaci su prikupljeni scraping-om javno dostupnih audio snimaka vesti sa RTS platforme.

Trenutno raspolažemo sa 370 audio fajlova sa odgovarajućim transkripcijama. Planiramo da proširimo skup dodatnim prikupljanjem podataka tokom realizacije projekta, sa ciljem da dostignemo oko 500-600 parova (audio, transkripcija). Audio fajlovi su različite dužine trajanja, od nekoliko sekundi do nekoliko minuta. Tekstualne transkripcije su na srpskom jeziku (mešavina ćirilice i latinice) i sadrže izgovoreni sadržaj audio snimka. Domen obuhvata vesti i informativne emisije sa RTS-a iz različitih kategorija - politika, ekonomija, kultura, društvo, sport.

Ciljno obeležje je tekstualna transkripcija koja predstavlja tačnu tekstualnu reprezentaciju onoga što je izgovoreno u audio snimku. Ovo je sekvenca reči koju model treba da nauči da generiše na osnovu audio signala. Skup ćemo podeliti na trening, validacioni i test set u omeru 80% / 10% / 10%, što znači oko 296 audio fajlova za trening, 37 za validaciju, i 37 za test.

Pre treniranja, audio fajlovi će biti konvertovani u WAV format sa 16kHz sample rate (standardni format za ASR modele). Transkripcije će biti normalizovane - konvertovane u mala slova, interpunkcija će biti uklonjena, višestruki razmaci će biti zamenjeni jednim razmakom. Razmatramo i dodavanje data augmentacije kao što su pitch shifting, speed perturbation i dodavanje šuma radi povećanja robusnosti modela na različite akustičke uslove.

5. Metodologija

U projektu ćemo koristiti transfer learning pristup - uzimamo pretreniran model koji je naučio opšte reprezentacije govora na velikom skupu podataka, i fine-tunujemo ga na našem manjem skupu podataka na srpskom jeziku. Razmatraćemo tri modela: OpenAI Whisper (base ili small verzija) koji je multilingvalni encoder-decoder model i već podržava više od 100 jezika, Wav2Vec2 koji je self-supervised model i može se fine-tunovati dodavanjem CTC head-a, i HuBERT koji je sličan Wav2Vec2 i koristi masked prediction za pretraining.

Za Whisper razmatramo zamrzavanje encoder dela modela (koji je naučio opšte akustičke features) i treniranje samo decoder dela koji uči specifičnosti srpskog jezika i ortografije, što smanjuje broj parametara koje treba trenirati i ubrzava proces. Za Wav2Vec2 i HuBERT dodaćemo CTC (Connectionist Temporal Classification) head na vrh pretrenirane feature encoder mreže i fine-tunovaćemo model na labeled audio podacima. Preliminarni hiperparametri su learning rate od 1e-5 ili 5e-5 (niže vrednosti za fine-tuning), batch size 4-8 u zavisnosti od dostupnih hardverskih resursa, 5-10 epoha uz praćenje validation loss-a radi ranog zaustavljanja, i AdamW optimizer sa weight decay. Ovi parametri će biti optimizovani tokom eksperimenata.

Koristićemo HuggingFace Transformers biblioteku koja ima gotove implementacije svih pomenutih modela i podržava jednostavan fine-tuning kroz Trainer API. Za procesiranje audio podataka koristićemo Librosa biblioteku, a za data loading koristićemo HuggingFace Datasets biblioteku.

6. Metod evaluacije

Podatke smo podelili na trening, validacioni i test skup u omeru 80/10/10. Tokom treninga ćemo pratiti validation loss nakon svake epohe kako bismo pratili da li model napreduje i da bismo sprečili overfitting (rano zaustavljanje). Nakon završenog treninga, izvršićemo finalnu evaluaciju na test skupu koji model nije video tokom treninga.

Koristićemo Word Error Rate (WER) kao primarnu metriku evaluacije. WER se računa kao (Broj supstitucija + Broj brisanja + Broj umetanja) / Ukupan broj reči u referenci. WER meri procenat grešaka u generisanoj transkripciji u odnosu na tačnu transkripciju. Niža vrednost WER-a je bolja - WER od 0% znači savršenu transkripciju, dok WER od 100% ili više označava da model generiše potpuno pogrešne transkripcije. Dodatno ćemo posmatrati i Character Error Rate (CER) kao sekundarnu metriku, koja računa greške na nivou karaktera umesto reči.

S obzirom na relativno malu količinu podataka (370 primera), ne očekujemo da ćemo postići state-of-the-art rezultate. Realna očekivanja su WER između 20-40% nakon nekoliko epoha fine-tuninga. Za poređenje, engleski ASR sistemi na clean audio postižu WER ispod 5%, ali rade na mnogo većim dataset-ovima. Cilj projekta nije dostizanje najnižeg mogućeg WER-a, već evaluacija i poređenje različitih pristupa na srpskom jeziku.

7. Softver

Za realizaciju projekta koristićemo Python 3.12 ili 3.13 kao programski jezik. PyTorch će biti korišćen kao deep learning framework za treniranje neuronskih mreža. HuggingFace Transformers biblioteka sadrži gotove implementacije Whisper, Wav2Vec2 i HuBERT modela i podržava jednostavan fine-tuning kroz Trainer API. Za učitavanje, procesiranje i analizu audio fajlova koristićemo Librosa biblioteku, dok će soundfile biti korišćen za čitanje i pisanje audio fajlova u različitim formatima. HuggingFace Datasets biblioteku ćemo koristiti za efikasan data loading i procesiranje, a evaluate biblioteku za računanje evaluacionih metrika (WER, CER). NumPy i Pandas će biti korišćeni za manipulaciju podacima, dok će Matplotlib biti korišćen za vizualizaciju rezultata.

Razvojno okruženje će biti VS Code editor sa Python ekstenzijom, Python virtual environment za izolaciju dependencija, i Git za version control. Projekat će biti realizovan na lokalnoj mašini. Ako budu dostupni GPU resursi, koristićemo ih za ubrzanje treninga, ali projekat može biti realizovan i na CPU-u (uz duže vreme treninga).

8. Plan realizacije

Plana rada na projektu obuhvata sledeće faze (milestones):

1. Priprema podataka i eksplorativna analiza (prvih 2-3 nedelje)
   - Verifikacija kvaliteta prikupljenih audio fajlova i transkripcija
   - Statistička analiza skupa (distribucija dužina audio fajlova, distribucija dužina transkripcija, najčešće reči)
   - Implementacija preprocessing pipeline-a (konverzija u WAV, normalizacija teksta)
   - Podela na train/val/test skupove

2. Implementacija baseline modela (naredne 2 nedelje)
   - Implementacija data loading-a za HuggingFace Transformers
   - Po

Plana rada na projektu obuhvata sledeće bitne tačke (milestones):
• Prikupljanje i proširenje skupa podataka
• Verifikacija kvaliteta audio fajlova i transkripcija
• Eksplorativna analiza podataka i preprocessing (konverzija u WAV, normalizacija teksta)
• Implementacija data loading pipeline-a za HuggingFace Transformers
• Fine-tuning Whisper modela sa različitim hiperparametrima
• Fine-tuning Wav2Vec2 modela
• Fine-tuning HuBERT modela
• Poređenje performansi sva tri modela na validation setu
• Optimizacija najbolje performujućeg modela
• Finalna evaluacija na test skupu
• Kvalitativna analiza grešaka
• Dokumentacija i pisanje finalnog izveštaja