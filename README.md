# Použiteľnosť prediktívnych modelov výskytu klostrídiovej infekcie počas vĺn COVID-19 vzhľadom na podávanú liečbu

Tento repozitár obsahuje zdrojový kód k diplomovej práci zameranej na návrh, implementáciu a vyhodnotenie prediktívnych modelov pre odhad rizika vzniku infekcie *Clostridioides difficile (CDI)* u hospitalizovaných pacientov s ochorením COVID-19.

Súčasťou riešenia je webová aplikácia, ktorá umožňuje pracovať s vytvorenými modelmi prostredníctvom používateľského rozhrania dostupného vo webovom prehliadači.

## Charakteristika riešenia

Aplikácia je navrhnutá ako klient–server systém. Serverová časť je implementovaná v jazyku Python s využitím frameworku Flask a zabezpečuje spracovanie vstupných údajov a samotnú predikciu. Používateľ pristupuje k aplikácii prostredníctvom webového prehliadača.

Z pohľadu použitia ide o nástroj, ktorý umožňuje zadať vybrané údaje o pacientovi, vykonať odhad rizika CDI a zároveň získať interpretáciu výsledku.

## Funkcionalita aplikácie

Aplikácia pozostáva z viacerých častí:

- **Predikcia rizika CDI** – formulár na zadanie vstupných údajov pacienta (demografia, laboratórne hodnoty, komorbidity, liečba)  
- **Výsledok predikcie** – zobrazenie pravdepodobnosti CDI, klasifikácie (CDI+ / CDI−) a relatívneho rizika  
- **Vysvetlenie predikcie** – interpretácia výsledku pomocou metód SHAP, LIME alebo na základe odds ratio pri modeli logistickej regresie  
- **Analýza klinických dát** – prehľad grafov a vizualizácií vytvorených z dát  
- **O projekte** – základné informácie o dátach, modeloch a metodike  

## Štruktúra projektu

Projekt je rozdelený do viacerých logických častí:

- `run.py` – hlavný súbor na spustenie aplikácie  
- `start_demo.bat` – pomocný súbor pre jednoduché spustenie aplikácie vo Windows  
- `requirements.txt` – zoznam závislostí projektu  

### Serverová časť

- `app/` – hlavný balík aplikácie obsahujúci serverovú logiku  
  - `routes/` – definície rout pre jednotlivé časti aplikácie (predikcia, stránky, analytická sekcia)  
  - `services/` – moduly zabezpečujúce spracovanie dát a vysvetliteľnosť
  - ostatné moduly – konfigurácia a pomocné funkcie  

### Používateľské rozhranie

- `templates/` – HTML šablóny aplikácie  
  - `pages/` – jednotlivé stránky aplikácie (úvod, formulár, výsledok, analytická časť, o projekte)  
  - `partials/` – znovupoužiteľné komponenty rozhrania  

### Statické súbory

- `static/` – statické súbory používané vo frontend časti  
  - `css/` – štýly aplikácie  
  - `js/` – súbory zabezpečujúce funkcionalitu používateľského rozhrania  
  - `plots/` – pripravené grafy a vizualizácie klinických dát  

### Notebooky

- `pochopenie_dat.ipynb` – exploratívna analýza dát  
- `priprava_dat.ipynb` – čistenie a predspracovanie dát  
- `modelovanie.ipynb` – trénovanie a porovnanie modelov  
- `grafy_aplikacia.ipynb` – príprava grafov využitých v aplikácii  

### Modely

- `models_2/` – priečinok pre modelové súbory a imputery (nie je súčasťou repozitára, je potrebné ich stiahnuť zo sekcie Releases)

## Dostupnosť modelov

Modely a imputery nie sú priamo súčasťou repozitára z dôvodu ich veľkosti. Sú dostupné v sekcii **Releases**.

Pre správne fungovanie aplikácie je potrebné tieto súbory stiahnuť a uložiť do priečinka `models_2` v koreňovom adresári projektu.

## Systémové požiadavky

Na spustenie je potrebné mať nainštalovaný Python a všetky knižnice uvedené v súbore `requirements.txt`.

## Spustenie aplikácie

Aplikácia je určená na lokálne spustenie.

### 1. Naklonovanie repozitára
```bash
git clone <repo-url>
cd <repo>
```

### 2. Vytvorenie priečinka pre modely
```bash
mkdir models_2
```

### 3. Stiahnutie modelov
Stiahni modely zo sekcie **Releases** a vlož ich do priečinka `models_2`.

### 4. Inštalácia závislostí
```bash
pip install -r requirements.txt
```

### 5. Spustenie aplikácie

#### Možnosť 1 – Python
```bash
python run.py
```

#### Možnosť 2 – Windows
```powershell
start_demo.bat
```

Po spustení je aplikácia dostupná na adrese:
```text
http://127.0.0.1:5000
```

## Dáta

Použité dáta pochádzajú z anonymizovaných klinických záznamov pacientov. Z dôvodu ochrany osobných údajov nie sú tieto dáta súčasťou repozitára.
