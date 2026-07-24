_Read the read me in [English](README.md)._


# Guida a Colibrì - Motore di Inferenza Locale

Una guida semplice per eseguire **Colibrì**, un motore di inferenza locale basato su GLM 5.2, senza conoscenze di programmazione. Se hai già Docker installato, sei a buon punto.

---

## 📋 Sommario

- [Cosa è Colibrì?](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#cosa-%C3%A8-colibr%C3%AC)
- [Cosa serve](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#cosa-serve)
    - [Hardware](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#hardware)
    - [Software](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#software)
- [Come iniziare](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#come-iniziare)
    - [Passo 1: Scarica il modello](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#passo-1-scarica-il-modello)
    - [Passo 2: Scarica il Dockerfile di Colibrì](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#passo-2-scarica-il-dockerfile-di-colibr%C3%AC)
    - [Passo 3: Compila l'immagine Docker](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#passo-3-compila-limmagine-docker)
    - [Passo 4: Avvia Colibrì](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#passo-4-avvia-colibr%C3%AC)
    - [Cosa significa quel comando?](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#cosa-significa-quel-comando)
    - [Usare Colibrì](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#usare-colibr%C3%AC)
- [Entrare nel container](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#entrare-nel-container)
- [Risoluzione dei problemi](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#risoluzione-dei-problemi)
- [Note tecniche](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#note-tecniche)
- [Domande frequenti](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#domande-frequenti)
- [Supporto e contributi](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#supporto-e-contributi)
- [Testing on a low resource PC](https://github.com/JustVugg/colibri/blob/main/docker/README.IT.md#test-su-un-pc-con-poche-risorse)

---

## Cosa è Colibrì?

Colibrì è un'applicazione che ti permette di eseguire un modello di intelligenza artificiale (GLM 5.2) direttamente sul tuo computer, senza connettersi a server esterni. È possbile anche farlo girare in Docker, che isola l'applicazione dal resto del sistema.

> **Nota importante**: Il modello è molto grande. Attendi anche diversi minuti per una risposta a una domanda semplice, specialmente con poca RAM. Alla fine di questo readme vedrai il risultato sul mio PC (senza scheda grafica discreta) e arrivo a 0.01 token al secondo.

---

## Cosa serve

### Hardware

| Memoria RAM | Funziona? | Note |
|:---:|:---:|---|
| < 16 GB | ❌ No | Memoria insufficiente |
| 24 GB | ⚠️ Forse | Possibile, da testare |
| 32 GB | ✅ Sì | Il minimo (ma vedi sezione memoria su Windows) |
| 48+ GB | ✅ Sì | Meglio |

Inoltre: un **disco SSD veloce** è essenziale. Colibrì usa il disco come memoria aggiuntiva. Con una scheda grafica NVidia è ancora meglio.

### Software

- **Docker Desktop** (Windows, Mac, Linux) — [scarica qui](https://www.docker.com/products/docker-desktop/)
- **Python** (solo se vuoi scaricare il modello da casa tua)
  - Windows: [python.org](https://www.python.org) oppure Microsoft Store
  - Linux: `apt-get install python3 python3-pip`
  - Mac: [python.org](https://www.python.org) oppure Homebrew

Non serve nessun ambiente di compilazione. Tutto avviene dentro il container Docker.

---

## Come iniziare

### Passo 1: Scarica il modello

Il modello GLM 5.2 è circa **360 GB**. Scegli uno di questi metodi:

#### Metodo A: Con Python (consigliato)

1. **Installa la libreria per Hugging Face:**
   ```bash
   python -m pip install -U huggingface_hub[cli]
   ```
   Su Linux, usa `python3` al posto di `python`.

2. **Scarica il modello** (apri il terminale nella cartella dove lo vuoi salvare):
   ```bash
   hf_download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp --local-dir .
   ```
   
   **Esempio**: se vuoi salvarlo in `C:\LLM\models\glm-5.2` (Windows):
   - Apri PowerShell in quella cartella
   - Copia e incolla il comando sopra
   - Attendi (molto)

#### Metodo B: Senza Python (solo se necessario)

Se sei su Windows e non riesci con Python:
- Scarica manualmente da [Hugging Face](https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp)
- Decomprimi in una cartella (es. `C:\LLM\models\glm-5.2`)

---

### Passo 2: Scarica il Dockerfile di Colibrì

1. Vai a: https://github.com/JustVugg/colibri/blob/main/docker/Dockerfile
2. Clicca il pulsante **Download** (icona ⬇️) in alto a destra
3. Salva il file in una cartella (es. `C:\LLM\Colibrì`)

---

### Passo 3: Compila l'immagine Docker

Apri il terminale (PowerShell su Windows, Terminal su Mac/Linux) **nella cartella dove hai salvato il Dockerfile** e digita:

**Windows:**
```bash
docker build -t colibri-i .
```

**Linux/Mac:**
```bash
sudo docker build -t colibri-i .
```

Attendi che finisca (pochi minuti). Se tutto va bene, vedrai: `Successfully tagged colibri-i:latest`

> **Se vuoi recepire gli aggiornamenti del repository**: Cancella prima l'immagine vecchia con `docker rmi colibri-i` e ricompila.

---

### Passo 4: Avvia Colibrì

Apri il terminale e digita il comando sottostante (sostituisci `C:\LLM\models\glm-5.2` con il percorso reale del tuo PC):

**Windows** (PowerShell):
```bash
$MODEL_PATH="C:\LLM\models\glm-5.2"
docker run --rm -it --name colibri-c `
  -v "$MODEL_PATH`:/app/glm-5.2" `
  -e COLI_MODEL=/app/glm-5.2 `
  colibri-i ./coli chat
```

**Mac/Linux** (Terminal/Bash):
```bash
MODEL_PATH="/path/to/glm-5.2"
docker run --rm -it --name colibri-c \
  -v "$MODEL_PATH:/app/glm-5.2" \
  -e COLI_MODEL=/app/glm-5.2 \
  colibri-i ./coli chat
```

**Esempio per Linux:**
```bash
MODEL_PATH="/home/user/LLM/glm-5.2"
docker run --rm -it --name colibri-c \
  -v "$MODEL_PATH:/app/glm-5.2" \
  -e COLI_MODEL=/app/glm-5.2 \
  colibri-i ./coli chat
```

---

### Cosa significa quel comando?

| Parte | Spiegazione |
|-------|-------------|
| `docker run` | Avvia un container |
| `--rm` | Cancella il container quando chiudi |
| `-it` | Modalità interattiva (puoi scrivere e leggere) |
| `-v "PERCORSO:/app/glm-5.2"` | Collega il tuo modello dentro il container |
| `-e COLI_MODEL=/app/glm-5.2` | Dice a Colibrì dove trovare il modello |
| `colibri-i` | Nome dell'immagine Docker |
| `./coli chat` | Avvia Colibrì in modalità chat |

---

### Usare Colibrì

Una volta avviato, vedrai un prompt come questo:

```
  ──────────────────────────────────────────────────────────
  type and press Enter · Ctrl-C stops the answer · :more continues · :reset clears memory · :q exits
```

**Comandi utili:**
- `Scrivi una domanda + Invio` → Ricevi la risposta
- `Ctrl + C` → Interrompi la risposta
- `:reset` → Cancella la memoria della conversazione
- `:q` → Esci

**Esempio di uso:**
```
› Quanti abitanti ha la Cina?

La Cina è attualmente il paese più popoloso al mondo. 
La popolazione è di circa 1,41 miliardi di abitanti.
```

Il modello capisce **italiano, inglese, cinese e altre lingue**, anche se è ottimizzato per inglese e cinese.

---

## Entrare nel container

Se vuoi esplorare il container come fosse una macchina Linux normale:

```bash
docker run --rm -it --name colibri-c \
  -v "PERCORSO_MODELLO:/app/glm-5.2" \
  -e COLI_MODEL=/app/glm-5.2 \
  colibri-i /bin/bash
```

Ora sei dentro Linux. Digita `exit` per uscire.

---

## Risoluzione dei problemi

### ❌ "Docker non trovato"

**Causa**: Docker non è installato o il terminale non lo riconosce.

**Soluzione**:
1. Reinstalla [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Riavvia il computer
3. Apri un nuovo terminale e riprova

---

### ❌ "Out of memory" (memoria insufficiente) o container che si chiude subito

**Causa**: Il tuo computer non ha abbastanza RAM, oppure su Windows, WSL usa meno memoria di quella disponibile.

**Soluzione per Windows (WSL):**

1. Apri PowerShell e controlla la memoria disponibile a WSL:
   ```bash
   wsl
   cat /proc/meminfo | grep MemTotal
   exit
   ```
   Dividi il numero per 1.073.741.824 (è 1024³) per averlo in GB.

2. Se WSL usa meno di quello che hai, crea un file di configurazione:
   - Apri un editor di testo (Notepad va bene)
   - Copia questo:
     ```ini
     [wsl2]
     memory=24GB
     processors=12
     swap=16GB
     ```
   - Salva il file con il nome: `.wslconfig` (con il punto)
   - Posizionalo in: `C:\Users\TuoNomeUtente\`

3. Riavvia WSL da PowerShell:
   ```bash
   wsl --shutdown
   wsl
   ```

4. Controlla di nuovo:
   ```bash
   # cat /proc/meminfo | grep MemTotal
   # exit
   ```

**Soluzione per Mac/Linux**: Aumenta la RAM disponibile a Docker dalle impostazioni di Docker Desktop, oppure aggiungi più RAM al computer.

---

### ❌ La risposta è molto lenta

**Cause possibili**:
1. Il disco è lento
2. Hai poca RAM
3. Colibrì sta usando il disco come memoria aggiuntiva (normale)

**Come controllare la velocità del disco:**

**Windows** (PowerShell da amministratore):
```bash
winsat disk -drive C
```
Cambia `C` con la lettera del tuo disco.

**Linux/Mac** (Terminal):
```bash
sudo hdparm -Tt /dev/sda
```
Cambia `/dev/sda` con il tuo disco (vedi con `lsblk` per Linux).

Un **SSD NVMe moderno** arriva a 15 GB/sec. Se il tuo è sotto 2-3 GB/sec, è lento.

---

### ❌ "Permission denied" su Linux

**Causa**: Docker richiede permessi da amministratore.

**Soluzione - Opzione 1** (rapida):
```bash
sudo docker build -t colibri-i .
sudo docker run ... (come sopra, con sudo davanti)
```

**Soluzione - Opzione 2** (permanente):
```bash
sudo usermod -aG docker $USER
# Riavvia il computer
docker run ... (senza sudo)
```

---

### ❌ "Image not found" o errore durante il build

**Causa**: Il Dockerfile è corrotto o non nella cartella giusta.

**Soluzione**:
1. Verifica che il Dockerfile sia nella cartella dove apri il terminale:
   ```bash
   ls Dockerfile  # Mac/Linux
   dir Dockerfile # Windows
   ```
2. Riscarica il Dockerfile dal repository GitHub
3. Elimina l'immagine vecchia: `docker rmi colibri-i`
4. Riprova il build

---

### ❌ "hf_download: command not found"

**Causa**: La libreria Hugging Face non è installata correttamente.

**Soluzione**:
```bash
pip install -U huggingface_hub[cli]
# oppure su Linux/Mac:
pip3 install -U huggingface_hub[cli]
```

Poi riprova il comando `hf_download`.

---

### ❌ Il modello non si scarica (timeout o errori di rete)

**Cause**: Connessione lenta o instabile, Hugging Face temporaneamente non disponibile.

**Soluzione**:
1. Attendi e riprova il comando `hf_download`
2. Se continua, scarica manualmente da [qui](https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp)
3. Decomprimi il file ZIP nella cartella desiderata

---

## Note tecniche

### Perché il disco è importante?

Colibrì usa il disco come "RAM aggiuntiva" virtuale (paging). Un disco **veloce** è cruciale per prestazioni decenti.

- **SSD NVMe** (consigliato): 1-15 GB/sec
- **SSD SATA**: 0.5-1 GB/sec
- **Hard disk meccanico**: 0.05-0.1 GB/sec ❌ (troppo lento)

Se il tuo disco è lento, le risposte saranno molto lente anche con molta RAM.

---

### Configurazione di default consigliata per WLS su Windows

Se hai **esattamente 32 GB di RAM** e usi Windows, è molto probabile che WLS di default sia settato per non consumare più di 16 GB di RAM. Bisogna aumentare questo limite [Risoluzione dei problemi](#risoluzione-dei-problemi) . Nel mio caso ho adottato questa configurazione:
```ini
[wsl2]
memory=24GB
processors=12
swap=16GB
```

Ovvero, nel mio caso, ho lasciato 8 GB di RAM e 4 CPU a Windows e dato 24 GB e 12 processori a WSL + Linux.

---

## Domande frequenti

**D: E se ho meno di 32 GB di RAM?**  
R: Probabile che non funzioni bene. Puoi provare se hai 24 GB, ma non è garantito.

**D: Posso aumentare la velocità di risposta?**  
R: Sì, in parte:
- Usa un SSD NVMe veloce
- Aumenta la RAM
- Riduci la complessità delle domande
- Usa `:reset` per cancellare la memoria e alleggerire il carico

**D: Posso usare Colibrì senza Docker?**  
R: Colibrì è nato così, ma questa guida assume Docker. Per compilare da sorgente, vedi il repository GitHub.

**D: Quanta connessione internet mi serve dopo aver scaricato il modello?**  
R: Zero. Colibrì funziona completamente offline.

---

## Supporto e contributi

Se trovi errori o hai suggerimenti per migliorare questa guida, aprici una issue o una pull request sul repository GitHub di Colibrì.

Buon divertimento! 🐦

---

## Test su un PC con poche risorse

Nel primo caso ho fatto una domanda in italiano, nel secondo in giapponese, e nel terzo ho rifatto la domanda in giapponese ma ho richiesto una risposta in italiano. 

```
PS C:\quack\llm\colibri\docker> docker run --rm -it --name colibri-c -v "C:\quack\llm\models\glm-5.2:/app/glm-5.2" -e COLI_MODEL=/app/glm-5.2 colibri-i ./coli chat

     ▄▀▀▀▄  ▄        colibrì v1.0
  ▄▄▄▄▀▀▀▀▄▀▀        tiny engine, immense model
      ▀▀▀▀▀▀▀        GLM-5.2 · 744B MoE · int4 · streaming CPU
        ▀▀▀▀         chat · glm-5.2 · ram -GB · topp off
          ▀
  ──────────────────────────────────────────────────────────
  type and press Enter · Ctrl-C stops the answer · :more continues · :reset clears memory · :q exits

  ╭────────────────────────────────────────────────────────────────────────────────────────────────╮
  │ › Quanti abitanti ha la Cina?                                                                  │
  ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

  ◆ colibrì
  La Cina è attualmente il paese più popoloso al mondo (sebbene, secondo alcune stime recenti, sia stata ormai superata dall'India).

  La popolazione totale della Repubblica Popolare Cinese è di circa 1,41 miliardi di abitanti (dati del 2020-2022 circa).
  └─ 76 tok · 0.04 tok/s · hit 3% · RSS 15.9 GB · 2012s

  ╭────────────────────────────────────────────────────────────────────────────────────────────────╮
  │ › 漫画「ワンピース」の主人公の名前を教えてください。名前だけで、それ以上のコメントはありません。                                              │
  ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

  ◆ colibrì
  ルフィ
  └─ 2 tok · 0.01 tok/s · hit 1% · RSS 16.7 GB · 260s

  ╭────────────────────────────────────────────────────────────────────────────────────────────────╮
  │ › 漫画「ワンピース」の主人公の名前を教えてください。名前だけで、それ以上のコメントはありません。イタリア語で返信                                      │
  ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

  ◆ colibrì
  Il nome del protagonista di One Piece è Monkey D. Luffy.
  └─ 14 tok · 0.02 tok/s · hit 2% · RSS 17.3 GB · 593s

  ╭────────────────────────────────────────────────────────────────────────────────────────────────╮
  │ ›
  ```