# Architettura

> **Note (2026-11-13).** This is the Phase 0/1 design record, written in Italian. It
> describes an HTTP API and an Angular `frontend/` (§2, §7); both were removed when the
> repository was stripped to the ingestion pipeline (branch
> `refactor/strip-to-ingestion-pipeline`), and the sections about them are history, not
> the current layout. The current packages, boundaries and commands are in `CLAUDE.md`.

Architettura di destinazione della piattaforma di investimento quantitativo. Questo
documento è il riferimento su come il sistema deve essere
strutturato mentre cresce dai due pezzi già in produzione — il demone di raccolta dati e la
libreria di ottimizzazione `portopt` — verso una piattaforma con un fondo a quattro agenti
LLM, un'API HTTP e un frontend completo.

Registra **decisioni e il loro razionale**, non implementazione. Non contiene codice: un
blocco di codice è l'unica cosa, in un documento di architettura, che *sembra* autorevole
mentre invecchia senza che nessuno se ne accorga — nessuno lo compila, nessuno lo testa, e
diventa una bugia sul sistema nel momento in cui la firma vera si sposta. Una tabella che
dice quale agente scrive quale campo non marcisce così.

È stato sintetizzato da una passata di ricerca multi-sorgente verificata in contraddittorio;
le affermazioni indagate e **confutate** sono elencate esplicitamente, così da non
costruirle per sbaglio più avanti.

## Come si legge

Le sezioni 4-8 sono una per contesto delimitato, e ciascuna contiene sia il razionale sia la
specifica di quel contesto: chi apre `fund/` trova il perché, l'albero dei file e la tabella
nello stesso posto. Dentro quelle sezioni, tutto ciò che segue il commento
`<!-- GENERATO: … -->` è prodotto da `scriptorium arch` a partire da
`data/architecture-spec/` e si riscrive a ogni assemblaggio: **non va modificato a mano**.
Ciò che sta sopra quel marcatore è scritto da una persona e nessuno strumento lo tocca.

---

## 1. Principi guida

1. **Architettura pulita ed esagonale, con la dipendenza che punta rigorosamente verso
   l'interno.** La logica di dominio di alto livello non dipende mai dall'infrastruttura di
   basso livello. Lo strato di dominio ignora l'esistenza di port, adapter e di qualunque
   cosa gli stia sopra. Le dipendenze fra sorgenti puntano solo verso il centro.

2. **Separazione netta delle responsabilità per contesti delimitati.** Raccolta dati,
   calcolo di ottimizzazione, orchestrazione degli agenti e API sono contesti delimitati
   distinti. Ciascuno è stratificato al proprio interno (dominio → port → adapter) e parla
   con gli altri solo attraverso port espliciti.

3. **Il controllo deterministico avvolge gli LLM, che deterministici non sono.** Ogni azione
   avviata da un modello — a maggior ragione un ordine — attraversa uno strato di validazione
   deterministico e un gate di approvazione umana prima di poter avere effetto sul mondo
   reale. Dove è possibile, le regole si valutano *prima* ancora di consultare il modello.

4. **Riproducibilità e verificabilità si costruiscono, non si danno per scontate.** Nessun
   LLM, locale o di frontiera, offre insieme alta accuratezza e alto determinismo. Costruiamo
   la tracciabilità, il checkpointing e il confine di output tipizzato *attorno* ai modelli,
   invece di confidare che siano riproducibili.

5. **Prima il monolite modulare; si estrae un servizio solo su evidenza dimostrata.** Le
   giunture fra moduli restano pulite, così estrarre un servizio più avanti è un'operazione
   meccanica e non una riscrittura.

---

## 2. Contesti delimitati

Quattro contesti delimitati. Ciascuno è esagonale: un nucleo di dominio puro, i port — le
interfacce che descrivono ciò di cui ha bisogno dal mondo esterno — e gli adapter, cioè le
implementazioni concrete al bordo.

| Contesto delimitato | Cosa possiede | Codice attuale | Dipendenze infrastrutturali |
|---|---|---|---|
| **Raccolta dati** | Fetch schedulato di dati di mercato, fondamentali e macro dentro PostgreSQL | demone `ingestion/` | yfinance, FRED, Trading 212, scraper di news, Postgres |
| **Calcolo di ottimizzazione** | Ottimizzazione di portafoglio (media-rischio, HRP/HERC, risk budgeting, robusta/DR-CVaR, modelli fattoriali, Black-Litterman) | `optimizer/` (`portopt`) | **Nessuna** — calcolo puro, e tale resta |
| **Orchestrazione degli agenti** | Il fondo a quattro agenti: macro, allocation, risk, execution | *nuovo* | Ollama (via BAML), API broker, checkpointer Postgres |
| **API** | La superficie HTTP per il frontend | *nuovo* (era stata rimossa) | Solo lo strato di servizio |

Il contesto di calcolo è già una libreria pura, senza dipendenze da database, HTTP o LLM.
**Deve restare tale**: è l'esempio di riferimento di un nucleo di dominio pulito. Il demone
di raccolta usa già il pattern repository, e quella convenzione si estende all'intera
piattaforma.

### 2.1 I quattro deliverable del monorepo

- **`optimizer/`** — la libreria `portopt`: il dominio puro del calcolo di ottimizzazione.
  Porta con sé lo stack pesante sklearn/skfolio/scipy. Strutturalmente non toccata da questo
  disegno; le uniche aggiunte sono tre sottomoduli di calcolo puro — `regimes/`,
  `attribution/`, `execution/` — che il fondo chiama e che non avevano altrove dove stare
  (§4).
- **`api/` → nome di destinazione `ingestion/`** — il demone di raccolta. Il nome `api/` è
  un residuo di quando aveva un'API HTTP, rimossa su `refactor/strip-to-ingestion-pipeline`:
  è un **worker guidato da schedulazione**, non un servizio HTTP, quindi la convenzione
  monorepo vuole un nome che descriva il dominio (`ingestion/`) e non `api/`. La rinomina è
  stata eseguita nel commit `8f47b07`, che ha toccato Docker, CI, i wrapper di `scheduler/`
  e CLAUDE.md. Resta snello e isolato: **non deve importare `optimizer`** e non porta
  lo stack sklearn (regola dura già in vigore). Scrive solo dati di mercato, macro e fondamentali su PostgreSQL.
- **`fund/`** — **nuovo.** Il fondo ad agenti, lo strato di servizio e l'API HTTP. È
  l'applicazione che porta lo stack pesante: *legge* il database di raccolta, *chiama*
  `portopt` per il calcolo, esegue il grafo a quattro agenti e gestisce gli ordini. Tenerlo
  separato da `ingestion/` è ciò che permette allo scheduler di restare snello.
- **`frontend/`** — **nuovo.** Un client **Angular** dell'API HTTP del fondo. Il nucleo è
  cieco alla sua esistenza. La disposizione segue la convenzione che gli altri frontend
  Angular 21 del workspace già condividono — `pages/`, `services/`, `models/`,
  `interceptors/`, `shared/`, con `app.config.ts` e `app.routes.ts` alla radice di
  `src/app/` — così chi si sposta fra i progetti ritrova la stessa forma. Componenti
  standalone ovunque, nessun NgModule.

`ingestion/` e `fund/` **condividono il database, non il codice**: ciascuno possiede il proprio
strato di accesso in lettura e scrittura, fatto di repository e modelli. È il confine
deliberato fra contesti delimitati — un Postgres condiviso con uno strato di accesso per
contesto, non un ORM condiviso.

```
optimizer/                                  radice del monorepo
│
├── optimizer/                          ┃ CORE · contesto di calcolo (portopt)      §4
├── ingestion/                              DEMONE DI RACCOLTA (snello)             §5
├── fund/                               NEW · FONDO AD AGENTI + SERVIZI + API HTTP  §6
├── frontend/                           NEW · DRIVING · client Angular              §7
│
├── tests/                                  suite di test della libreria            §8
├── scheduler/                              wrapper shell sulla CLI di ingestion
├── scripts/                                helper di CI
├── todo/                                   documentazione di progetto
│       ARCHITECTURE.md   PLAN.md   TASKS.md
│       SPEC-RECONCILIATION.md   DECISIONS.md   VALIDATION.md
├── docker-compose.yml                      db · adminer · scheduler · fund · frontend
├── pyproject.toml                          configurazione di portopt (libreria)
├── CLAUDE.md   README.md
```

L'albero dettagliato di ciascun deliverable sta nella sua sezione, generato da ciò che è
stato effettivamente specificato.

---

## 3. Stratificazione esagonale

**La direzione delle dipendenze è il punto di tutto il diagramma: ogni freccia punta verso
l'interno, verso il nucleo.** Il frontend è l'adapter driving più esterno — un *client*
dell'API. Nulla nel nucleo sa che il frontend esiste; il nucleo girerebbe invariato se il
frontend venisse cancellato e sostituito da una CLI o da un banco di prova. È la disposizione
a port e adapter: gli adapter **driving**, a sinistra, iniziano le chiamate verso il nucleo;
il nucleo chiama verso l'esterno attraverso i port fino agli adapter **driven**, a destra; e
tutte le dipendenze fra sorgenti puntano al centro.

```
       LATO DRIVING                       NUCLEO                      LATO DRIVEN
  (chi chiama verso dentro)     (dominio — non dipende da nulla)   (ciò che il nucleo chiama)
   ───────────────────►  ═══════════════════════════════════►  ───────────────────►
        le dipendenze puntano verso l'interno  ──►        ◄──  verso l'interno

 ┌─────────────────┐            ┌──────────────────────────────┐
 │ FRONTEND        │            │  STRATO DI SERVIZIO          │        ┌──────────────────┐
 │ Angular         │            │  «esegui ottimizzazione»     │──port─►│ repository       │
 │ (un client; il  │──REST────► │  «esegui deliberazione»      │        │ Postgres         │
 │  nucleo ignora  │  +SSE      │  «approva ordine»            │        └──────────────────┘
 │  che esista)    │  via API   │  leggi → aggiorna → persisti │        ┌──────────────────┐
 └─────────────────┘  adapter   └───┬───────────┬───────────┬──┘──port─►│ yfinance / FRED /│
                          │         │           │           │           │ T212 / news      │
 ┌─────────────────┐      │      ┌──▼──────┐ ┌──▼──────┐ ┌──▼─────────┐ └──────────────────┘
 │ ADAPTER API     │      │      │DOMINIO: │ │DOMINIO: │ │ ORCHESTRA- │  ┌──────────────────┐
 │ (sottile,       │──────┘      │ portopt │ │raccolta │ │ ZIONE      │◄─┤ Ollama (via BAML)│
 │  FastAPI)       │             │ (puro)  │ │(regole) │ │ AGENTI     │  │ I/O LLM tipizzato│
 │ parse · authz · │             │         │ │         │ │ (grafo)    │  └──────────────────┘
 │ JSON · SSE ·    │             │media-   │ │         │ │            │  ┌──────────────────┐
 │ NESSUNA logica  │             │rischio  │ │         │ │ macro →    │  │ API broker       │
 └─────────────────┘             │HRP/HERC │ │         │ │ alloc →    │──┤ (ordini reali)   │
                                 │BL/CVaR  │ │         │ │ risk       │  │ OPZIONALE (§9)   │
 ┌─────────────────┐             │fattori  │ │         │ │   ↓ interrupt └──────────────────┘
 │ SCHEDULER       │──invoca───► │         │ │         │ │ GUARDRAIL  │  ┌──────────────────┐
 │ (APScheduler;   │  use-case   │ NO DB   │ │         │ │ (regole,   │──┤ checkpointer     │
 │  guida la       │             │ NO HTTP │ │         │ │  kill-sw)  │  │ (PostgresSaver)  │
 │  raccolta)      │             │ NO LLM  │ │         │ │   ↓        │  └──────────────────┘
 └─────────────────┘             └─────────┘ └─────────┘ │ GATE UMANO │
                                                         │   ↓ esecuz.│
                                                         └────────────┘
```

Il grafo dei quattro agenti sta *dentro* il nucleo, a destra. Raggiunge il mondo esterno —
Ollama, il broker, il checkpointer — solo attraverso i **port**: le tre scatole sul lato
driven sono adapter che implementano quei port. Gli agenti non importano mai un client HTTP
né il frontend; chiamano use-case e port, nient'altro.

**Responsabilità, strato per strato:**

- **Adapter driving** (frontend, adapter API, scheduler) — chi chiama *dentro* il nucleo. Il
  frontend è un client dell'API; l'adapter API si occupa solo di questioni web (parsing,
  autorizzazione, JSON, SSE, errore→stato) senza logica di business; lo scheduler guida la
  raccolta. Nessuno dei tre contiene logica di dominio, e il nucleo non dipende da nessuno di
  loro.
- **Strato di servizio / use-case** — orchestra: legge da un repository, aggiorna il modello
  di dominio, persiste. Dipende da *astrazioni* di repository, quindi gira su finti nei test
  e su adapter veri in produzione. È l'unico ingresso al nucleo: ogni adapter driving passa
  di qui.
- **Dominio** — le regole di business. Puro. Nessun import di framework, database o rete.
- **Port** — le interfacce che dominio e servizi dichiarano: un port broker, un port
  repository, un port LLM.
- **Adapter driven** — le implementazioni concrete dei port al bordo del sistema: Postgres,
  client delle sorgenti dati, Ollama/BAML, API broker, checkpointer. Il nucleo li chiama;
  loro non chiamano mai il nucleo.

---

## 4. `optimizer/` — il nucleo di calcolo

È il contesto senza dipendenze infrastrutturali: niente database, niente HTTP, niente LLM.
Prende in ingresso rendimenti e vincoli, restituisce pesi. È anche il contesto che realizza
la parte più densa della teoria — stima dei momenti, costruzione del rischio, frontiera,
budget di rischio, backtest — ed è quello dove un errore non si vede a occhio.

Questa sezione **registra un impegno che la libreria ha già preso**, invece di sciogliere una
scelta aperta: è letta da `portopt` e dai suoi metadati di dipendenza, non prodotta dalla
passata di ricerca. Sta scritta perché §9 decideva tutto *tranne* come la matematica viene
effettivamente risolta.

### 4.1 Ogni ottimizzazione è un programma conico convesso

**Tutte le ottimizzazioni che la piattaforma esegue sono programmi conici convessi, e un solo
solver conico li copre tutti.** Le classi trattabili sono annidate — LP ⊂ QP ⊂ QCQP ⊂ SOCP —
quindi un algoritmo per SOCP risolve l'intera gerarchia; ed è la convessità a garantire che il
punto restituito sia un ottimo *globale* con un certificato di ottimalità verificabile, non un
ottimo locale.

Concretamente, attraverso gli obiettivi che la piattaforma offre:

| Obiettivo | Classe | Come ci arriva |
|---|---|---|
| Media-varianza | QP | forma quadratica diretta |
| Shortfall risk, controparte robusta | SOCP | il termine di norma solleva il QP |
| CVaR su scenari | LP | una variabile di eccesso per scenario, per *qualunque* distribuzione dei rendimenti, non solo la normale |
| Famiglia drawdown (MaxDD, AvDD, CDaR) | LP | variabili di massimo corrente |
| Risk budgeting | cono esponenziale | minimizzazione di varianza sotto vincolo logaritmico sui pesi |

Una sola famiglia di solver, un solo strato di costruzione del problema, nessuna numerica
artigianale per obiettivo.

### 4.2 Il solver non lo sceglie la piattaforma — lo ha già scelto `portopt`

La catena di dipendenze è fissa e a senso unico: il fondo chiama `portopt`, `portopt`
configura `skfolio`, `skfolio` costruisce il problema **CVXPY** e lo passa a **CLARABEL**, un
backend conico a punti interni che copre LP, QP, QCQP, SOCP e il cono esponenziale. CLARABEL è
il default dichiarato su ogni configurazione di ottimizzazione di `portopt` oggi, e nessuno dei
due pacchetti è una dipendenza nuova: arrivano entrambi in modo transitivo con il pin di
`skfolio` che la libreria porta già.

CVXPY è lo strato di modellazione, quindi un problema si esprime una volta sola e in modo
indipendente dal solver; il solver stesso è **un nome più un dizionario di parametri su un
oggetto di configurazione**, il che rende la sostituzione una questione di configurazione e non
di codice. È quella proprietà che vale la pena proteggere.

**Questo non aggiunge infrastruttura.** CVXPY e CLARABEL sono calcolo puro in-process. Il
"dipendenze infrastrutturali: **nessuna**" di §2 per questo contesto resta valido parola per
parola.

### 4.3 Un solve è un pass/fail deterministico, come ogni altro gate

Lanciare un'ottimizzazione è uno strumento raggiungibile da un LLM (§6.3) ed è sempre guidato
da codice, quindi il suo esito deve avere lo stesso taglio netto del guardrail: **lo stato
terminale del solver *è* il verdetto**. Uno stato non ottimale — infeasible, unbounded, bassa
accuratezza, tempo scaduto — è un **fallimento registrato**, mai un vettore di pesi restituito
come se avesse converguto. `portopt` ha già un tipo di eccezione dedicato al fallimento del
solver, quindi questa è una disciplina sull'*usarlo*, non nuova macchineria.

È la validazione classica basata su regole che §11 prescrive, applicata allo stato che il
solver stesso riporta; non è affatto il gate di verifica formale che §11 confuta.

Poiché la feasibility diventa NP-completa non appena entrano vincoli interi (§4.5), ogni solve
raggiungibile da un passo di agente ha bisogno anche di un **budget di tempo di parete**, con
il timeout trattato come uno stato di fallimento in più. Il valore del budget non si fissa qui:
non esiste ancora una misura con cui fissarlo.

### 4.4 Le metaeuristiche non sono adottate

La via standard per uscire dai casi non convessi è una riformulazione a penalità minimizzata da
una metaeuristica a popolazione (PSO, algoritmi genetici). Non la prendiamo.

Un metodo del genere converge all'ottimo globale solo *in probabilità* e — questo è il punto
decisivo — soddisfa i vincoli solo *approssimativamente*: negli esperimenti pubblicati su
portafogli il vincolo di budget teneva esattamente, ma il vincolo di rendimento obiettivo
teneva in una quota compresa fra lo 0% e il 90% delle esecuzioni a seconda dello scenario, e le
variabili binarie di selezione atterravano leggermente fuori da {0,1}.

Un'allocazione il cui budget o la cui cardinalità sono soltanto *quasi* soddisfatti non può
passare un guardrail pass/fail e non è eseguibile così com'è. Va contro il principio di
controllo deterministico su cui è costruito tutto il documento. Prima le formulazioni esatte;
una metaeuristica resta un ripiego documentato per una classe di vincoli senza forma esatta
trattabile, mai un percorso di default.

### 4.5 Vincoli interi misti — un secondo solver, rimandato

Cardinalità ("tieni al massimo K titoli"), soglie e lotti minimi sono esattamente il punto in
cui il problema esce dalla classe convessa. Con un vincolo di cardinalità presente, decidere la
sola *feasibility* è già NP-completo con appena tre vincoli lineari — la difficoltà viene
dall'integralità, non dal numero di asset — e la frontiera efficiente diventa discontinua,
quindi non è più tracciabile come una curva sola.

`skfolio` esprime questi vincoli con variabili booleane, il che richiede un solver
**mixed-integer**; CLARABEL non lo è, e `skfolio` solleva un errore invece di degradare in
silenzio. La sua raccomandazione open-source è **SCIP**, installato come extra opzionale di
CVXPY — oggi è solo un extra dev/docs di `skfolio`, non una dipendenza di `portopt`.

Quindi `portopt` espone già la superficie del vincolo (un campo `cardinality` sulla
configurazione mean-risk) senza portare un solver in grado di onorarlo. La decisione è
volutamente stretta: **non aggiungere un solver MIP finché un percorso con vincolo di
cardinalità non entra davvero in produzione**, e quando lo farà, aggiungerlo come extra
opzionale esplicito e non come installazione di default — lo stack pesante resta opt-in.
L'NP-completezza qui sopra significa anche che un percorso MIP ha bisogno del budget di tempo e
dello stato di fallimento *prima* di aver bisogno di un solver più veloce.

### 4.6 I tre sottomoduli nuovi

Strutturalmente questo disegno non tocca `optimizer/`: le uniche aggiunte sono tre sottomoduli
di puro calcolo che il fondo chiama e che non ha nessun altro posto dove mettere. Ciascuno
risponde a una richiesta che l'albero già fa senza avere un modulo che la serva.

**`regimes/`** — lo stimatore unico dello stato di mercato: il regime latente e il suo filtro,
il nowcast macro e gli indici osservabili (turbolenza, absorption ratio, rango effettivo). Lo
consumano l'agente macro, le pagine di rischio e la selezione trasversale, che altrimenti ne
porterebbero ciascuno una copia propria. *Teoria: capitoli 21, 26, 28.*

**`attribution/`** — scompone il rischio di un portafoglio *già allocato* in contributi
additivi. È l'inverso del risk budgeting che `optimization/` già costruisce, e produce i numeri
che §6.3 pretende che l'agente di rischio riceva invece di calcolare. *Teoria: capitolo 23.*

**`execution/`** — trasforma una lista di operazioni in un piano temporale e ne prezza l'impatto
di mercato. È ciò che rende "dimensionamento, scheduling e impatto atteso sono consigli" di §6
una consegna vera e non una promessa senza codice dietro. *Teoria: capitolo 22.*

Tutti e tre sono puro calcolo — niente DB, niente HTTP, niente LLM — come tutto il resto sotto
`optimizer/`, e tutti e tre si raggiungono attraverso il registro strumenti fisso già esistente
(§6.3), non con port nuove.

<!-- GENERATO: optimizer -->

_Da qui in giù la sezione è generata da `scriptorium arch` il 2026-08-19: non modificarla a mano, si riscrive a ogni assemblaggio. Il record completo — responsabilità estese e citazioni riga per riga — sta in `data/architecture-spec/`, da cui questo è renderizzato._

### 4.1 Albero

121 file, ciascuno annotato col suo ruolo esagonale. Ruoli: config 1 · CORE 120

```
└── optimizer/
    ├── __init__.py  ┃ CORE
    ├── attribution/
    │   ├── __init__.py  ┃ CORE · planned
    │   ├── _config.py  ┃ CORE · planned
    │   ├── _diagnostics.py  ┃ CORE · planned
    │   ├── _euler.py  ┃ CORE · planned
    │   ├── _expected_loss.py  ┃ CORE · planned
    │   ├── _factor_impact.py  ┃ CORE · planned
    │   └── _sample.py  ┃ CORE · planned
    ├── cluster/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ config
    │   └── _factory.py  ┃ CORE
    ├── distance/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   └── _factory.py  ┃ CORE
    ├── execution/
    │   ├── __init__.py  ┃ CORE · planned
    │   ├── _aim_portfolio.py  ┃ CORE · planned
    │   ├── _config.py  ┃ CORE · planned
    │   ├── _cost.py  ┃ CORE · planned
    │   ├── _kyle_impact.py  ┃ CORE · planned
    │   ├── _power_law_schedule.py  ┃ CORE · planned
    │   ├── _resilience.py  ┃ CORE · planned
    │   ├── _trajectory.py  ┃ CORE · planned
    │   └── _turnover.py  ┃ CORE · planned
    ├── factors/
    │   ├── __init__.py  ┃ CORE
    │   ├── _alpha.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   ├── _construction.py  ┃ CORE
    │   ├── _diagnostics.py  ┃ CORE
    │   ├── _integration.py  ┃ CORE
    │   ├── _mimicking.py  ┃ CORE
    │   ├── _ml_scoring.py  ┃ CORE
    │   ├── _oos_validation.py  ┃ CORE
    │   ├── _regime.py  ┃ CORE
    │   ├── _scoring.py  ┃ CORE
    │   ├── _sector_bands.py  ┃ CORE
    │   ├── _selection.py  ┃ CORE
    │   ├── _standardization.py  ┃ CORE
    │   └── _validation.py  ┃ CORE
    ├── fx/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   ├── _converter.py  ┃ CORE
    │   ├── _decomposition.py  ┃ CORE
    │   ├── _factory.py  ┃ CORE
    │   └── _rates.py  ┃ CORE
    ├── linear_model/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   └── _factory.py  ┃ CORE
    ├── moments/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   ├── _factory.py  ┃ CORE
    │   └── _scaling.py  ┃ CORE
    ├── online/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   └── _factory.py  ┃ CORE
    ├── optimization/
    │   ├── __init__.py  ┃ CORE
    │   ├── _benchmark_tracker.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   ├── _dr_cvar.py  ┃ CORE
    │   ├── _factory.py  ┃ CORE
    │   ├── _herc.py  ┃ CORE
    │   ├── _hierarchical_common.py  ┃ CORE
    │   ├── _hrp.py  ┃ CORE
    │   ├── _max_diversification.py  ┃ CORE
    │   ├── _naive.py  ┃ CORE
    │   ├── _nco.py  ┃ CORE
    │   ├── _regime_blended_mean_risk.py  ┃ CORE
    │   ├── _region_constraints.py  ┃ CORE
    │   ├── _risk_budgeting.py  ┃ CORE
    │   ├── _robust.py  ┃ CORE
    │   ├── _schur.py  ┃ CORE
    │   └── _stacking.py  ┃ CORE
    ├── pipeline/
    │   ├── __init__.py  ┃ CORE
    │   ├── _builder.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   └── _orchestrator.py  ┃ CORE
    ├── pre_selection/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   └── _pipeline.py  ┃ CORE
    ├── preprocessing/
    │   ├── __init__.py  ┃ CORE
    │   ├── _cs_transformers.py  ┃ CORE
    │   ├── _delisting.py  ┃ CORE
    │   ├── _imputation.py  ┃ CORE
    │   ├── _outliers.py  ┃ CORE
    │   ├── _regression_imputer.py  ┃ CORE
    │   └── _validation.py  ┃ CORE
    ├── rebalancing/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   └── _rebalancer.py  ┃ CORE
    ├── regimes/
    │   ├── __init__.py  ┃ CORE · planned
    │   ├── _absorption.py  ┃ CORE · planned
    │   ├── _config.py  ┃ CORE · planned
    │   ├── _cross_sectional.py  ┃ CORE · planned
    │   ├── _factory.py  ┃ CORE · planned
    │   ├── _markov_switching.py  ┃ CORE · planned
    │   ├── _nowcast.py  ┃ CORE · planned
    │   └── _turbulence.py  ┃ CORE · planned
    ├── scoring/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   └── _factory.py  ┃ CORE
    ├── synthetic/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   └── _factory.py  ┃ CORE
    ├── tuning/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   └── _factory.py  ┃ CORE
    ├── uncertainty_set/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   └── _factory.py  ┃ CORE
    ├── universe/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   ├── _factory.py  ┃ CORE
    │   └── _screener.py  ┃ CORE
    ├── validation/
    │   ├── __init__.py  ┃ CORE
    │   ├── _config.py  ┃ CORE
    │   ├── _factory.py  ┃ CORE
    │   ├── _forecast.py  ┃ CORE
    │   └── _regime.py  ┃ CORE
    └── views/
        ├── __init__.py  ┃ CORE
        ├── _config.py  ┃ CORE
        ├── _factory.py  ┃ CORE
        └── _uncertainty.py  ┃ CORE
```

### 4.2 I file

The optimizer/ (portopt) bounded context is, by ARCHITECTURE.md §2/§4, the reference example of a pure hexagonal domain core: "no DB, HTTP, or LLM dependencies... it must stay that way" (lines 49-56, 154). Concretely this means every file in this context is layer=core — there are no ports/ or driven/ subpackages here at all, because the context has zero infrastructure dependencies to abstract behind a port. All twenty submodules are pure numerical/statistical Python built on numpy/pandas/scipy/sklearn/skfolio; every "connects_to" edge listed above is an intra-context import between two core files (e.g. optimization/ imports moments/, distance/, cluster/, uncertainty_set/, factors/ to assemble an estimator; pipeline/ imports pre_selection/, optimization/, factors/, fx/, universe/, validation/, tuning/, scoring/, rebalancing/ to compose the end-to-end orchestration). None of these edges cross into DB/HTTP/LLM territory, so the "core never connects_to driven directly" rule is trivially satisfied: there is no driven/ in this context to violate it with. The only outward-facing seam is the one the rest of the platform must respect, not this context: fund/app/services/optimization/ (a different bounded context) is the sole driving-side caller that imports optimizer.* as a library dependency to build portfolios (ARCHITECTURE.md §4, fund/app/services/optimization/ "run_optimization (wraps portopt)"), and optimizer/ itself never imports back into fund/, ingestion/, or frontend/.

| File | Ruolo | Contiene | Si collega a |
|---|---|---|---|
| `optimizer/__init__.py` | CORE | Package root: documents every submodule's role in a single module docstring, re-exports the shared exception hierarchy (OptimizerError, ConfigurationError, DataError, ConvergenceError, ValidationError, OptimizationError), attaches a NullHandler to the 'optimizer' logger, and resolves __version__ from installed package metadata ('portopt'), falling back to '0.0.0.dev0' when not installed. | `optimizer/preprocessing/`, `optimizer/moments/`, `optimizer/views/`, `optimizer/optimization/`, `optimizer/pre_selection/`, `optimizer/factors/`, `optimizer/synthetic/`, `optimizer/scoring/`, `optimizer/distance/`, `optimizer/cluster/`, `optimizer/uncertainty_set/`, `optimizer/linear_model/`, `optimizer/online/`, `optimizer/fx/`, `optimizer/universe/`, `optimizer/validation/`, `optimizer/tuning/`, `optimizer/rebalancing/`, `optimizer/pipeline/` |
| `optimizer/attribution/__init__.py` | CORE ·  pianificato | Punto di ingresso pubblico del package attribution: raccoglie e ri-esporta il vocabolario di configurazione, il motore di Eulero e le funzioni di stima/diagnostica definiti nei sei moduli sorgente sottostanti, seguendo la stessa convenzione a barrel-module già usata da optimizer/optimization/ e optimizer/factors/. | `optimizer/attribution/_config.py`, `optimizer/attribution/_euler.py`, `optimizer/attribution/_sample.py`, `optimizer/attribution/_diagnostics.py`, `optimizer/attribution/_factor_impact.py`, `optimizer/attribution/_expected_loss.py` |
| `optimizer/attribution/_config.py` | CORE ·  pianificato | Vocabolario condiviso del package: quale famiglia di misura di rischio coerente il chiamante vuole attribuire (deviazione standard, Value-at-Risk, Expected Shortfall), a quale livello di confidenza, e con quale via di stima — forma chiusa oppure campionaria/kernel. | `optimizer/exceptions.py` |
| `optimizer/attribution/_diagnostics.py` | CORE ·  pianificato | Diagnostica di concentrazione e diversificazione calcolata sopra un insieme di contributi di Eulero già prodotti, per l'identificazione delle concentrazioni di rischio e per alimentare i vincoli sui contributi che il guardrail verifica. | `optimizer/attribution/_config.py`, `optimizer/attribution/_euler.py`, `external:numpy` |
| `optimizer/attribution/_euler.py` | CORE ·  pianificato | Motore centrale del package: dalle esposizioni già allocate di un portafoglio (pesi) e dai suoi input di rischio (tipicamente la matrice di covarianza) deriva i contributi di Eulero per componente — l'unica decomposizione RORAC-compatibile e a somma piena del rischio totale. | `optimizer/attribution/_config.py`, `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `external:numpy` |
| `optimizer/attribution/_expected_loss.py` | CORE ·  pianificato | Riconciliazione dei contributi di rischio prodotti dal resto del package con la loro interpretazione finanziaria come contributi attesi a una perdita effettiva, così che il racconto dell'agente di rischio possa affermare cosa un contributo significa, non solo riportarne il numero. | `optimizer/attribution/_config.py`, `optimizer/attribution/_euler.py`, `external:numpy` |
| `optimizer/attribution/_factor_impact.py` | CORE ·  pianificato | Decomposizione del rischio di portafoglio nella parte spiegata da un insieme scelto di fattori sistematici e nel residuo non correlato, tramite allocazione di Eulero applicata alla decomposizione della perdita in attesa condizionata ai fattori — una generalizzazione del coefficiente di determinazione per misure di rischio coerenti. | `optimizer/attribution/_config.py`, `optimizer/attribution/_euler.py`, `optimizer/attribution/_sample.py`, `optimizer/factors/_config.py`, `external:numpy` |
| `optimizer/attribution/_sample.py` | CORE ·  pianificato | Stima dei contributi di Eulero da un pannello di rendimenti o perdite realizzato o simulato, quando la distribuzione non è nota in forma analitica — la via di stima che il rischio di portafoglio richiede nella pratica, poiché le perdite non sono mai date in forma chiusa. | `optimizer/attribution/_config.py`, `optimizer/attribution/_euler.py`, `external:numpy`, `external:scipy` |
| `optimizer/cluster/__init__.py` | CORE | Superficie pubblica del sottopacchetto di hierarchical clustering: il docstring di modulo dichiara che avvolge skfolio.cluster.HierarchicalClustering dietro un config tipizzato, a uso degli ottimizzatori HRP/HERC/NCO/Schur, che ricevono in input uno stimatore di clustering già costruito. | `optimizer/cluster/_config.py`, `optimizer/cluster/_factory.py` |
| `optimizer/cluster/_config.py` | config | Vocabolario tipizzato e configurazione immutabile per la selezione del clustering gerarchico. | `optimizer/exceptions.py` |
| `optimizer/cluster/_factory.py` | CORE | Punto di traduzione dalla configurazione dichiarativa di _config.py all'oggetto skfolio HierarchicalClustering concreto, pronto per il fit. | `optimizer/cluster/_config.py`, `external:skfolio` |
| `optimizer/distance/__init__.py` | CORE | Facciata pubblica del sottopacchetto distance: fissa nel docstring il contratto comune a ogni stimatore di distanza selezionabile e ne dichiara la superficie pubblica. | `optimizer/distance/_config.py`, `optimizer/distance/_factory.py` |
| `optimizer/distance/_config.py` | CORE | Vocabolario tipizzato e configurazione immutabile per la selezione dello stimatore di distanza/dipendenza fra serie di rendimenti. | `optimizer/exceptions.py` |
| `optimizer/distance/_factory.py` | CORE | Punto di traduzione dalla configurazione dichiarativa DistanceConfig all'oggetto skfolio concreto, pronto per il fit. | `optimizer/distance/_config.py`, `external:skfolio` |
| `optimizer/execution/__init__.py` | CORE ·  pianificato | Superficie pubblica del pacchetto: raccoglie in un unico namespace la configurazione e le funzioni di calcolo definite negli altri otto file, seguendo la stessa convenzione a barrel-module di optimizer/attribution/ e optimizer/moments/. | `optimizer/execution/_config.py`, `optimizer/execution/_aim_portfolio.py`, `optimizer/execution/_turnover.py`, `optimizer/execution/_kyle_impact.py`, `optimizer/execution/_trajectory.py`, `optimizer/execution/_resilience.py`, `optimizer/execution/_power_law_schedule.py`, `optimizer/execution/_cost.py` |
| `optimizer/execution/_aim_portfolio.py` | CORE ·  pianificato | Implementa il bersaglio dinamico dell'esecuzione: non il portafoglio ottimo statico, ma l'aim portfolio di Gârleanu-Pedersen che pesa ciascun segnale per la propria velocità di ritorno alla media, più la regola di trade parziale verso di esso. | `optimizer/execution/_config.py`, `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `external:numpy` |
| `optimizer/execution/_config.py` | CORE ·  pianificato | Vocabolario tipizzato e configurazione immutabile per la scelta del modello di scheduling/impatto e per i suoi parametri, condivisa da tutti gli altri file del pacchetto. | `optimizer/exceptions.py` |
| `optimizer/execution/_cost.py` | CORE ·  pianificato | Aggrega la traiettoria scelta (lineare, a resilienza o a legge di potenza) e il coefficiente di impatto permanente di Kyle in un'unica stima di costo atteso — il numero che chiude la catena e che l'agente di esecuzione impacchetta in proposed_orders. | `optimizer/execution/_config.py`, `optimizer/execution/_kyle_impact.py`, `optimizer/execution/_trajectory.py`, `optimizer/execution/_resilience.py`, `optimizer/execution/_power_law_schedule.py`, `external:numpy` |
| `optimizer/execution/_kyle_impact.py` | CORE ·  pianificato | Microfonda il coefficiente di impatto permanente λ nel modello ad aste di Kyle, fornendone la stima/calibrazione che alimenta ogni altro file del pacchetto come termine di impatto permanente. | `optimizer/execution/_config.py`, `external:numpy` |
| `optimizer/execution/_power_law_schedule.py` | CORE ·  pianificato | Generalizza lo scheduling a un impatto temporaneo non lineare a legge di potenza di esponente arbitrario k, la forma motivata dalla concavità empirica documentata da Bouchaud-Farmer-Lillo. | `optimizer/execution/_config.py`, `optimizer/execution/_trajectory.py`, `external:numpy`, `external:scipy` |
| `optimizer/execution/_resilience.py` | CORE ·  pianificato | Sostituisce la dicotomia impatto-permanente/impatto-istantaneo con la dinamica di ricostituzione dell'order book di Obizhaeva-Wang, producendo lo schema di negoziazione a blocco-flusso-blocco che ne segue. | `optimizer/execution/_config.py`, `optimizer/execution/_kyle_impact.py`, `external:numpy` |
| `optimizer/execution/_trajectory.py` | CORE ·  pianificato | Risolve la traiettoria di esecuzione a impatto lineare: la frontiera efficiente media-varianza di Almgren-Chriss (mono-titolo) e il controllo ottimo dinamico di Bertsimas-Lo, entrambi estesi al caso multi-titolo con impatto incrociato — il motore di scheduling portfolio-level del pacchetto. | `optimizer/execution/_config.py`, `optimizer/execution/_kyle_impact.py`, `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `external:numpy`, `external:scipy` |
| `optimizer/execution/_turnover.py` | CORE ·  pianificato | Stima ex-ante, dal solo mix di segnali, il turnover che l'esecuzione dovrà sostenere — una previsione, non una misura a posteriori. | `optimizer/execution/_config.py`, `external:numpy` |
| `optimizer/factors/__init__.py` | CORE | Modulo radice del pacchetto factors/: fissa nel docstring il perimetro del pacchetto (costruzione, scoring e selezione dei titoli per la pre-selezione) e ne definisce la superficie pubblica. | `optimizer/factors/_alpha.py`, `optimizer/factors/_config.py`, `optimizer/factors/_construction.py`, `optimizer/factors/_diagnostics.py`, `optimizer/factors/_integration.py`, `optimizer/factors/_mimicking.py`, `optimizer/factors/_ml_scoring.py`, `optimizer/factors/_oos_validation.py`, `optimizer/factors/_regime.py`, `optimizer/factors/_scoring.py`, `optimizer/factors/_sector_bands.py`, `optimizer/factors/_selection.py`, `optimizer/factors/_standardization.py`, `optimizer/factors/_validation.py` |
| `optimizer/factors/_alpha.py` | CORE | Ponte tra punteggi fattoriali standardizzati e rendimenti attesi, via un modello lineare CAPM-più-tilt-fattoriale. | — |
| `optimizer/factors/_config.py` | CORE | Vocabolario tipizzato e configurazione centrale per l'intera pipeline dei fattori — la tassonomia e i dataclass di configurazione che gli altri tredici moduli importano. | `optimizer/factors/_sector_bands.py`, `optimizer/exceptions.py` |
| `optimizer/factors/_construction.py` | CORE | Calcola i valori grezzi, corretti per la disponibilità point-in-time, di ciascun fattore per ogni titolo, a partire da fondamentali, prezzi, volumi, dati analista e insider. | `optimizer/factors/_config.py` |
| `optimizer/factors/_diagnostics.py` | CORE | Diagnostica di multicollinearità e qualità dei dati sulla matrice di punteggi fattoriali cross-sezionale. | `optimizer/factors/_validation.py`, `optimizer/exceptions.py` |
| `optimizer/factors/_integration.py` | CORE | Ponte tra punteggi fattoriali compositi/standardizzati e gli oggetti che lo strato di ottimizzazione di portopt consuma davvero. | `optimizer/factors/_config.py`, `optimizer/rebalancing/_rebalancer.py`, `optimizer/views/_config.py`, `optimizer/views/_factory.py`, `optimizer/exceptions.py` |
| `optimizer/factors/_mimicking.py` | CORE | Costruisce serie di rendimento di portafogli fattoriali long-short (factor-mimicking) e la diagnostica di spread per quintile usata per prezzare e validare i singoli fattori. | `optimizer/exceptions.py` |
| `optimizer/factors/_ml_scoring.py` | CORE | Scoring composito basato su apprendimento supervisionato: regressione ridge o alberi con gradient boosting, alternativa alla ponderazione a peso fisso. | — |
| `optimizer/factors/_oos_validation.py` | CORE | Validazione fuori campione a blocchi rolling o combinatoriale-purgata del potere predittivo dei fattori, a guardia dell'overfitting sullo 'zoo dei fattori'. | `optimizer/factors/_validation.py`, `optimizer/validation/_config.py` |
| `optimizer/factors/_regime.py` | CORE | Classifica il regime macroeconomico prevalente da indicatori osservabili e ne deriva i tilt moltiplicativi regime-condizionati sui pesi di gruppo fattoriale. | `optimizer/factors/_config.py` |
| `optimizer/factors/_scoring.py` | CORE | Combina i punteggi fattoriali standardizzati e raggruppati in un singolo punteggio composito per titolo, con una scelta tra più schemi di ponderazione. | `optimizer/factors/_config.py`, `optimizer/factors/_ml_scoring.py`, `optimizer/factors/_validation.py`, `optimizer/exceptions.py` |
| `optimizer/factors/_sector_bands.py` | CORE | Matrice statica regime → settore → (pavimento, tetto) usata per vincolare il peso settoriale nell'ottimizzatore in funzione del regime macro classificato. | `optimizer/factors/_config.py` |
| `optimizer/factors/_selection.py` | CORE | Trasforma un universo scorato in una lista effettiva di titoli detenuti, con controllo del turnover tramite buffer/isteresi e bilanciamento settoriale. | `optimizer/factors/_config.py` |
| `optimizer/factors/_standardization.py` | CORE | Pipeline di standardizzazione cross-sezionale che trasforma valori fattoriali grezzi, su scale diverse, in punteggi comparabili, coerenti in segno e opzionalmente sector-neutral. | `optimizer/factors/_config.py`, `optimizer/exceptions.py` |
| `optimizer/factors/_validation.py` | CORE | Validazione statistica del potere predittivo dei fattori individuali e del composito — coefficienti di informazione, loro significatività, e correzione per test multipli. | `optimizer/factors/_config.py`, `optimizer/exceptions.py`, `optimizer/linear_model/` |
| `optimizer/fx/__init__.py` | CORE | Punto di ingresso pubblico del package fx: raccoglie ed espone le classi, gli enum e le funzioni definiti nei cinque moduli sorgente sottostanti per la conversione multi-valuta e la decomposizione dei rendimenti. | `optimizer/fx/_config.py`, `optimizer/fx/_converter.py`, `optimizer/fx/_decomposition.py`, `optimizer/fx/_factory.py`, `optimizer/fx/_rates.py` |
| `optimizer/fx/_config.py` | CORE | Configurazione immutabile della conversione multi-valuta: tre enum di scelte chiuse (modalità di conversione, valuta base supportata, fonte dei tassi) e il dataclass congelato FxConfig che le compone insieme ai parametri operativi di allineamento e tolleranza. | — |
| `optimizer/fx/_converter.py` | CORE | Trasformatore sklearn-compatibile (FxPriceConverter) che converte una matrice di prezzi in valute locali verso un'unica valuta base, da inserire nella pipeline prima del calcolo dei rendimenti. | `optimizer/exceptions.py`, `optimizer/fx/_rates.py` |
| `optimizer/fx/_decomposition.py` | CORE | Scomposizione del rendimento totale di un asset in valuta estera nelle sue componenti azionaria, valutaria e di interazione, tramite il dataclass FxReturnDecomposition e la funzione decompose_fx_returns. | — |
| `optimizer/fx/_factory.py` | CORE | Funzione di fabbrica che costruisce un FxPriceConverter pronto all'uso a partire da un FxConfig, dai tassi di cambio pre-caricati e dalla mappa valutaria. | `optimizer/fx/_config.py`, `optimizer/fx/_converter.py` |
| `optimizer/fx/_rates.py` | CORE | Funzioni pure di costruzione e allineamento dei tassi di cambio: costruzione del ticker della coppia valutaria, calcolo del cross-rate via USD, allineamento dei tassi al calendario dei prezzi e determinazione delle valute estere richieste da un universo. | `optimizer/exceptions.py` |
| `optimizer/linear_model/__init__.py` | CORE | Superficie pubblica del sottopacchetto linear_model: nel docstring di modulo fissa il contratto d'ingresso della regressione lineare cross-sezionale — forme X:(T,N,K), y:(T,N), sample_weight:(T,N) opzionale, con le coppie a peso zero escluse e ammesse a contenere NaN. | `optimizer/linear_model/_config.py`, `optimizer/linear_model/_factory.py` |
| `optimizer/linear_model/_config.py` | CORE | Configurazione immutabile per uno stimatore di regressione lineare cross-sezionale — quali parametri vengono inoltrati a skfolio.linear_model.CSLinearRegression e quali restano indizi ad uso del solo chiamante. | `optimizer/exceptions.py` |
| `optimizer/linear_model/_factory.py` | CORE | Punto di traduzione dalla configurazione dichiarativa di _config.py allo stimatore skfolio CSLinearRegression concreto, pronto per il fit. | `optimizer/linear_model/_config.py`, `external:skfolio` |
| `optimizer/moments/__init__.py` | CORE | Modulo radice del pacchetto: fissa nel docstring il contratto d'uso di tutto ciò che il pacchetto espone e ne definisce la superficie pubblica. | `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `optimizer/moments/_scaling.py` |
| `optimizer/moments/_config.py` | CORE | Vocabolario tipizzato e configurazione immutabile per la stima dei momenti e la costruzione del prior. | — |
| `optimizer/moments/_factory.py` | CORE | Punto di traduzione dalla configurazione dichiarativa di _config.py agli oggetti skfolio concreti, pronti per il fit. | `optimizer/moments/_config.py`, `optimizer/exceptions.py`, `external:skfolio` |
| `optimizer/moments/_scaling.py` | CORE | Riscala media e covarianza di rendimenti log-giornalieri a un orizzonte d'investimento multi-periodo sotto l'ipotesi di rendimenti log-normali. | `optimizer/exceptions.py` |
| `optimizer/online/__init__.py` | CORE | Punto di ingresso pubblico del sottopacchetto di apprendimento incrementale ("online") costruito attorno a `partial_fit` di skfolio. | `optimizer/optimization/__init__.py`, `optimizer/online/_config.py`, `optimizer/online/_factory.py` |
| `optimizer/online/_config.py` | CORE | Configurazione immutabile dei tre flussi di apprendimento incrementale che il pacchetto offre: una singola esecuzione online_predict/online_score, un OnlineGridSearch e un OnlineRandomizedSearch. | `optimizer/tuning/_config.py` |
| `optimizer/online/_factory.py` | CORE | Funzioni di fabbrica ed esecuzione che adattano gli oggetti di configurazione di optimizer alle API di apprendimento incrementale di skfolio, imponendo al confine del wrapper il contratto partial_fit. | `optimizer/online/_config.py`, `optimizer/exceptions.py`, `optimizer/scoring/_factory.py`, `external:skfolio`, `external:scikit-learn` |
| `optimizer/optimization/__init__.py` | CORE | Punto di ingresso pubblico del package optimization: raccoglie ed espone le classi Config e le funzioni factory definite nei sedici moduli sorgente sottostanti, uno per famiglia di ottimizzatore skfolio. | `optimizer/optimization/_benchmark_tracker.py`, `optimizer/optimization/_config.py`, `optimizer/optimization/_dr_cvar.py`, `optimizer/optimization/_factory.py`, `optimizer/optimization/_herc.py`, `optimizer/optimization/_hrp.py`, `optimizer/optimization/_max_diversification.py`, `optimizer/optimization/_naive.py`, `optimizer/optimization/_nco.py`, `optimizer/optimization/_regime_blended_mean_risk.py`, `optimizer/optimization/_region_constraints.py`, `optimizer/optimization/_risk_budgeting.py`, `optimizer/optimization/_robust.py`, `optimizer/optimization/_schur.py`, `optimizer/optimization/_stacking.py` |
| `optimizer/optimization/_benchmark_tracker.py` | CORE | Configurazione e factory per skfolio.optimization.BenchmarkTracker, l'ottimizzatore che minimizza la tracking error rispetto a una serie di rendimenti benchmark fornita a fit-time. | `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `optimizer/optimization/_config.py`, `optimizer/optimization/_factory.py`, `external:skfolio` |
| `optimizer/optimization/_config.py` | CORE | Definizione degli enum di selezione (ObjectiveFunctionType, RiskMeasureType, RatioMeasureType) e della dataclass immutabile MeanRiskConfig, la configurazione serializzabile centrale per l'ottimizzatore mean-risk generico. | `optimizer/moments/_config.py` |
| `optimizer/optimization/_dr_cvar.py` | CORE | Configurazione e factory per skfolio.optimization.DistributionallyRobustCVaR, l'ottimizzatore che minimizza il CVaR nel caso peggiore su una palla di Wasserstein attorno alla distribuzione empirica dei rendimenti. | `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `external:skfolio` |
| `optimizer/optimization/_factory.py` | CORE | Factory principale del package: costruisce skfolio.optimization.MeanRisk da MeanRiskConfig, e fornisce gli helper di traduzione (_OBJECTIVE_MAP, _RISK_MEASURE_MAP) e di costruzione dei vincoli settoriali (build_sector_constraints, floor/cap) riusati da quasi tutti gli altri moduli del package. | `optimizer/factors/_integration.py`, `optimizer/moments/_factory.py`, `optimizer/optimization/_config.py`, `external:skfolio` |
| `optimizer/optimization/_herc.py` | CORE | Configurazione e factory per skfolio.optimization.HierarchicalEqualRiskContribution (HERC), l'ottimizzatore gerarchico che equalizza i contributi di rischio tra cluster individuati da un albero di clustering. | `optimizer/cluster/_config.py`, `optimizer/distance/_config.py`, `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `optimizer/optimization/_config.py`, `optimizer/optimization/_factory.py`, `optimizer/optimization/_hierarchical_common.py`, `external:skfolio` |
| `optimizer/optimization/_hierarchical_common.py` | CORE | Helper condiviso dalla famiglia di ottimizzatori gerarchici (HERC, HRP, Schur Complementary): compone in modo uniforme lo stimatore di distanza e lo stimatore di clustering gerarchico a partire dai rispettivi Config, o restituisce None se non configurati. | `optimizer/cluster/_config.py`, `optimizer/cluster/_factory.py`, `optimizer/distance/_config.py`, `optimizer/distance/_factory.py`, `external:skfolio` |
| `optimizer/optimization/_hrp.py` | CORE | Configurazione e factory per skfolio.optimization.HierarchicalRiskParity (HRP), l'ottimizzatore a bisezione ricorsiva su un albero di clustering gerarchico che alloca il rischio senza invertire la matrice di covarianza. | `optimizer/cluster/_config.py`, `optimizer/distance/_config.py`, `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `optimizer/optimization/_config.py`, `optimizer/optimization/_factory.py`, `optimizer/optimization/_hierarchical_common.py`, `external:skfolio` |
| `optimizer/optimization/_max_diversification.py` | CORE | Configurazione e factory per skfolio.optimization.MaximumDiversification, l'ottimizzatore che massimizza il diversification ratio — il rapporto tra la media pesata delle volatilità individuali e la volatilità di portafoglio. | `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `external:skfolio` |
| `optimizer/optimization/_naive.py` | CORE | Configurazione e factory per le tre baseline leggere skfolio.optimization.EqualWeighted, InverseVolatility e Random, usate come punti di confronto per gli ottimizzatori convessi/gerarchici. | `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `external:skfolio` |
| `optimizer/optimization/_nco.py` | CORE | Configurazione e factory per skfolio.optimization.NestedClustersOptimization (NCO), che ottimizza in due stadi — un estimator interno per cluster (intra-cluster) e uno esterno tra cluster (inter-cluster) — su una gerarchia di clustering. | `optimizer/cluster/_config.py`, `optimizer/distance/_config.py`, `optimizer/optimization/_config.py`, `optimizer/optimization/_factory.py`, `optimizer/optimization/_hierarchical_common.py`, `external:skfolio` |
| `optimizer/optimization/_regime_blended_mean_risk.py` | CORE | Configurazione e factory per una pipeline mean-risk con covarianza regime-blended: uno stimatore di covarianza privato pesa ogni osservazione per probabilità di regime esterna (HMM) e per decadimento esponenziale, prima di alimentare un MeanRisk dentro un TimeSeriesFactorModel. | `optimizer/exceptions.py`, `optimizer/optimization/_config.py`, `optimizer/optimization/_factory.py`, `optimizer/pipeline/_builder.py`, `optimizer/pre_selection/_config.py`, `optimizer/validation/_config.py`, `optimizer/validation/_factory.py`, `external:skfolio`, `external:scikit-learn` |
| `optimizer/optimization/_region_constraints.py` | CORE | Helper indipendente che costruisce vincoli lineari di gruppo per regione geografica (analogo di build_sector_constraints in _factory.py ma sull'asse paese→regione anziché ticker→settore) e sanifica i separatori dei nomi di gruppo per il parser DSL di skfolio. | — |
| `optimizer/optimization/_risk_budgeting.py` | CORE | Configurazione e factory per skfolio.optimization.RiskBudgeting, l'ottimizzatore che assegna a ciascun asset un budget di rischio target (default: contributi di rischio uguali, ERC) e risolve i pesi che lo realizzano. | `optimizer/exceptions.py`, `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `external:skfolio` |
| `optimizer/optimization/_robust.py` | CORE | Configurazione e factory per la famiglia robusta di MeanRisk: attacca insiemi di incertezza ellissoidali su media (mu) e/o covarianza a un MeanRisk altrimenti standard, in modo che con entrambi i Config a None il risultato coincida esattamente con build_mean_risk. | `optimizer/optimization/_config.py`, `optimizer/optimization/_factory.py`, `optimizer/uncertainty_set/_config.py`, `optimizer/uncertainty_set/_factory.py`, `external:skfolio` |
| `optimizer/optimization/_schur.py` | CORE | Configurazione e factory per skfolio.optimization.SchurComplementary, l'ottimizzatore gerarchico che interpola tramite il parametro gamma tra Hierarchical Risk Parity (gamma=0) e il portafoglio di minima varianza (gamma=1). | `optimizer/cluster/_config.py`, `optimizer/distance/_config.py`, `optimizer/exceptions.py`, `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `optimizer/optimization/_hierarchical_common.py`, `external:skfolio` |
| `optimizer/optimization/_stacking.py` | CORE | Configurazione e factory per skfolio.optimization.StackingOptimization, un meta-ottimizzatore ensemble che combina più estimator di base tramite un final_estimator che apprende come pesarli. | `optimizer/exceptions.py`, `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `optimizer/optimization/_config.py`, `optimizer/optimization/_factory.py`, `external:skfolio` |
| `optimizer/pipeline/__init__.py` | CORE | Facciata pubblica del sottopacchetto pipeline: il docstring dichiara che il pacchetto compone pre-selezione, ottimizzazione, validazione, scoring, tuning degli iperparametri e ribilanciamento in un unico workflow end-to-end. | `optimizer/pipeline/_builder.py`, `optimizer/pipeline/_config.py`, `optimizer/pipeline/_orchestrator.py` |
| `optimizer/pipeline/_builder.py` | CORE | Punto di composizione fra la pipeline di pre-selezione (optimizer/pre_selection/) e un ottimizzatore skfolio, in un'unica sklearn Pipeline utilizzabile come singolo estimator. | `optimizer/pre_selection/_config.py`, `optimizer/pre_selection/_pipeline.py`, `external:scikit-learn` |
| `optimizer/pipeline/_config.py` | CORE | Dataclass PortfolioResult, il contenitore tipizzato che raccoglie l'intero output di una run di ottimizzazione di portafoglio end-to-end. | `optimizer/factors/_config.py`, `optimizer/fx/_decomposition.py`, `external:pandas` |
| `optimizer/pipeline/_orchestrator.py` | CORE | Le funzioni di orchestrazione end-to-end del pacchetto: dal prezzo grezzo al portafoglio validato, con backtest, tuning, iniezione di bande settoriali dipendenti dal regime, correzione FX, costi di transazione netti e analisi di ribilanciamento. | `optimizer/exceptions.py`, `optimizer/factors/_config.py`, `optimizer/factors/_construction.py`, `optimizer/factors/_regime.py`, `optimizer/factors/_scoring.py`, `optimizer/factors/_sector_bands.py`, `optimizer/factors/_selection.py`, `optimizer/factors/_standardization.py`, `optimizer/factors/_integration.py`, `optimizer/fx/_config.py`, `optimizer/fx/_decomposition.py`, `optimizer/fx/_factory.py`, `optimizer/optimization/_config.py`, `optimizer/optimization/_region_constraints.py`, `optimizer/pipeline/_builder.py`, `optimizer/pipeline/_config.py`, `optimizer/pre_selection/_config.py`, `optimizer/preprocessing/_delisting.py`, `optimizer/rebalancing/_config.py`, `optimizer/rebalancing/_rebalancer.py`, `optimizer/scoring/_config.py`, `optimizer/tuning/_config.py`, `optimizer/tuning/_factory.py`, `optimizer/universe/_config.py`, `optimizer/universe/_factory.py`, `optimizer/validation/_config.py`, `optimizer/validation/_factory.py`, `external:skfolio`, `external:scikit-learn`, `external:numpy` |
| `optimizer/pre_selection/__init__.py` | CORE | Facciata pubblica del sottopacchetto `pre_selection`: dichiara cosa è importabile dall'esterno del pacchetto. | `optimizer/pre_selection/_config.py`, `optimizer/pre_selection/_pipeline.py` |
| `optimizer/pre_selection/_config.py` | CORE | Dataclass immutabile `PreSelectionConfig` che raccoglie tutti gli iperparametri della pipeline di pre-selezione, più due preset di fabbrica. | — |
| `optimizer/pre_selection/_pipeline.py` | CORE | Factory function `build_preselection_pipeline` che assembla una `sklearn.pipeline.Pipeline` di pulizia dato e pre-selezione asset a partire da una `PreSelectionConfig`. | `optimizer/pre_selection/_config.py`, `optimizer/preprocessing/_validation.py`, `optimizer/preprocessing/_outliers.py`, `optimizer/preprocessing/_imputation.py`, `external:skfolio`, `external:scikit-learn` |
| `optimizer/preprocessing/__init__.py` | CORE | Punto di ingresso pubblico del package preprocessing: raccoglie ed espone le classi e la funzione definite nei sei moduli sorgente sottostanti. | `optimizer/preprocessing/_cs_transformers.py`, `optimizer/preprocessing/_delisting.py`, `optimizer/preprocessing/_imputation.py`, `optimizer/preprocessing/_outliers.py`, `optimizer/preprocessing/_regression_imputer.py`, `optimizer/preprocessing/_validation.py` |
| `optimizer/preprocessing/_cs_transformers.py` | CORE | Wrapper di ri-esportazione dei trasformatori cross-sectional forniti dalla libreria esterna skfolio (versione 0.20+), da usare come feature preprocessor per CSLinearRegression o segnali fattoriali. | `external:skfolio` |
| `optimizer/preprocessing/_delisting.py` | CORE | Guardia contro il survivorship bias: sostituisce l'ultimo rendimento valido di un ticker con il suo rendimento di delisting fornito dal chiamante. | `optimizer/exceptions.py` |
| `optimizer/preprocessing/_imputation.py` | CORE | Trasformatore sklearn-compatibile (SectorImputer) che riempie i valori NaN con la media cross-sezionale leave-one-out del settore di appartenenza dell'asset. | `optimizer/exceptions.py` |
| `optimizer/preprocessing/_outliers.py` | CORE | Trasformatore sklearn-compatibile (OutlierTreater) che applica una metodologia a tre gruppi sugli z-score per colonna, distinguendo osservazioni normali, outlier ed errori di dato. | `optimizer/exceptions.py` |
| `optimizer/preprocessing/_regression_imputer.py` | CORE | Trasformatore sklearn-compatibile (RegressionImputer) che imputa i valori NaN tramite regressione OLS sui top-K asset più correlati, con riserva su SectorImputer. | `optimizer/exceptions.py`, `optimizer/preprocessing/_imputation.py` |
| `optimizer/preprocessing/_validation.py` | CORE | Trasformatore sklearn-compatibile (DataValidator) pensato come primo stadio di una pipeline di pre-selezione: sostituisce infiniti e rendimenti estremi con NaN. | `optimizer/exceptions.py` |
| `optimizer/rebalancing/__init__.py` | CORE | Punto d'ingresso pubblico del sottopacchetto rebalancing: raccoglie in un unico namespace le configurazioni di innesco e le funzioni di calcolo/decisione definite negli altri due file. | `optimizer/rebalancing/_config.py`, `optimizer/rebalancing/_rebalancer.py` |
| `optimizer/rebalancing/_config.py` | CORE | Definisce, come dati immutabili, le politiche di innesco della revisione di portafoglio: a calendario fisso, a soglia di drift, e ibrida (calendario che apre una finestra di verifica della soglia). | — |
| `optimizer/rebalancing/_rebalancer.py` | CORE | Implementa la logica operativa di revisione: calcola i pesi dopo il drift dei rendimenti, il turnover e il costo di transazione fra pesi correnti e target, e decide se innescare la revisione secondo le configurazioni di _config.py. | `optimizer/rebalancing/_config.py`, `fund/app/agents/execution.py` |
| `optimizer/regimes/__init__.py` | CORE ·  pianificato | Superficie pubblica del nuovo pacchetto regimes/ — il barrel module che raccoglie config, enum, tipi di risultato e funzioni build_* dei sei moduli di implementazione (_config, _markov_switching, _nowcast, _turbulence, _absorption, _cross_sectional) più il modulo di composizione _factory, in un unico spazio dei nomi importabile. | `optimizer/regimes/_config.py`, `optimizer/regimes/_markov_switching.py`, `optimizer/regimes/_nowcast.py`, `optimizer/regimes/_turbulence.py`, `optimizer/regimes/_absorption.py`, `optimizer/regimes/_cross_sectional.py`, `optimizer/regimes/_factory.py` |
| `optimizer/regimes/_absorption.py` | CORE ·  pianificato | Absorption ratio e rango effettivo — due misure basate su PCA di quanto sia compatta, e quindi fragile, l'attuale struttura di covarianza del mercato. | `optimizer/regimes/_config.py`, `optimizer/exceptions.py`, `external:numpy`, `external:scikit-learn` |
| `optimizer/regimes/_config.py` | CORE ·  pianificato | Vocabolario condiviso del pacchetto regimes/: la tassonomia RegimeLabel, un dataclass di configurazione frozen per ciascuna famiglia di stimatore (Markov-switching, nowcast, turbulence, absorption ratio, tilt cross-sezionale) e il contenitore di risultato frozen MarketState che _factory.py assembla campo per campo. | `optimizer/exceptions.py` |
| `optimizer/regimes/_cross_sectional.py` | CORE ·  pianificato | Trasforma la lettura filtrata del regime nell'intensità state-conditional con cui i segnali cross-sezionali di momentum, value e size vanno pesati — il ponte da regime a livello di classe di attività a selezione all'interno della classe che ARCHITECTURE.md e il capitolo 26 richiedono entrambi. | `optimizer/regimes/_config.py`, `optimizer/regimes/_markov_switching.py`, `optimizer/exceptions.py` |
| `optimizer/regimes/_factory.py` | CORE ·  pianificato | L'unico punto di composizione del pacchetto — build_market_state — che trasforma un pannello di rendimenti/dati macro nel singolo oggetto MarketState che il resto della piattaforma è inteso leggere, invece che ciascun consumatore stimi il proprio regime. | `optimizer/regimes/_config.py`, `optimizer/regimes/_markov_switching.py`, `optimizer/regimes/_nowcast.py`, `optimizer/regimes/_turbulence.py`, `optimizer/regimes/_absorption.py`, `optimizer/regimes/_cross_sectional.py` |
| `optimizer/regimes/_markov_switching.py` | CORE ·  pianificato | Stimatore a stato latente con cambio di regime di Markov e il filtro ricorsivo di Hamilton — il pezzo dell'architettura che oggi non esiste in nessun punto della base di codice. | `optimizer/regimes/_config.py`, `optimizer/exceptions.py`, `external:statsmodels`, `external:numpy` |
| `optimizer/regimes/_nowcast.py` | CORE ·  pianificato | Stima in tempo reale dello stato macro corrente da un pannello sbilanciato e a frequenza mista di serie macro — la metà a stima deterministica della quantità 'premi al rischio attesi condizionati' dell'agente macro. | `optimizer/regimes/_config.py`, `optimizer/exceptions.py`, `external:statsmodels` |
| `optimizer/regimes/_turbulence.py` | CORE ·  pianificato | Turbulence finanziaria — una misura a distanza di Mahalanobis di quanto sia statisticamente inusuale la sezione trasversale corrente dei rendimenti rispetto al proprio pattern storico — e la sua decomposizione in sorpresa di grandezza e sorpresa di correlazione. | `optimizer/regimes/_config.py`, `optimizer/exceptions.py`, `external:numpy`, `external:scipy` |
| `optimizer/scoring/__init__.py` | CORE | Modulo radice del pacchetto scoring: fissa nel docstring il ruolo dell'intero submodulo — avvolgere le misure di rapporto (ratio measure) di skfolio e funzioni di scoring custom in callable compatibili con la cross-validation di sklearn. | `optimizer/scoring/_config.py`, `optimizer/scoring/_factory.py` |
| `optimizer/scoring/_config.py` | CORE | Vocabolario tipizzato e configurazione immutabile per costruire una funzione di scoring. | `optimizer/optimization/_config.py` |
| `optimizer/scoring/_factory.py` | CORE | Punto di traduzione dalla configurazione dichiarativa di _config.py a un callable di scoring concreto, pronto per GridSearchCV/RandomizedSearchCV di sklearn. | `optimizer/scoring/_config.py`, `optimizer/optimization/_config.py`, `optimizer/exceptions.py`, `external:skfolio`, `external:numpy`, `external:pandas` |
| `optimizer/synthetic/__init__.py` | CORE | Superficie pubblica del pacchetto `optimizer/synthetic`, il sottomodulo di dominio puro per la generazione di dati sintetici e i modelli a vine copula. | `optimizer/synthetic/_config.py`, `optimizer/synthetic/_factory.py` |
| `optimizer/synthetic/_config.py` | CORE | Configurazioni immutabili e serializzabili per il modello a vine copula (`VineCopulaConfig`) e per il prior di generazione di scenari sintetici (`SyntheticDataConfig`), più gli enum che ne tipizzano i parametri categoriali. | — |
| `optimizer/synthetic/_factory.py` | CORE | Funzioni factory che traducono le configurazioni di `_config.py` in stimatori skfolio pronti all'uso: il modello a vine copula e il prior di dati sintetici che campiona da esso. | `optimizer/synthetic/_config.py`, `optimizer/exceptions.py`, `external:skfolio` |
| `optimizer/tuning/__init__.py` | CORE | Modulo radice del pacchetto tuning: il docstring dichiara il ruolo dell'intero submodulo — avvolgere GridSearchCV e RandomizedSearchCV di sklearn con default di cross-validation temporale (walk-forward) per prevenire il look-ahead bias nelle serie storiche finanziarie. | `optimizer/tuning/_config.py`, `optimizer/tuning/_factory.py` |
| `optimizer/tuning/_config.py` | CORE | Vocabolario tipizzato e configurazione immutabile per la ricerca di iperparametri, con la cross-validation temporale (walk-forward) imposta come default anziché come opzione. | `optimizer/scoring/_config.py`, `optimizer/validation/_config.py` |
| `optimizer/tuning/_factory.py` | CORE | Punto di traduzione dalla configurazione dichiarativa di _config.py a un estimator di ricerca iperparametri concreto — GridSearchCV o RandomizedSearchCV di sklearn — pronto per il fit su un optimizer o pipeline di skfolio. | `optimizer/tuning/_config.py`, `optimizer/scoring/_factory.py`, `optimizer/validation/_factory.py`, `external:scikit-learn` |
| `optimizer/uncertainty_set/__init__.py` | CORE | Facciata pubblica del pacchetto: fissa nel docstring di modulo il perimetro degli stimatori di insiemi di incertezza per l'ottimizzazione robusta di portafoglio, inclusa la nota implementativa che la variante bootstrap si appoggia ad arch.StationaryBootstrap con block size di default alla regola di Politis-White quando block_size è None. | `optimizer/uncertainty_set/_config.py`, `optimizer/uncertainty_set/_factory.py` |
| `optimizer/uncertainty_set/_config.py` | CORE | Superficie di configurazione tipizzata e immutabile per i due stimatori di insieme di incertezza — sui rendimenti attesi (mu) e sulla covarianza — che il pacchetto costruisce. | `optimizer/exceptions.py` |
| `optimizer/uncertainty_set/_factory.py` | CORE | Punto di traduzione dagli oggetti Config propri del pacchetto alle classi stimatore di insieme di incertezza di skfolio. | `optimizer/uncertainty_set/_config.py`, `external:skfolio` |
| `optimizer/universe/__init__.py` | CORE | Facciata pubblica del sottopacchetto `universe`: fissa quali nomi sono importabili dall'esterno del pacchetto. | `optimizer/universe/_config.py`, `optimizer/universe/_factory.py`, `optimizer/universe/_screener.py` |
| `optimizer/universe/_config.py` | CORE | Configurazione tipizzata e immutabile che parametrizza lo screening di investability — soglie di ingresso/uscita per capitalizzazione, liquidità, prezzo, storia di quotazione e disponibilità di bilanci. | `optimizer/exceptions.py` |
| `optimizer/universe/_factory.py` | CORE | Funzione di convenienza che avvolge la pipeline di screening con una configurazione di default. | `optimizer/universe/_config.py`, `optimizer/universe/_screener.py`, `optimizer/pipeline/_orchestrator.py` |
| `optimizer/universe/_screener.py` | CORE | Logica di calcolo dello screening di investability: le funzioni pure che misurano capitalizzazione, liquidità, prezzo, anzianità di quotazione e copertura di bilanci, e le compongono in una sequenza fissa di filtri con isteresi. | `optimizer/universe/_config.py` |
| `optimizer/validation/__init__.py` | CORE | Facciata pubblica del pacchetto validation: il docstring di modulo lo inquadra come il punto che raccoglie walk-forward backtesting, Combinatorial Purged Cross-Validation (CPCV), Multiple Randomized Cross-Validation e la valutazione (offline + online) dei forecast di covarianza, oltre al validatore regime-conditional. | `optimizer/validation/_config.py`, `optimizer/validation/_factory.py`, `optimizer/validation/_forecast.py`, `optimizer/validation/_regime.py`, `external:skfolio` |
| `optimizer/validation/_config.py` | CORE | Vocabolario di configurazione immutabile per le tre tecniche di cross-validation temporale che il pacchetto avvolge: WalkForward, CombinatorialPurgedCV e MultipleRandomizedCV di skfolio. | — |
| `optimizer/validation/_factory.py` | CORE | Punto di traduzione dalle config dichiarative di _config.py a cross-validator skfolio concreti, più un wrapper temporale-only attorno a cross_val_predict e un ottimizzatore del numero di fold per CPCV. | `optimizer/validation/_config.py`, `external:skfolio` |
| `optimizer/validation/_forecast.py` | CORE | Valutazione walk-forward, offline e online, di stimatori di covarianza candidati, esplicitamente dichiarata agnostica rispetto all'ottimizzatore a valle e non sostitutiva di un backtest completo. | `optimizer/exceptions.py`, `external:skfolio` |
| `optimizer/validation/_regime.py` | CORE | Decomposizione regime-conditional dei rendimenti di portafoglio fuori campione: separa la performance per regime macro e per sottoperiodo contiguo dello stesso regime, e segnala le strategie il cui alpha positivo è concentrato in un unico regime. | `optimizer/exceptions.py`, `optimizer/factors/_config.py`, `optimizer/factors/_regime.py` |
| `optimizer/views/__init__.py` | CORE | Punto d'ingresso pubblico del sottopacchetto `views`: dichiara quali classi di configurazione e quali funzioni di fabbrica sono la API stabile del pacchetto. | `optimizer/views/_config.py`, `optimizer/views/_factory.py`, `optimizer/views/_uncertainty.py` |
| `optimizer/views/_config.py` | CORE | Configurazioni immutabili e serializzabili per i tre framework di integrazione delle view: Black-Litterman, Entropy Pooling, Opinion Pooling; nessun calcolo, solo dati tipizzati e i loro vincoli di validità. | `optimizer/moments/_config.py` |
| `optimizer/views/_factory.py` | CORE | Le funzioni di fabbrica che trasformano una configurazione di `_config.py` in uno stimatore `skfolio.prior` pronto per il fit, più una sottoclasse di `BlackLitterman` che sostituisce l'omega calcolato analiticamente con un omega empirico pre-calcolato. | `optimizer/views/_config.py`, `optimizer/views/_uncertainty.py`, `optimizer/moments/_config.py`, `optimizer/moments/_factory.py`, `optimizer/exceptions.py` |
| `optimizer/views/_uncertainty.py` | CORE | Calibrazione empirica della matrice di incertezza delle view (omega) a partire da uno storico di errori di previsione, come alternativa ai metodi He-Litterman/Idzorek. | `optimizer/exceptions.py` |

<!-- FINE GENERATO: optimizer -->

---

## 5. `ingestion/` — il demone di raccolta dati

Il demone è, insieme a `portopt`, l'unico pezzo già in produzione. Una passata di ricerca
dedicata ha valutato se spostarlo su una coda di task distribuita (Celery, Temporal,
Dramatiq), su I/O asincrono (asyncio, httpx-async, asyncpg) o su cron multi-replica.
**Verdetto: si tiene l'architettura attuale** — APScheduler in-process, SQLAlchemy 2.0
sincrono, worker a thread, heartbeat e reaper — con quattro raffinamenti mirati. Le
affermazioni che avrebbero giustificato la migrazione sono state **confutate** in
contraddittorio (§11); le "modernizzazioni" sarebbero state errori per questo carico.

Vale la pena dire perché, altrimenti fra un anno qualcuno rifà la stessa analisi. Questo
carico è **poche decine di job al giorno, ciascuno lungo** — una fetch yfinance su qualche
migliaio di ticker, un seed dell'indice di riferimento. Non è il carico per cui le code
distribuite sono state progettate, cioè molti task brevi con throughput alto. Ciò che qui
costa è la latenza di rete verso le sorgenti esterne, e quella non la si toglie cambiando
scheduler.

### 5.1 Scheduler — APScheduler in-process; un demone per database è corretto

APScheduler 3.x vieta di condividere un job store fra scheduler diversi (*"Job stores must
never be shared between schedulers"*), quindi la regola **"esattamente un demone per
database" è la mitigazione standard corretta, non un difetto da correggere**. Per job cron
non idempotenti la giusta inclinazione è **saltare un lancio piuttosto che rischiarne uno
doppio** (Google SRE): un exactly-once vero richiede consenso sincrono alla Paxos, che
APScheduler in singolo processo non fornisce e qui non serve.

APScheduler 4.x — che il multi-scheduler lo sanziona esplicitamente — era ancora
pre-release al cutoff di gennaio 2026. **Non pianificare uno scale-out su 4.x senza
rivalidare** questa sezione.

### 5.2 Percorso di scala — documentato, non costruito

Un demone per database resta il disegno **adesso**. Se il tempo di parete delle fetch
dovesse superare la finestra giornaliera, il percorso di scala sanzionato è una **elezione
del leader con advisory lock PostgreSQL** (`pg_try_advisory_xact_lock`): le repliche
competono, solo chi tiene il lock fa scattare i trigger cron, le altre restano hot standby
o worker di sola fetch. Riusa il Postgres già presente — niente Redis, niente Redlock.

**Avvertenza di deployment:** il transaction pooling di PgBouncer rompe gli advisory lock a
livello di sessione. Va usata la variante con scope di transazione. Nessun codice adesso;
questo è registrato come percorso futuro.

### 5.3 Reaper degli orfani — a lease, e RECLAIM invece di FAIL

Due decisioni, con una dipendenza fra loro.

**Reaper a lease (adottato).** I worker rinnovano `last_heartbeat_at` sui job in volo a ogni
cadenza di heartbeat; il reaper recupera una claim solo quando il **TTL della lease** è
scaduto — un multiplo della cadenza, per esempio 120s di TTL contro 30s di cadenza. Questo
elimina il false-reaping dei passi sincroni lunghi, che è esattamente il modo di fallire già
documentato: la fetch yfinance e il seed dell'indice di riferimento superano il timeout
piatto di 300s. È il *rinnovo* della lease, non un timeout fisso sulla claim, a impedire di
mietere lavoro lento ma vivo. Sostituisce l'attuale aggiramento a heartbeat sincrono.

**RECLAIM e rimessa in coda invece di FAIL (adottato, ma GATED).** Si passa dal far fallire
i job dei worker morti al rieseguirli — at-least-once, auto-riparante: è il modello del
visibility timeout delle code. **Questo è insicuro finché §5.4 non è soddisfatta**:
rieseguire un job non idempotente corrompe i dati. Il gate è esplicito — si abilita il
reclaim solo a audit degli upsert completato.

### 5.4 Scritture idempotenti — requisito duro, prerequisito del reclaim

Poiché il reaper (e qualunque coda futura) riesegue il lavoro orfano, la consegna è
**at-least-once**, quindi **ogni job di fetch DEVE scrivere con upsert idempotenti** —
`INSERT ... ON CONFLICT DO UPDATE` su chiavi naturali. Una riesecuzione deve essere sicura e
convergente.

**Azione:** verificare i repository e i service esistenti e convertire in upsert ogni
scrittura a insert semplice, **prima** di abilitare il reclaim di §5.3. Finché l'audit non è
chiuso, il reaper resta in modalità FAIL.

L'ordine fra le due non è negoziabile e non è una preferenza stilistica: invertirlo
significa avere un sistema che si auto-ripara duplicando righe.

### 5.5 Spegnimento pulito — da irrobustire

Intercettare SIGTERM → smettere di reclamare nuovi job → drenare e committare (o mettere un
checkpoint) il lavoro in volo → uscire, con un drain limitato nel tempo e un'uscita forzata
di riserva, così che un worker bloccato non possa restare appeso per sempre.

**Il periodo di grazia del container va dimensionato ben sopra i 30s di default**
(`terminationGracePeriodSeconds` ≈ preStop + durata massima del job + pulizia + margine):
altrimenti i passi di fetch e seed, che durano minuti, prendono SIGKILL a metà scrittura. Uno
spegnimento drenato bene evita anche di lasciare righe orfane al reaper, che è il modo
migliore di gestire un problema. Vale per la restart-policy di Docker e per la supervisione
systemd, non solo per Kubernetes.

### 5.6 Sincrono contro asincrono — si resta sincroni, con worker a thread

Nessuna riscrittura async. Il "async è ~15x più veloce" che si cita spesso è un effetto di
**concorrenza** — sequenziale contro concorrente — non di async che batte i thread. Il pool
di thread del demone (`YFINANCE_FETCH_WORKERS`, 1-16) cattura già il guadagno di concorrenza
e si compone in modo pulito con CircuitBreaker e RateLimiter per-scraper, che servono a
rispettare i rate limit delle sorgenti.

Le affermazioni che avrebbero giustificato la migrazione — "asyncio ~3.5x più veloce dei
thread", "asyncpg ~3.4x di latenza in meno e ~1.9x di throughput rispetto a SQLAlchemy" —
sono state tutte **confutate 0-3** (§11).

*Da rivalidare al momento della costruzione:* il free-threading di Python 3.13+ (rimozione
del GIL) può spostare il compromesso fra thread e async. È l'unica cosa che rimetterebbe in
discussione questa sezione.

<!-- GENERATO: ingestion -->

_Da qui in giù la sezione è generata da `scriptorium arch` il 2026-08-19: non modificarla a mano, si riscrive a ogni assemblaggio. Il record completo — responsabilità estese e citazioni riga per riga — sta in `data/architecture-spec/`, da cui questo è renderizzato._

### 5.1 Albero

269 file, ciascuno annotato col suo ruolo esagonale. Ruoli: config 8 · CORE 29 · DRIVEN 115 · DRIVING 114 · PORTS 3

```
└── ingestion/
    ├── alembic/
    │   ├── README  ┃ config
    │   ├── env.py  ┃ config
    │   ├── script.py.mako  ┃ config
    │   └── versions/
    │       ├── .gitkeep  ┃ config
    │       ├── 1c95f3fd3da4_remove_jsonb_columns_raw_json_and_.py  ┃ DRIVEN
    │       ├── 1e14baf18033_add_server_defaults_to_timestamp_columns.py  ┃ DRIVEN
    │       ├── 2a3be07fc983_drop_regime_tables.py  ┃ DRIVEN
    │       ├── 6b106e82ce20_increase_financial_statements_value_.py  ┃ DRIVEN
    │       ├── a1b2c3d4e5f6_rebuild_macro_tables_to_match_orm.py  ┃ DRIVEN
    │       ├── a7b8c9d0e1f2_seed_reference_indices_expanded.py  ┃ DRIVEN
    │       ├── b0b30daa80c5_normalize_and_optimize_models.py  ┃ DRIVEN
    │       ├── b1c2d3e4f5g6_add_regime_classification_to_macro_calibrations.py  ┃ DRIVEN
    │       ├── b2c3d4e5f6a7_drop_orphaned_tables.py  ┃ DRIVEN
    │       ├── bdd615f586f8_remove_users_table.py  ┃ DRIVEN
    │       ├── c3d4e5f6a7b8_align_exchanges_with_orm.py  ┃ DRIVEN
    │       ├── c51c70973022_normalize_instruments_drop_exchange_.py  ┃ DRIVEN
    │       ├── c7d8e9f0a1b2_add_worker_liveness_columns.py  ┃ DRIVEN
    │       ├── cfc1deec33d6_add_yfinance_data_tables.py  ┃ DRIVEN
    │       ├── d1e2f3a4b5c6_drop_non_ingestion_tables.py  ┃ DRIVEN
    │       ├── d4e5f6a7b8c9_add_delisting_columns_to_instruments.py  ┃ DRIVEN
    │       ├── e5f6a7b8c9d0_add_fred_observations_table.py  ┃ DRIVEN
    │       ├── ecf0a9a2bfdc_add_macro_regime_tables.py  ┃ DRIVEN
    │       ├── f6a7b8c9d0e1_drop_ilsole_real_columns.py  ┃ DRIVEN
    │       ├── ff232ce0534f_initial_schema_users_exchanges_.py  ┃ DRIVEN
    │       ├── g7b8c9d0e1f2_drop_st_rate_forecast.py  ┃ DRIVEN
    │       ├── h8c9d0e1f2g3_add_macro_observation_tables.py  ┃ DRIVEN
    │       ├── i9d0e1f2g3h4_add_econ_indicator_observations.py  ┃ DRIVEN
    │       ├── j0e1f2g3h4i5_add_macro_news_table.py  ┃ DRIVEN
    │       ├── k1f2g3h4i5j6_add_macro_news_full_content.py  ┃ DRIVEN
    │       ├── l2g3h4i5j6k7_add_ticker_news_full_content.py  ┃ DRIVEN
    │       ├── m3h4i5j6k7l8_drop_ticker_news_related_tickers.py  ┃ DRIVEN
    │       ├── n4i5j6k7l8m9_add_ticker_news_ticker_name.py  ┃ DRIVEN
    │       ├── o5j6k7l8m9n0_add_composite_indexes_on_observation_tables.py  ┃ DRIVEN
    │       ├── p6q7r8s9t0u1_add_macro_calibrations_table.py  ┃ DRIVEN
    │       ├── q7r8s9t0u1v2_add_macro_news_summaries_table.py  ┃ DRIVEN
    │       ├── r8s9t0u1v2w3_add_background_jobs_table.py  ┃ DRIVEN
    │       ├── s9t0u1v2w3x4_add_portfolio_tables.py  ┃ DRIVEN
    │       ├── t0u1v2w3x4y5_normalize_json_columns.py  ┃ DRIVEN
    │       ├── u1v2w3x4y5z6_add_currency_code_to_financial_statements.py  ┃ DRIVEN
    │       ├── v2w3x4y5z6a7_add_external_id_to_activity_events.py  ┃ DRIVEN
    │       ├── w3x4y5z6a7b8_add_snapshot_date_to_broker_account_snapshots.py  ┃ DRIVEN
    │       ├── x4y5z6a7b8c9_seed_reference_indices.py  ┃ DRIVEN
    │       ├── y5z6a7b8c9d0_add_factor_scores_execution_runs_risk_limits_rebalancing_policies_tables.py  ┃ DRIVEN
    │       └── z6a7b8c9d0e1_add_api_keys_table.py  ┃ DRIVEN
    ├── app/
    │   ├── cli.py  ┃ DRIVING
    │   ├── metrics.py  ┃ DRIVEN
    │   ├── models/
    │   │   ├── __init__.py  ┃ DRIVEN
    │   │   ├── _shared/
    │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   └── base.py  ┃ DRIVEN
    │   │   ├── jobs/
    │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   └── background_job.py  ┃ DRIVEN
    │   │   ├── macro/
    │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   └── macro_regime.py  ┃ DRIVEN
    │   │   ├── market_data/
    │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   └── yfinance_data.py  ┃ DRIVEN
    │   │   └── universe/
    │   │       ├── __init__.py  ┃ DRIVEN
    │   │       └── universe.py  ┃ DRIVEN
    │   ├── repositories/
    │   │   ├── __init__.py  ┃ DRIVEN
    │   │   ├── _shared/
    │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   ├── base.py  ┃ DRIVEN
    │   │   │   └── database_admin_repository.py  ┃ DRIVEN
    │   │   ├── jobs/
    │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   └── background_job_repository.py  ┃ DRIVEN
    │   │   ├── macro/
    │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   ├── macro_regime_repository.py  ┃ DRIVEN
    │   │   │   └── sentiment_repository.py  ┃ DRIVEN
    │   │   ├── market_data/
    │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   └── yfinance_repository.py  ┃ DRIVEN
    │   │   └── universe/
    │   │       ├── __init__.py  ┃ DRIVEN
    │   │       └── universe_repository.py  ┃ DRIVEN
    │   ├── schemas/
    │   │   ├── __init__.py  ┃ DRIVING
    │   │   ├── _shared/
    │   │   │   ├── __init__.py  ┃ DRIVING
    │   │   │   ├── base.py  ┃ DRIVING
    │   │   │   └── base_job.py  ┃ DRIVING
    │   │   ├── macro/
    │   │   │   ├── __init__.py  ┃ DRIVING
    │   │   │   └── macro_regime.py  ┃ DRIVING
    │   │   ├── market_data/
    │   │   │   ├── __init__.py  ┃ DRIVING
    │   │   │   ├── reference_index.py  ┃ DRIVING
    │   │   │   └── yfinance_data.py  ┃ DRIVING
    │   │   └── universe/
    │   │       ├── __init__.py  ┃ DRIVING
    │   │       └── trading212.py  ┃ DRIVING
    │   ├── services/
    │   │   ├── __init__.py  ┃ CORE
    │   │   ├── _shared/
    │   │   │   ├── __init__.py  ┃ CORE
    │   │   │   ├── _benchmark_bootstrap.py  ┃ CORE
    │   │   │   ├── _json_safe.py  ┃ CORE
    │   │   │   ├── _price_fetcher.py  ┃ CORE
    │   │   │   ├── _progress.py  ┃ CORE
    │   │   │   ├── _sector_resolver.py  ┃ CORE
    │   │   │   ├── notifications.py  ┃ DRIVEN
    │   │   │   └── trading_calendar.py  ┃ CORE
    │   │   ├── infrastructure/
    │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   ├── cache.py  ┃ DRIVEN
    │   │   │   ├── circuit_breaker.py  ┃ DRIVEN
    │   │   │   ├── rate_limiter.py  ┃ DRIVEN
    │   │   │   └── retry.py  ┃ DRIVEN
    │   │   ├── jobs/
    │   │   │   ├── __init__.py  ┃ CORE
    │   │   │   ├── background_job.py  ┃ CORE
    │   │   │   └── scheduler.py  ┃ DRIVING
    │   │   ├── macro/
    │   │   │   ├── __init__.py  ┃ CORE
    │   │   │   ├── macro_calibration.py  ┃ CORE
    │   │   │   ├── macro_news_summary.py  ┃ CORE
    │   │   │   ├── macro_regime_service.py  ┃ CORE
    │   │   │   └── scrapers/
    │   │   │       ├── __init__.py  ┃ DRIVEN
    │   │   │       ├── exceptions.py  ┃ DRIVEN
    │   │   │       ├── fred_scraper.py  ┃ DRIVEN
    │   │   │       ├── ilsole_scraper.py  ┃ DRIVEN
    │   │   │       └── tradingeconomics_scraper.py  ┃ DRIVEN
    │   │   ├── market_data/
    │   │   │   ├── __init__.py  ┃ CORE
    │   │   │   ├── reference_index_seeder.py  ┃ CORE
    │   │   │   ├── yfinance/
    │   │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   │   ├── _base.py  ┃ DRIVEN
    │   │   │   │   ├── _facade.py  ┃ DRIVEN
    │   │   │   │   ├── infrastructure/
    │   │   │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   │   │   ├── cache.py  ┃ DRIVEN
    │   │   │   │   │   ├── circuit_breaker.py  ┃ DRIVEN
    │   │   │   │   │   ├── rate_limiter.py  ┃ DRIVEN
    │   │   │   │   │   └── retry.py  ┃ DRIVEN
    │   │   │   │   ├── market/
    │   │   │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   │   │   └── search.py  ┃ DRIVEN
    │   │   │   │   ├── news/
    │   │   │   │   │   ├── __init__.py  ┃ DRIVEN
    │   │   │   │   │   ├── aggregator.py  ┃ DRIVEN
    │   │   │   │   │   ├── client.py  ┃ DRIVEN
    │   │   │   │   │   ├── macro_news.py  ┃ DRIVEN
    │   │   │   │   │   └── scraper.py  ┃ DRIVEN
    │   │   │   │   ├── protocols/
    │   │   │   │   │   ├── __init__.py  ┃ PORTS
    │   │   │   │   │   └── interfaces.py  ┃ PORTS
    │   │   │   │   └── ticker/
    │   │   │   │       ├── __init__.py  ┃ DRIVEN
    │   │   │   │       ├── analysis.py  ┃ DRIVEN
    │   │   │   │       ├── corporate_actions.py  ┃ DRIVEN
    │   │   │   │       ├── financials.py  ┃ DRIVEN
    │   │   │   │       ├── holders.py  ┃ DRIVEN
    │   │   │   │       └── metadata.py  ┃ DRIVEN
    │   │   │   └── yfinance_data_service.py  ┃ CORE
    │   │   └── universe/
    │   │       ├── __init__.py  ┃ CORE
    │   │       ├── trading212/
    │   │       │   ├── __init__.py  ┃ CORE
    │   │       │   ├── builder.py  ┃ CORE
    │   │       │   ├── cache/
    │   │       │   │   ├── __init__.py  ┃ DRIVEN
    │   │       │   │   └── ticker_cache.py  ┃ DRIVEN
    │   │       │   ├── client.py  ┃ DRIVEN
    │   │       │   ├── config.py  ┃ CORE
    │   │       │   ├── filters/
    │   │       │   │   ├── __init__.py  ┃ CORE
    │   │       │   │   ├── data_coverage.py  ┃ CORE
    │   │       │   │   ├── historical_data.py  ┃ CORE
    │   │       │   │   ├── liquidity.py  ┃ CORE
    │   │       │   │   ├── market_cap.py  ┃ CORE
    │   │       │   │   ├── pipeline.py  ┃ CORE
    │   │       │   │   └── price.py  ┃ CORE
    │   │       │   ├── protocols.py  ┃ PORTS
    │   │       │   └── ticker_mapper.py  ┃ DRIVEN
    │   │       └── universe_build_service.py  ┃ CORE
    │   └── worker.py  ┃ DRIVING
    ├── baml_client/
    │   ├── __init__.py  ┃ DRIVEN
    │   ├── async_client.py  ┃ DRIVEN
    │   ├── config.py  ┃ DRIVEN
    │   ├── globals.py  ┃ DRIVEN
    │   ├── inlinedbaml.py  ┃ DRIVEN
    │   ├── parser.py  ┃ DRIVEN
    │   ├── runtime.py  ┃ DRIVEN
    │   ├── stream_types.py  ┃ DRIVEN
    │   ├── sync_client.py  ┃ DRIVEN
    │   ├── tracing.py  ┃ DRIVEN
    │   ├── type_builder.py  ┃ DRIVEN
    │   ├── type_map.py  ┃ DRIVEN
    │   ├── types.py  ┃ DRIVEN
    │   └── watchers.py  ┃ DRIVEN
    ├── baml_src/
    │   ├── ClassifyMacroRegime.baml  ┃ config
    │   ├── SummarizeCountryNews.baml  ┃ config
    │   ├── clients.baml  ┃ config
    │   └── generators.baml  ┃ config
    └── tests/
        ├── __init__.py  ┃ DRIVING
        ├── _fixtures/
        │   ├── __init__.py  ┃ DRIVING
        │   ├── _helpers.py  ┃ DRIVING
        │   ├── assertions.py  ┃ DRIVING
        │   ├── job_service_mock.py  ┃ DRIVING
        │   ├── macro.py  ┃ DRIVING
        │   ├── market_data.py  ┃ DRIVING
        │   └── universe.py  ┃ DRIVING
        ├── conftest.py  ┃ DRIVING
        ├── fixtures/
        │   └── smoke_prices.sql  ┃ DRIVING
        ├── integration/
        │   └── __init__.py  ┃ DRIVING
        ├── repositories/
        │   ├── __init__.py  ┃ DRIVING
        │   └── test_macro_regime_repository.py  ┃ DRIVING
        └── unit/
            ├── __init__.py  ┃ DRIVING
            ├── _shared/
            │   └── __init__.py  ┃ DRIVING
            ├── schemas/
            │   └── __init__.py  ┃ DRIVING
            ├── services/
            │   ├── __init__.py  ┃ DRIVING
            │   ├── _shared/
            │   │   ├── __init__.py  ┃ DRIVING
            │   │   ├── test_benchmark_bootstrap.py  ┃ DRIVING
            │   │   ├── test_json_safe.py  ┃ DRIVING
            │   │   ├── test_notifications_error_branch.py  ┃ DRIVING
            │   │   ├── test_price_fetcher.py  ┃ DRIVING
            │   │   ├── test_progress.py  ┃ DRIVING
            │   │   ├── test_sector_resolver.py  ┃ DRIVING
            │   │   └── test_trading_calendar.py  ┃ DRIVING
            │   ├── jobs/
            │   │   ├── __init__.py  ┃ DRIVING
            │   │   ├── test_background_job_lifecycle.py  ┃ DRIVING
            │   │   └── test_scheduler_registration.py  ┃ DRIVING
            │   ├── macro/
            │   │   ├── __init__.py  ┃ DRIVING
            │   │   ├── test_ilsole_scraper.py  ┃ DRIVING
            │   │   ├── test_macro_news_summary.py  ┃ DRIVING
            │   │   └── test_macro_regime_service.py  ┃ DRIVING
            │   ├── market_data/
            │   │   ├── __init__.py  ┃ DRIVING
            │   │   ├── test_yfinance_macro_news_service.py  ┃ DRIVING
            │   │   ├── test_yfinance_news_service.py  ┃ DRIVING
            │   │   └── test_yfinance_ticker_service.py  ┃ DRIVING
            │   ├── scenarios/
            │   │   └── __init__.py  ┃ DRIVING
            │   └── universe/
            │       ├── __init__.py  ┃ DRIVING
            │       ├── test_t212_builder.py  ┃ DRIVING
            │       ├── test_t212_cache.py  ┃ DRIVING
            │       ├── test_t212_filters.py  ┃ DRIVING
            │       ├── test_t212_mapper.py  ┃ DRIVING
            │       └── test_universe_build_service.py  ┃ DRIVING
            ├── test_background_job_liveness_columns.py  ┃ DRIVING
            ├── test_background_job_repository.py  ┃ DRIVING
            ├── test_background_job_repository_branches.py  ┃ DRIVING
            ├── test_background_job_service_heartbeat.py  ┃ DRIVING
            ├── test_background_job_service_paths.py  ┃ DRIVING
            ├── test_cli.py  ┃ DRIVING
            ├── test_database_admin_repository_branches.py  ┃ DRIVING
            ├── test_database_lazy_init.py  ┃ DRIVING
            ├── test_database_manager.py  ┃ DRIVING
            ├── test_date_parsing_branches.py  ┃ DRIVING
            ├── test_fetch_earnings_deprecation.py  ┃ DRIVING
            ├── test_fetch_fred_series_progress.py  ┃ DRIVING
            ├── test_fred_progress.py  ┃ DRIVING
            ├── test_fred_scraper_catalog.py  ┃ DRIVING
            ├── test_infrastructure_shims.py  ┃ DRIVING
            ├── test_ingestion_error_logging.py  ┃ DRIVING
            ├── test_lru_cache.py  ┃ DRIVING
            ├── test_macro_calibration.py  ┃ DRIVING
            ├── test_macro_calibration_service_branches.py  ┃ DRIVING
            ├── test_macro_news_summary_repository.py  ┃ DRIVING
            ├── test_macro_news_summary_service.py  ┃ DRIVING
            ├── test_macro_regime_repository_branches.py  ┃ DRIVING
            ├── test_macro_regime_service_branches.py  ┃ DRIVING
            ├── test_macro_regime_service_fred_key.py  ┃ DRIVING
            ├── test_market_data_service_branches.py  ┃ DRIVING
            ├── test_new_tables_migration.py  ┃ DRIVING
            ├── test_notifications.py  ┃ DRIVING
            ├── test_price_fetcher.py  ┃ DRIVING
            ├── test_reference_index_seeder.py  ┃ DRIVING
            ├── test_repository_base_branches.py  ┃ DRIVING
            ├── test_scheduler_liveness_settings.py  ┃ DRIVING
            ├── test_scheduler_pipeline.py  ┃ DRIVING
            ├── test_sector_resolver.py  ┃ DRIVING
            ├── test_seed_reference_indices_expanded_migration.py  ┃ DRIVING
            ├── test_seed_reference_indices_migration.py  ┃ DRIVING
            ├── test_sentiment_repository_branches.py  ┃ DRIVING
            ├── test_shared_infrastructure.py  ┃ DRIVING
            ├── test_trading212_client.py  ┃ DRIVING
            ├── test_trading212_filters_branches.py  ┃ DRIVING
            ├── test_trading212_mapper_builder_branches.py  ┃ DRIVING
            ├── test_tradingeconomics_scraper.py  ┃ DRIVING
            ├── test_universe_repository_branches.py  ┃ DRIVING
            ├── test_upsert_updated_at.py  ┃ DRIVING
            ├── test_worker.py  ┃ DRIVING
            ├── test_yfinance_analysis_eps.py  ┃ DRIVING
            ├── test_yfinance_cycle3_price_quality.py  ┃ DRIVING
            ├── test_yfinance_cycle4_cached_property.py  ┃ DRIVING
            ├── test_yfinance_cycle5_search_expansion.py  ┃ DRIVING
            ├── test_yfinance_data_service_branches.py  ┃ DRIVING
            ├── test_yfinance_data_service_cycle2.py  ┃ DRIVING
            ├── test_yfinance_facade_paths.py  ┃ DRIVING
            ├── test_yfinance_metadata_valuation.py  ┃ DRIVING
            ├── test_yfinance_parallel_fetch.py  ┃ DRIVING
            ├── test_yfinance_protocol_cycle3.py  ┃ DRIVING
            ├── test_yfinance_repository_branches.py  ┃ DRIVING
            ├── test_yfinance_schemas.py  ┃ DRIVING
            └── test_yfinance_timeout_and_throughput.py  ┃ DRIVING
```

> [!warning] 34 violazioni esagonali in questo contesto
> Un file `core` che raggiunge un file `driven` senza passare da un
> `ports/`. Registrate come sono: il gate deve poterle vedere.
>
> - ingestion/app/services/__init__.py (core) connects directly to ingestion/app/services/infrastructure/__init__.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/_shared/__init__.py (core) connects directly to ingestion/app/services/_shared/notifications.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/_shared/_benchmark_bootstrap.py (core) connects directly to ingestion/app/repositories/market_data/yfinance_repository.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/_shared/_benchmark_bootstrap.py (core) connects directly to ingestion/app/services/market_data/yfinance/_facade.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/_shared/_price_fetcher.py (core) connects directly to ingestion/app/repositories/market_data/yfinance_repository.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/_shared/_sector_resolver.py (core) connects directly to ingestion/app/repositories/market_data/yfinance_repository.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/jobs/background_job.py (core) connects directly to ingestion/app/repositories/jobs/background_job_repository.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/jobs/background_job.py (core) connects directly to ingestion/app/services/_shared/notifications.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/jobs/background_job.py (core) connects directly to ingestion/app/metrics.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/macro/macro_calibration.py (core) connects directly to ingestion/app/repositories/macro/macro_regime_repository.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/macro/macro_news_summary.py (core) connects directly to ingestion/app/repositories/macro/macro_regime_repository.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/macro/macro_news_summary.py (core) connects directly to ingestion/app/models/macro/macro_regime.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/macro/macro_regime_service.py (core) connects directly to ingestion/app/repositories/macro/macro_regime_repository.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/macro/macro_regime_service.py (core) connects directly to ingestion/app/services/macro/scrapers/fred_scraper.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/macro/macro_regime_service.py (core) connects directly to ingestion/app/services/macro/scrapers/ilsole_scraper.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/macro/macro_regime_service.py (core) connects directly to ingestion/app/services/macro/scrapers/tradingeconomics_scraper.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/macro/macro_regime_service.py (core) connects directly to ingestion/app/services/market_data/yfinance/_facade.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/macro/macro_regime_service.py (core) connects directly to ingestion/app/services/market_data/yfinance/news/macro_news.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/macro/macro_regime_service.py (core) connects directly to ingestion/app/services/market_data/yfinance/news/scraper.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/market_data/reference_index_seeder.py (core) connects directly to ingestion/app/models/universe/universe.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/market_data/reference_index_seeder.py (core) connects directly to ingestion/app/repositories/market_data/yfinance_repository.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/market_data/reference_index_seeder.py (core) connects directly to ingestion/app/services/market_data/yfinance/_facade.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/market_data/yfinance_data_service.py (core) connects directly to ingestion/app/repositories/market_data/yfinance_repository.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/market_data/yfinance_data_service.py (core) connects directly to ingestion/app/services/market_data/yfinance/_facade.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/market_data/yfinance_data_service.py (core) connects directly to ingestion/app/services/market_data/yfinance/news/scraper.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/universe/trading212/__init__.py (core) connects directly to ingestion/app/services/universe/trading212/client.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/universe/trading212/__init__.py (core) connects directly to ingestion/app/services/universe/trading212/ticker_mapper.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/universe/trading212/builder.py (core) connects directly to ingestion/app/services/universe/trading212/ticker_mapper.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/universe/trading212/builder.py (core) connects directly to ingestion/app/services/market_data/yfinance/_facade.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/universe/trading212/filters/historical_data.py (core) connects directly to ingestion/app/services/market_data/yfinance/_facade.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/universe/universe_build_service.py (core) connects directly to ingestion/app/repositories/universe/universe_repository.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/universe/universe_build_service.py (core) connects directly to ingestion/app/services/universe/trading212/cache/ticker_cache.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/universe/universe_build_service.py (core) connects directly to ingestion/app/services/universe/trading212/client.py (driven) — must go through a ports/ file instead
> - ingestion/app/services/universe/universe_build_service.py (core) connects directly to ingestion/app/services/universe/trading212/ticker_mapper.py (driven) — must go through a ports/ file instead

### 5.2 I file

Ingestion is the lean, headless worker ARCHITECTURE.md keeps deliberately unchanged in shape (§7: "keep the current architecture") while refining four operational details. It has no HTTP layer and no fund-style ports/ folder — its one enforced seam is the repository pattern the doc calls out explicitly ("the ingestion daemon already uses the repository pattern; that convention extends across the whole platform", L55-56) and describes in layering terms as something the service layer "Depends on ... as abstractions, so it runs against fakes in tests and real adapters in production" (L113-116). That sentence is the load-bearing justification for this spec's one deliberate deviation from ARCHITECTURE.md's terse folder-structure tags (L166-167, which label both models/ and repositories/ "DRIVEN"): here `repositories/` is classified `ports` and `baml_client/` is classified `ports` too, because both fold a stable typed interface (repository methods; `b.ClassifyMacroRegime`/`b.SummarizeCountryNews`) together with their own concrete implementation (raw SQL upserts; the Ollama HTTP call) in a single module tree, exactly the way the fund/ context's explicit ports/adapters split does with two files instead of one. With that reading, the inward chain is clean: driving (`worker.py`, `cli.py`) → core (`services/`, `schemas/`) → ports (`repositories/`, `baml_client/`) → driven (`models/`) → `external:postgres`; `baml_src/` (config) generates `baml_client/` and is never imported at runtime. `services/` does list several `external:*` systems directly (yfinance, FRED, Trading Economics, Il Sole 24 Ore, Trading 212) because ARCHITECTURE.md itself labels the whole `services/` tree — scrapers and yfinance sub-clients included — as CORE (L168); this is an acknowledged looseness in the existing ingestion codebase (unlike fund/, which explicitly separates `adapters/` from `services/`), not something this spec introduces, and the tool's own `external:*` convention treats those edges as non-violations regardless. `metrics.py` and `alembic/` sit in `config` as cross-cutting instrumentation/schema-migration wiring touched by both driving and core files. `tests/` is also `config` and is exempted from the inward-dependency check: it deliberately imports every layer (fakes/in-memory doubles) to verify the §7 daemon decisions end-to-end, which is the opposite of runtime composition.

| File | Ruolo | Contiene | Si collega a |
|---|---|---|---|
| `ingestion/alembic/README` | config | Nota a corredo della configurazione Alembic del daemon di ingestion: dichiara che si tratta di una configurazione 'generic single-database'. | — |
| `ingestion/alembic/env.py` | config | Script di bootstrap che collega il runtime delle migrazioni Alembic all'applicazione di ingestion: è qui che la fonte di verità dello schema (i modelli ORM) e la fonte di verità della connessione (i settings applicativi) vengono cablate insieme. | `ingestion/app/config.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/script.py.mako` | config | Template Mako usato da `alembic revision --autogenerate` per generare lo scheletro di ogni nuovo file in versions/. | — |
| `ingestion/alembic/versions/.gitkeep` | config | Segnaposto che mantiene la cartella versions/ sotto controllo di versione quando è vuota, con un promemoria su come generare le migrazioni reali. | — |
| `ingestion/alembic/versions/1c95f3fd3da4_remove_jsonb_columns_raw_json_and_.py` | DRIVEN | Rimuove/converte le colonne JSONB introdotte dalla migrazione yfinance precedente. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/1e14baf18033_add_server_defaults_to_timestamp_columns.py` | DRIVEN | Aggiunge server_default=now() alle colonne created_at/updated_at di exchanges e instruments. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/2a3be07fc983_drop_regime_tables.py` | DRIVEN | Rimuove regime_states e regime_state_probabilities dallo schema di ingestion. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/6b106e82ce20_increase_financial_statements_value_.py` | DRIVEN | Amplia la precisione numerica della colonna value di financial_statements. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/a1b2c3d4e5f6_rebuild_macro_tables_to_match_orm.py` | DRIVEN | Ricostruisce da zero le tabelle macro il cui schema era rimasto stale a causa del segnaposto ecf0a9a2bfdc mai completato. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres`, `external:tradingeconomics` |
| `ingestion/alembic/versions/a7b8c9d0e1f2_seed_reference_indices_expanded.py` | DRIVEN | Estende il seed dei benchmark da 1 (SPY) a 10 ETF di riferimento canonici. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/b0b30daa80c5_normalize_and_optimize_models.py` | DRIVEN | Batch di normalizzazioni sui modelli esistenti dopo l'introduzione dei dati yfinance: rinomina, vincoli, indici mancanti e precisione numerica. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/b1c2d3e4f5g6_add_regime_classification_to_macro_calibrations.py` | DRIVEN | Aggiunge una colonna che mette in cache l'output del classificatore di regime a regole, accanto alla fase prodotta dalla fase LLM. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/b2c3d4e5f6a7_drop_orphaned_tables.py` | DRIVEN | Rimuove nove tabelle che non hanno più alcun modello ORM corrispondente nel codice. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/bdd615f586f8_remove_users_table.py` | DRIVEN | Rimuove la tabella users, introdotta dalla migrazione precedente ma mai usata dal daemon di ingestion. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/c3d4e5f6a7b8_align_exchanges_with_orm.py` | DRIVEN | Allinea la tabella exchanges, che nel database aveva ancora le colonne exchange_name/exchange_id/is_active/last_updated, al modello ORM corrente (name/t212_id). | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/c51c70973022_normalize_instruments_drop_exchange_.py` | DRIVEN | Normalizza instruments rimuovendo la colonna ridondante exchange_name e aggiungendo gli indici/vincoli di unicità mancanti. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/c7d8e9f0a1b2_add_worker_liveness_columns.py` | DRIVEN | Aggiunge il substrato dati per il reaper lease-based descritto in §7.3. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/cfc1deec33d6_add_yfinance_data_tables.py` | DRIVEN | Crea le 11 tabelle che ospitano i dati scaricati da yfinance per ciascuno strumento. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres`, `external:yfinance` |
| `ingestion/alembic/versions/d1e2f3a4b5c6_drop_non_ingestion_tables.py` | DRIVEN | Materializza lo strip-to-ingestion-pipeline: rimuove le 17 tabelle non-ingestion introdotte dalle migrazioni s9t0u1v2w3x4 → z6a7b8c9d0e1. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/d4e5f6a7b8c9_add_delisting_columns_to_instruments.py` | DRIVEN | Aggiunge le colonne necessarie a correggere il survivorship bias nei dati storici di prezzo. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/e5f6a7b8c9d0_add_fred_observations_table.py` | DRIVEN | Aggiunge la tabella che ospita le osservazioni delle serie macro provenienti da FRED. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres`, `external:fred` |
| `ingestion/alembic/versions/ecf0a9a2bfdc_add_macro_regime_tables.py` | DRIVEN | Migrazione segnaposto generata per le tabelle macro/regime, mai completata. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/f6a7b8c9d0e1_drop_ilsole_real_columns.py` | DRIVEN | Rimuove la fonte dati 'ilsole_real', ridondante rispetto a TradingEconomics. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/ff232ce0534f_initial_schema_users_exchanges_.py` | DRIVEN | Prima migrazione della storia: crea lo schema iniziale con le tabelle users, exchanges e instruments. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/g7b8c9d0e1f2_drop_st_rate_forecast.py` | DRIVEN | Rimuove la colonna st_rate_forecast, sempre NULL perché la fonte dati non la pubblica mai. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/h8c9d0e1f2g3_add_macro_observation_tables.py` | DRIVEN | Introduce lo schema a serie temporale (una riga per osservazione) per gli indicatori TradingEconomics e i rendimenti obbligazionari, in sostituzione dello schema a singolo snapshot. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres`, `external:tradingeconomics` |
| `ingestion/alembic/versions/i9d0e1f2g3h4_add_econ_indicator_observations.py` | DRIVEN | Replica lo stesso pattern a serie temporale per gli indicatori economici derivati. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/j0e1f2g3h4i5_add_macro_news_table.py` | DRIVEN | Introduce la tabella che ospita le notizie macro raccolte per paese/tema. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/k1f2g3h4i5j6_add_macro_news_full_content.py` | DRIVEN | Aggiunge la colonna che ospita il testo integrale di una notizia macro. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/l2g3h4i5j6k7_add_ticker_news_full_content.py` | DRIVEN | Applica lo stesso pattern di k1f2g3h4i5j6 alle notizie a livello di singolo ticker. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/m3h4i5j6k7l8_drop_ticker_news_related_tickers.py` | DRIVEN | Rimuove definitivamente una colonna di ticker_news giudicata inutile. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/n4i5j6k7l8m9_add_ticker_news_ticker_name.py` | DRIVEN | Denormalizza il nome dello strumento dentro ticker_news per evitare join ripetuti in lettura. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/o5j6k7l8m9n0_add_composite_indexes_on_observation_tables.py` | DRIVEN | Aggiunge indici compositi mancanti sulle due tabelle di osservazioni introdotte da h8c9d0e1f2g3. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/p6q7r8s9t0u1_add_macro_calibrations_table.py` | DRIVEN | Introduce la tabella che mette in cache l'output del classificatore di regime macro basato su LLM. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres`, `external:ollama` |
| `ingestion/alembic/versions/q7r8s9t0u1v2_add_macro_news_summaries_table.py` | DRIVEN | Introduce la tabella dei riassunti giornalieri per paese delle notizie macro. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres`, `external:ollama` |
| `ingestion/alembic/versions/r8s9t0u1v2w3_add_background_jobs_table.py` | DRIVEN | Introduce la tabella che traccia in modo persistente i job pianificati dal daemon di ingestion. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/s9t0u1v2w3x4_add_portfolio_tables.py` | DRIVEN | Introduce lo schema dei portafogli, primo passo verso quella che era allora un'API più ampia, poi stralciata. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/t0u1v2w3x4y5_normalize_json_columns.py` | DRIVEN | Normalizza otto colonne JSONB delle tabelle portfolio/regime/job in tabelle figlie chiave-valore. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/u1v2w3x4y5z6_add_currency_code_to_financial_statements.py` | DRIVEN | Aggiunge la valuta di denominazione ai valori di bilancio, per audit e normalizzazione multi-valuta. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/v2w3x4y5z6a7_add_external_id_to_activity_events.py` | DRIVEN | Aggiunge la deduplicazione degli ordini broker sulle activity_events, requisito di scrittura idempotente. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/w3x4y5z6a7b8_add_snapshot_date_to_broker_account_snapshots.py` | DRIVEN | Risolve la crescita illimitata di broker_account_snapshots causata da sincronizzazioni ripetute nello stesso giorno. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/x4y5z6a7b8c9_seed_reference_indices.py` | DRIVEN | Semina lo strumento di riferimento SPY, necessario perché i benchmark non passano dalla pipeline dell'universo Trading 212 (che filtra su type=='STOCK' ed escluderebbe gli ETF). | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/y5z6a7b8c9d0_add_factor_scores_execution_runs_risk_limits_rebalancing_policies_tables.py` | DRIVEN | Aggiunge lo schema per fattori, run di ottimizzazione/backtest, limiti di rischio e policy di ribilanciamento — ulteriore estensione dell'API pre-strip. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/alembic/versions/z6a7b8c9d0e1_add_api_keys_table.py` | DRIVEN | Aggiunge la tabella per l'autenticazione tramite chiave API, altro residuo dell'era HTTP del repo. | `ingestion/alembic/env.py`, `ingestion/app/models/__init__.py`, `external:postgres` |
| `ingestion/app/cli.py` | DRIVING | Interfaccia a riga di comando per l'esecuzione manuale della pipeline di ingestion (`python -m app.cli <comando>`), costruita con Typer: un comando per pipeline composite (`daily`, `refetch-all`) e un comando per ciascuno step singolo (`universe`, `yfinance`, `macro`, `fred`, `news`, `summarize`, `calibrate`, `reference-indices`). | `ingestion/app/config.py`, `ingestion/app/database.py`, `ingestion/app/services/jobs/scheduler.py` |
| `ingestion/app/metrics.py` | DRIVEN | Definizioni dei contatori, istogramma e gauge Prometheus con cui il daemon di ingestion espone lo stato dei propri job in background: job avviati, completati, falliti (Counter per `domain`), durata run-to-terminal (Histogram con bucket da 10s a 3600s) e job correntemente in esecuzione (Gauge). | `ingestion/app/worker.py`, `ingestion/app/services/jobs/background_job_service.py`, `external:prometheus` |
| `ingestion/app/models/__init__.py` | DRIVEN | Superficie pubblica del pacchetto ORM dell'ingestion: raccoglie in un unico namespace tutte le classi mappate definite nei sottopacchetti di dominio. | `ingestion/app/models/_shared/base.py`, `ingestion/app/models/jobs/background_job.py`, `ingestion/app/models/macro/macro_regime.py`, `ingestion/app/models/market_data/yfinance_data.py`, `ingestion/app/models/universe/universe.py` |
| `ingestion/app/models/_shared/__init__.py` | DRIVEN | Superficie pubblica del sottopacchetto _shared, che raccoglie le classi base e i mixin trasversali usati da ogni modello di dominio dell'ingestion. | `ingestion/app/models/_shared/base.py` |
| `ingestion/app/models/_shared/base.py` | DRIVEN | Fondamenta comuni dell'object-relational mapping: la classe dichiarativa di base e i mixin che ogni tabella dell'ingestion eredita. | `external:postgres` |
| `ingestion/app/models/jobs/__init__.py` | DRIVEN | Superficie pubblica del sottopacchetto jobs, che raccoglie i modelli di tracciamento dei job in background. | `ingestion/app/models/jobs/background_job.py` |
| `ingestion/app/models/jobs/background_job.py` | DRIVEN | Persistenza dello stato dei job schedulati dal daemon di ingestion, in sostituzione di uno stato tenuto solo in memoria di processo. | `ingestion/app/models/_shared/base.py`, `external:postgres` |
| `ingestion/app/models/macro/__init__.py` | DRIVEN | Superficie pubblica del sottopacchetto macro, che raccoglie i modelli dei dati macroeconomici ingeriti. | `ingestion/app/models/macro/macro_regime.py` |
| `ingestion/app/models/macro/macro_regime.py` | DRIVEN | Persistenza dei dati macroeconomici e creditizi grezzi — previsioni, curva dei tassi, spread e news — che alimentano la lettura del regime di mercato a valle. | `ingestion/app/models/_shared/base.py`, `external:postgres` |
| `ingestion/app/models/market_data/__init__.py` | DRIVEN | Superficie pubblica del sottopacchetto market_data, che raccoglie i modelli dei dati di mercato e fondamentali scaricati da yfinance. | `ingestion/app/models/market_data/yfinance_data.py` |
| `ingestion/app/models/market_data/yfinance_data.py` | DRIVEN | Persistenza dei dati grezzi di mercato e di bilancio per singolo strumento — profilo, prezzi, bilanci, dividendi, analisti, azionariato, news — che costituiscono la materia prima della selezione su fondamentali. | `ingestion/app/models/_shared/base.py`, `ingestion/app/models/universe/universe.py`, `external:postgres` |
| `ingestion/app/models/universe/__init__.py` | DRIVEN | Superficie pubblica del sottopacchetto universe, che raccoglie i modelli dell'universo investibile. | `ingestion/app/models/universe/universe.py` |
| `ingestion/app/models/universe/universe.py` | DRIVEN | Persistenza dell'universo investibile: le borse e gli strumenti che sono il punto di partenza di ogni restrizione successiva basata su fondamentali. | `ingestion/app/models/_shared/base.py`, `ingestion/app/models/market_data/yfinance_data.py`, `external:postgres` |
| `ingestion/app/repositories/__init__.py` | DRIVEN | Punto di ingresso pubblico del package repositories: ri-esporta le classi repository di ogni dominio e la base condivisa. | `ingestion/app/repositories/_shared/base.py`, `ingestion/app/repositories/jobs/background_job_repository.py`, `ingestion/app/repositories/macro/macro_regime_repository.py`, `ingestion/app/repositories/macro/sentiment_repository.py`, `ingestion/app/repositories/market_data/yfinance_repository.py`, `ingestion/app/repositories/universe/universe_repository.py` |
| `ingestion/app/repositories/_shared/__init__.py` | DRIVEN | Punto di ingresso del sotto-package `_shared`: ri-esporta l'infrastruttura repository trasversale usata da tutti i repository di dominio. | `ingestion/app/repositories/_shared/base.py`, `ingestion/app/repositories/_shared/database_admin_repository.py` |
| `ingestion/app/repositories/_shared/base.py` | DRIVEN | Definisce la base comune di tutti i repository: la funzione di supporto per estrarre la `Table` SQLAlchemy da un modello mappato, la classe minimale `RepositoryBase` con l'upsert condiviso, e il generico `BaseRepository[Model, Create, Update]` con CRUD sincrono completo. | `external:postgres` |
| `ingestion/app/repositories/_shared/database_admin_repository.py` | DRIVEN | Repository di amministrazione del database: health check, introspezione delle tabelle gestite dall'applicazione e troncamento controllato, con un allowlist esplicito (`APP_TABLES`) come confine di sicurezza prima di ogni DDL. | `ingestion/app/repositories/_shared/base.py`, `external:postgres` |
| `ingestion/app/repositories/jobs/__init__.py` | DRIVEN | Punto di ingresso del sotto-package `jobs`: ri-esporta il repository dei job in background. | `ingestion/app/repositories/jobs/background_job_repository.py` |
| `ingestion/app/repositories/jobs/background_job_repository.py` | DRIVEN | Repository per la tabella `background_jobs`: gestisce il ciclo di vita di un job schedulato (creazione, claim atomico, heartbeat, aggiornamento con guardia sullo stato, riconciliazione degli orfani) — è il substrato dati dietro il reaper e lo scheduler descritti in §7. | `external:postgres` |
| `ingestion/app/repositories/macro/__init__.py` | DRIVEN | Punto di ingresso del sotto-package `macro`: ri-esporta i repository dei dati macroeconomici e del sentiment. | `ingestion/app/repositories/macro/macro_regime_repository.py`, `ingestion/app/repositories/macro/sentiment_repository.py` |
| `ingestion/app/repositories/macro/macro_regime_repository.py` | DRIVEN | Repository per tutti i dati macro ingeriti da fonti esterne (indicatori economici IlSole24Ore, Trading Economics, rendimenti obbligazionari, osservazioni FRED, notizie macro e i loro riassunti, calibrazioni regime): sette famiglie di dati, ciascuna con upsert puntuale/batch e query di lettura filtrate. | `external:postgres` |
| `ingestion/app/repositories/macro/sentiment_repository.py` | DRIVEN | Repository di sola lettura che risolve un ticker nella sua notizia più recente: prima le notizie dirette sullo strumento, poi — a cascata — le notizie macro rilevanti per settore o per corrispondenza testuale, tramite una mappa statica ETF-di-settore → nome settore GICS. | `external:postgres` |
| `ingestion/app/repositories/market_data/__init__.py` | DRIVEN | Punto di ingresso del sotto-package `market_data`: ri-esporta il repository dei dati yfinance. | `ingestion/app/repositories/market_data/yfinance_repository.py` |
| `ingestion/app/repositories/market_data/yfinance_repository.py` | DRIVEN | Il repository più grande del package: converte le strutture dati eterogenee restituite da yfinance (dict, `pd.Series`, `pd.DataFrame`) in righe tipizzate e le upserta su undici tabelle — profilo, prezzi OHLCV storici, bilanci (formato EAV), dividendi, split, raccomandazioni e target price degli analisti, holder istituzionali e di fondi comuni, transazioni insider, notizie sul ticker. | `external:postgres`, `external:yfinance` |
| `ingestion/app/repositories/universe/__init__.py` | DRIVEN | Punto di ingresso del sotto-package `universe`: ri-esporta il repository dell'universo investibile. | `ingestion/app/repositories/universe/universe_repository.py` |
| `ingestion/app/repositories/universe/universe_repository.py` | DRIVEN | Repository per il dominio universo investibile: gli exchange e gli strumenti (con il loro stato di delisting) provenienti da Trading 212. | `external:postgres`, `external:trading212` |
| `ingestion/app/schemas/__init__.py` | DRIVING | Docstring di pacchetto per schemas/: spiega che questi Pydantic model erano corpi di richiesta/risposta HTTP e ora, con l'HTTP rimosso, sono gli oggetti argomento tipizzati passati a CLI/scheduler e i payload di progresso che i servizi restituiscono. | — |
| `ingestion/app/schemas/_shared/__init__.py` | DRIVING | Punto di raccolta dei modelli Pydantic trasversali riusati da tutti i sotto-domini di schemas/. | `ingestion/app/schemas/_shared/base.py`, `ingestion/app/schemas/_shared/base_job.py` |
| `ingestion/app/schemas/_shared/base.py` | DRIVING | Base comuni di serializzazione per i modelli di risposta: la classe che forza camelCase in uscita verso il frontend e l'adattatore di tipo che normalizza gli UUID a stringa. | — |
| `ingestion/app/schemas/_shared/base_job.py` | DRIVING | Schemi base per i job asincroni in background dell'ingestion (fetch, calibrazione, riepilogo notizie): la risposta di creazione job e il payload di avanzamento. | — |
| `ingestion/app/schemas/macro/__init__.py` | DRIVING | Punto di ri-esportazione del sotto-pacchetto macro/: raccoglie tutti gli schemi Pydantic relativi a indicatori economici, rendimenti obbligazionari, serie FRED, notizie macro e calibrazione Black-Litterman. | `ingestion/app/schemas/macro/macro_regime.py` |
| `ingestion/app/schemas/macro/macro_regime.py` | DRIVING | Tipi tipizzati per l'intero ciclo dati macro dell'ingestion: richieste di fetch (FRED, indicatori, notizie), payload di progresso dei relativi job in background, modelli di lettura degli indicatori economici/obbligazionari/FRED e la risposta di calibrazione macro-Black-Litterman prodotta da un LLM. | `ingestion/app/schemas/_shared/base.py`, `ingestion/app/schemas/_shared/base_job.py` |
| `ingestion/app/schemas/market_data/__init__.py` | DRIVING | Punto di ri-esportazione del sotto-pacchetto market_data/: raccoglie gli schemi relativi al fetch e alla lettura dei dati yfinance e agli indici di riferimento. | `ingestion/app/schemas/market_data/reference_index.py`, `ingestion/app/schemas/market_data/yfinance_data.py` |
| `ingestion/app/schemas/market_data/reference_index.py` | DRIVING | Tipi per il seeding e la diagnostica di copertura degli indici di riferimento (benchmark) usati dall'ingestion. | `ingestion/app/schemas/_shared/base.py`, `ingestion/app/schemas/_shared/base_job.py` |
| `ingestion/app/schemas/market_data/yfinance_data.py` | DRIVING | Tipi per il fetch massivo/singolo da yfinance e per la lettura di ogni tabella di dati fondamentali/di mercato che l'ingestion popola: profilo ticker, storico prezzi, bilanci, dividendi, split, raccomandazioni analisti, target di prezzo, holder istituzionali/mutual fund, transazioni insider, notizie. | `ingestion/app/schemas/_shared/base_job.py` |
| `ingestion/app/schemas/universe/__init__.py` | DRIVING | Punto di ri-esportazione del sotto-pacchetto universe/: raccoglie gli schemi relativi alla costruzione e lettura dell'universo Trading212. | `ingestion/app/schemas/universe/trading212.py` |
| `ingestion/app/schemas/universe/trading212.py` | DRIVING | Tipi per la costruzione dell'universo investibile da Trading212: richiesta di build, progresso del job, risultato aggregato e modelli di lettura di scambi e strumenti. | `ingestion/app/schemas/_shared/base_job.py` |
| `ingestion/app/services/__init__.py` | CORE | Docstring di pacchetto che elenca i sei sotto-pacchetti dei servizi di ingestion e la responsabilità di ciascuno. | `ingestion/app/services/market_data/__init__.py`, `ingestion/app/services/macro/__init__.py`, `ingestion/app/services/universe/__init__.py`, `ingestion/app/services/jobs/__init__.py`, `ingestion/app/services/infrastructure/__init__.py`, `ingestion/app/services/_shared/__init__.py` |
| `ingestion/app/services/_shared/__init__.py` | CORE | Aggregatore che ri-esporta gli helper cross-cutting usati da più domini di ingestion (progress, prezzi, settori, calendario, webhook). | `ingestion/app/services/_shared/_json_safe.py`, `ingestion/app/services/_shared/_price_fetcher.py`, `ingestion/app/services/_shared/_progress.py`, `ingestion/app/services/_shared/_sector_resolver.py`, `ingestion/app/services/_shared/notifications.py`, `ingestion/app/services/_shared/trading_calendar.py` |
| `ingestion/app/services/_shared/_benchmark_bootstrap.py` | CORE | Use-case di startup che garantisce che ogni ticker di `settings.benchmark_tickers` abbia una riga Instrument e storico prezzi recente. | `ingestion/app/repositories/market_data/yfinance_repository.py`, `ingestion/app/services/market_data/reference_index_seeder.py`, `ingestion/app/services/market_data/yfinance/_facade.py` |
| `ingestion/app/services/_shared/_json_safe.py` | CORE | Utility pura per coercizione di float non finiti (NaN/Inf) a `None` prima della persistenza JSONB. | — |
| `ingestion/app/services/_shared/_price_fetcher.py` | CORE | Use-case che legge i prezzi di chiusura per una lista di ticker dalla tabella `price_history`. | `ingestion/app/repositories/market_data/yfinance_repository.py` |
| `ingestion/app/services/_shared/_progress.py` | CORE | Costruttori di closure per il callback di progresso condiviso da tutti i job in background. | `ingestion/app/services/jobs/background_job.py` |
| `ingestion/app/services/_shared/_sector_resolver.py` | CORE | Unifica le due fonti autoritative di settore ticker→settore (snapshot dell'ottimizzatore e tabella TickerProfile) dietro un'unica funzione. | `ingestion/app/repositories/market_data/yfinance_repository.py` |
| `ingestion/app/services/_shared/notifications.py` | DRIVEN | Adapter driven verso un webhook compatibile Discord/Slack per notificare i fallimenti di pipeline. | `external:discord-slack-webhook` |
| `ingestion/app/services/_shared/trading_calendar.py` | CORE | Calcola il numero atteso di sessioni di borsa per una borsa e un periodo, per validare la completezza dello storico prezzi scaricato. | — |
| `ingestion/app/services/infrastructure/__init__.py` | DRIVEN | Aggregatore delle primitive di resilienza condivise da ogni client verso un servizio esterno. | `ingestion/app/services/infrastructure/cache.py`, `ingestion/app/services/infrastructure/circuit_breaker.py`, `ingestion/app/services/infrastructure/rate_limiter.py`, `ingestion/app/services/infrastructure/retry.py` |
| `ingestion/app/services/infrastructure/cache.py` | DRIVEN | Cache LRU thread-safe con TTL opzionale per oggetti costosi da ricostruire (es. `yf.Ticker`). | — |
| `ingestion/app/services/infrastructure/circuit_breaker.py` | DRIVEN | Interruttore a soglia con backoff esponenziale e auto-guarigione per proteggere un servizio esterno da retry aggressivi. | — |
| `ingestion/app/services/infrastructure/rate_limiter.py` | DRIVEN | Limitatore di frequenza per-chiave a intervallo fisso, thread-safe. | — |
| `ingestion/app/services/infrastructure/retry.py` | DRIVEN | Retry generico con backoff esponenziale a full-jitter e classificazione degli errori di rete transienti. | — |
| `ingestion/app/services/jobs/__init__.py` | CORE | Aggregatore che espone `BackgroundJobService`, `JobAlreadyRunningError` e `create_scheduler`. | `ingestion/app/services/jobs/background_job.py`, `ingestion/app/services/jobs/scheduler.py` |
| `ingestion/app/services/jobs/background_job.py` | CORE | Use-case di ciclo di vita dei job in background con persistenza Postgres, heartbeat lease-based e metriche. | `ingestion/app/repositories/jobs/background_job_repository.py`, `ingestion/app/services/_shared/notifications.py`, `ingestion/app/config.py`, `ingestion/app/metrics.py` |
| `ingestion/app/services/jobs/scheduler.py` | DRIVING | Cablaggio APScheduler della pipeline giornaliera dell'ingestion: definisce ogni cron/interval job e la sequenza delle sue tappe. | `ingestion/app/services/macro/macro_calibration.py`, `ingestion/app/services/macro/macro_news_summary.py`, `ingestion/app/services/macro/macro_regime_service.py`, `ingestion/app/services/market_data/yfinance_data_service.py`, `ingestion/app/services/market_data/reference_index_seeder.py`, `ingestion/app/services/universe/universe_build_service.py`, `ingestion/app/services/jobs/background_job.py`, `ingestion/app/services/_shared/_progress.py`, `ingestion/app/database.py` |
| `ingestion/app/services/macro/__init__.py` | CORE | Aggregatore che espone le tre use-case di macro (calibrazione, sintesi notizie, fetch regime) del dominio macro. | `ingestion/app/services/macro/macro_calibration.py`, `ingestion/app/services/macro/macro_news_summary.py`, `ingestion/app/services/macro/macro_regime_service.py` |
| `ingestion/app/services/macro/macro_calibration.py` | CORE | Use-case che classifica la fase del ciclo economico e calibra i parametri Black-Litterman (delta, tau) per paese, via LLM. | `ingestion/app/repositories/macro/macro_regime_repository.py`, `external:ollama-baml`, `ingestion/app/services/_shared/_progress.py` |
| `ingestion/app/services/macro/macro_news_summary.py` | CORE | Use-case che mappa articoli macro a paesi e li sintetizza in un riepilogo con sentiment via LLM, con refresh incrementale. | `ingestion/app/repositories/macro/macro_regime_repository.py`, `external:ollama-baml`, `ingestion/app/models/macro/macro_regime.py` |
| `ingestion/app/services/macro/macro_regime_service.py` | CORE | Use-case di orchestrazione che invoca gli scraper macro (IlSole, Trading Economics, FRED, news yfinance) e persiste i risultati via repository, con isolamento errori per paese/fonte. | `ingestion/app/repositories/macro/macro_regime_repository.py`, `ingestion/app/services/macro/scrapers/fred_scraper.py`, `ingestion/app/services/macro/scrapers/ilsole_scraper.py`, `ingestion/app/services/macro/scrapers/tradingeconomics_scraper.py`, `ingestion/app/services/market_data/yfinance/_facade.py`, `ingestion/app/services/market_data/yfinance/news/macro_news.py`, `ingestion/app/services/market_data/yfinance/news/scraper.py` |
| `ingestion/app/services/macro/scrapers/__init__.py` | DRIVEN | Aggregatore dei tre scraper macro e dell'eccezione di parsing condivisa, più la lista `PORTFOLIO_COUNTRIES`. | `ingestion/app/services/macro/scrapers/exceptions.py`, `ingestion/app/services/macro/scrapers/fred_scraper.py`, `ingestion/app/services/macro/scrapers/ilsole_scraper.py`, `ingestion/app/services/macro/scrapers/tradingeconomics_scraper.py` |
| `ingestion/app/services/macro/scrapers/exceptions.py` | DRIVEN | Eccezione di dominio degli scraper: segnala che l'HTML remoto è cambiato struttura, non che il dato è legittimamente vuoto. | — |
| `ingestion/app/services/macro/scrapers/fred_scraper.py` | DRIVEN | Client REST verso l'API FRED per serie di spread di credito/tasso, volatilità, indicatori di ciclo e recessione. | `ingestion/app/services/infrastructure/circuit_breaker.py`, `ingestion/app/services/infrastructure/retry.py`, `external:FRED` |
| `ingestion/app/services/macro/scrapers/ilsole_scraper.py` | DRIVEN | Scraper HTML del sito Il Sole 24 Ore per indicatori reali e previsioni consenso per paese. | `ingestion/app/services/infrastructure/circuit_breaker.py`, `ingestion/app/services/infrastructure/rate_limiter.py`, `ingestion/app/services/infrastructure/retry.py`, `external:ilsole24ore` |
| `ingestion/app/services/macro/scrapers/tradingeconomics_scraper.py` | DRIVEN | Scraper HTML di Trading Economics per indicatori macro, bond yield, produzione industriale e capacity utilization, con matching per nome indicatore. | `ingestion/app/services/infrastructure/circuit_breaker.py`, `ingestion/app/services/infrastructure/rate_limiter.py`, `ingestion/app/services/infrastructure/retry.py`, `ingestion/app/services/macro/scrapers/exceptions.py`, `external:tradingeconomics` |
| `ingestion/app/services/market_data/__init__.py` | CORE | Aggregatore delle due use-case di market data: seeding indici di riferimento e fetch bulk yfinance. | `ingestion/app/services/market_data/reference_index_seeder.py`, `ingestion/app/services/market_data/yfinance_data_service.py` |
| `ingestion/app/services/market_data/reference_index_seeder.py` | CORE | Use-case che crea idempotentemente le righe Instrument per i benchmark (es. SPY) e ne scarica lo storico prezzi, perché non sono membri di alcun universo T212 tradabile. | `ingestion/app/models/universe/universe.py`, `ingestion/app/repositories/market_data/yfinance_repository.py`, `ingestion/app/services/_shared/_progress.py`, `ingestion/app/services/market_data/yfinance/_facade.py`, `ingestion/app/services/market_data/yfinance_data_service.py` |
| `ingestion/app/services/market_data/yfinance/__init__.py` | DRIVEN | Aggregatore pubblico del client yfinance completo: base, facade, infrastruttura, market, news, protocolli, ticker. | `ingestion/app/services/market_data/yfinance/_base.py`, `ingestion/app/services/market_data/yfinance/_facade.py`, `ingestion/app/services/market_data/yfinance/infrastructure/__init__.py`, `ingestion/app/services/market_data/yfinance/market/__init__.py`, `ingestion/app/services/market_data/yfinance/news/__init__.py`, `ingestion/app/services/market_data/yfinance/protocols/__init__.py`, `ingestion/app/services/market_data/yfinance/ticker/__init__.py` |
| `ingestion/app/services/market_data/yfinance/_base.py` | DRIVEN | Classe base condivisa dai sub-client ticker-scoped: cache del `yf.Ticker`, rate limiter e circuit breaker in comune. | `ingestion/app/services/market_data/yfinance/infrastructure/retry.py`, `ingestion/app/services/market_data/yfinance/protocols/interfaces.py`, `external:yfinance` |
| `ingestion/app/services/market_data/yfinance/_facade.py` | DRIVEN | Client yfinance singleton: cache/rate-limiter/circuit-breaker condivisi, fetch di info/storico/bulk-download, e facciata lazy verso i sub-client per statement, analisi, holder, corporate action, metadata, ricerca. | `ingestion/app/services/market_data/yfinance/infrastructure/__init__.py`, `ingestion/app/services/market_data/yfinance/market/search.py`, `ingestion/app/services/market_data/yfinance/protocols/interfaces.py`, `ingestion/app/services/market_data/yfinance/ticker/__init__.py`, `external:yfinance` |
| `ingestion/app/services/market_data/yfinance/infrastructure/__init__.py` | DRIVEN | Aggregatore locale che ri-espone le primitive di resilienza condivise sotto il namespace yfinance. | `ingestion/app/services/market_data/yfinance/infrastructure/cache.py`, `ingestion/app/services/market_data/yfinance/infrastructure/circuit_breaker.py`, `ingestion/app/services/market_data/yfinance/infrastructure/rate_limiter.py`, `ingestion/app/services/market_data/yfinance/infrastructure/retry.py` |
| `ingestion/app/services/market_data/yfinance/infrastructure/cache.py` | DRIVEN | Re-export puro di `LRUCache` dall'infrastruttura condivisa, per mantenere stabili i percorsi di import di yfinance. | `ingestion/app/services/infrastructure/cache.py` |
| `ingestion/app/services/market_data/yfinance/infrastructure/circuit_breaker.py` | DRIVEN | Re-export puro di `CircuitBreaker` dall'infrastruttura condivisa. | `ingestion/app/services/infrastructure/circuit_breaker.py` |
| `ingestion/app/services/market_data/yfinance/infrastructure/rate_limiter.py` | DRIVEN | Re-export puro di `RateLimiter` dall'infrastruttura condivisa. | `ingestion/app/services/infrastructure/rate_limiter.py` |
| `ingestion/app/services/market_data/yfinance/infrastructure/retry.py` | DRIVEN | Wrapper yfinance-specifico su `retry_with_backoff`/`is_transient_network_error` condivisi, con l'alias storico `is_rate_limit_error`. | `ingestion/app/services/infrastructure/retry.py` |
| `ingestion/app/services/market_data/yfinance/market/__init__.py` | DRIVEN | Aggregatore dei client module-level (non ticker-scoped) di yfinance. | `ingestion/app/services/market_data/yfinance/market/search.py` |
| `ingestion/app/services/market_data/yfinance/market/search.py` | DRIVEN | Sub-client per ricerca e lookup via `yf.Search`/`yf.Lookup`. | `ingestion/app/services/market_data/yfinance/infrastructure/retry.py`, `ingestion/app/services/market_data/yfinance/protocols/interfaces.py`, `external:yfinance` |
| `ingestion/app/services/market_data/yfinance/news/__init__.py` | DRIVEN | Aggregatore del sotto-pacchetto news: fetch, scraping articolo, aggregazione per paese, fetch macro-tematico. | `ingestion/app/services/market_data/yfinance/news/aggregator.py`, `ingestion/app/services/market_data/yfinance/news/client.py`, `ingestion/app/services/market_data/yfinance/news/macro_news.py`, `ingestion/app/services/market_data/yfinance/news/scraper.py` |
| `ingestion/app/services/market_data/yfinance/news/aggregator.py` | DRIVEN | Aggrega, deduplica e filtra per recency le news finanziarie per paese, da un set fisso di ticker indice. | `ingestion/app/services/market_data/yfinance/protocols/interfaces.py`, `ingestion/app/services/market_data/yfinance/news/client.py` |
| `ingestion/app/services/market_data/yfinance/news/client.py` | DRIVEN | Client di basso livello per le news di un singolo ticker via `yf.Ticker.news`, con arricchimento opzionale di full-content. | `ingestion/app/services/market_data/yfinance/infrastructure/retry.py`, `ingestion/app/services/market_data/yfinance/protocols/interfaces.py`, `ingestion/app/services/market_data/yfinance/news/scraper.py` |
| `ingestion/app/services/market_data/yfinance/news/macro_news.py` | DRIVEN | Raccoglie news macro-tematiche da ticker fissi (tassi, VIX, commodity, FX, ETF settoriali) e da query di ricerca mirate, le classifica per tema e le cura per densità di segnale. | `ingestion/app/services/market_data/yfinance/protocols/interfaces.py`, `ingestion/app/services/market_data/yfinance/news/aggregator.py`, `ingestion/app/services/market_data/yfinance/news/client.py` |
| `ingestion/app/services/market_data/yfinance/news/scraper.py` | DRIVEN | Scraper HTTP generico che estrae il testo integrale di un articolo da un URL arbitrario, provando una lista ordinata di selettori CSS comuni. | `external:news-sites` |
| `ingestion/app/services/market_data/yfinance/protocols/__init__.py` | PORTS | Aggregatore dei protocolli (interfacce) che i sub-client yfinance implementano. | `ingestion/app/services/market_data/yfinance/protocols/interfaces.py` |
| `ingestion/app/services/market_data/yfinance/protocols/interfaces.py` | PORTS | Dichiara i protocolli (ISP) che disaccoppiano `_base.py`, `_facade.py` e i sub-client dall'implementazione concreta di cache, rate limiter, circuit breaker, scraper articoli e client yfinance/statement/analisi/holder/azioni-societarie/metadata/ricerca. | — |
| `ingestion/app/services/market_data/yfinance/ticker/__init__.py` | DRIVEN | Aggregatore dei cinque sub-client ticker-scoped che estendono `BaseClient`. | `ingestion/app/services/market_data/yfinance/ticker/analysis.py`, `ingestion/app/services/market_data/yfinance/ticker/corporate_actions.py`, `ingestion/app/services/market_data/yfinance/ticker/financials.py`, `ingestion/app/services/market_data/yfinance/ticker/holders.py`, `ingestion/app/services/market_data/yfinance/ticker/metadata.py` |
| `ingestion/app/services/market_data/yfinance/ticker/analysis.py` | DRIVEN | Sub-client per raccomandazioni analista, stime utili/ricavi, sustainability, trend/revisioni EPS. | `ingestion/app/services/market_data/yfinance/_base.py`, `external:yfinance` |
| `ingestion/app/services/market_data/yfinance/ticker/corporate_actions.py` | DRIVEN | Sub-client per dividendi, split, azioni societarie, capital gain e storico azioni in circolazione. | `ingestion/app/services/market_data/yfinance/_base.py`, `external:yfinance` |
| `ingestion/app/services/market_data/yfinance/ticker/financials.py` | DRIVEN | Sub-client per conto economico, stato patrimoniale, flussi di cassa (annuali e trimestrali) e SEC filings. | `ingestion/app/services/market_data/yfinance/_base.py`, `external:yfinance` |
| `ingestion/app/services/market_data/yfinance/ticker/holders.py` | DRIVEN | Sub-client per holder maggiori, istituzionali, fondi comuni e transazioni/acquisti insider. | `ingestion/app/services/market_data/yfinance/_base.py`, `external:yfinance` |
| `ingestion/app/services/market_data/yfinance/ticker/metadata.py` | DRIVEN | Sub-client per ISIN, fast_info, calendario earnings, catena opzioni e il pannello di 9 metriche di valutazione. | `ingestion/app/services/market_data/yfinance/_base.py`, `external:yfinance` |
| `ingestion/app/services/market_data/yfinance_data_service.py` | CORE | Use-case centrale che orchestra il fetch di ogni categoria dato yfinance (profilo, prezzi, bilanci, dividendi/split, raccomandazioni, target, holder, insider, news) per un ticker e la relativa persistenza incrementale. | `ingestion/app/repositories/market_data/yfinance_repository.py`, `ingestion/app/services/market_data/yfinance/_facade.py`, `ingestion/app/services/_shared/_progress.py`, `ingestion/app/services/_shared/trading_calendar.py`, `ingestion/app/utils/currency.py`, `ingestion/app/services/market_data/yfinance/news/scraper.py`, `ingestion/app/config.py` |
| `ingestion/app/services/universe/__init__.py` | CORE | Docstring che indirizza a importare direttamente da `app.services.universe.trading212`. | `ingestion/app/services/universe/trading212/__init__.py` |
| `ingestion/app/services/universe/trading212/__init__.py` | CORE | Aggregatore del sotto-pacchetto trading212: builder, client, config, ticker mapper. | `ingestion/app/services/universe/trading212/builder.py`, `ingestion/app/services/universe/trading212/client.py`, `ingestion/app/services/universe/trading212/config.py`, `ingestion/app/services/universe/trading212/ticker_mapper.py` |
| `ingestion/app/services/universe/trading212/builder.py` | CORE | Use-case che costruisce l'universo strumenti: fetch metadata T212, mappatura ticker yfinance, applicazione della pipeline filtri, persistenza a batch e rilevamento delisting. | `ingestion/app/services/universe/trading212/config.py`, `ingestion/app/services/universe/trading212/protocols.py`, `ingestion/app/services/universe/trading212/ticker_mapper.py`, `ingestion/app/services/market_data/yfinance/_facade.py` |
| `ingestion/app/services/universe/trading212/cache/__init__.py` | DRIVEN | Aggregatore che espone `TickerMappingCache`. | `ingestion/app/services/universe/trading212/cache/ticker_cache.py` |
| `ingestion/app/services/universe/trading212/cache/ticker_cache.py` | DRIVEN | Cache in-memoria (symbol, exchange) → ticker yfinance, con una vita pari a un singolo job di build. | — |
| `ingestion/app/services/universe/trading212/client.py` | DRIVEN | Client REST verso l'API Trading 212 (metadati exchange/strumenti, cash/info account, posizioni, storico ordini e dividendi paginato). | `ingestion/app/config.py`, `external:trading212` |
| `ingestion/app/services/universe/trading212/config.py` | CORE | Value object immutabile con tutte le soglie di costruzione dell'universo: market cap, prezzo, liquidità per fascia, storico minimo, borse per paese, campi istituzionali richiesti, suffissi Yahoo per borsa. | — |
| `ingestion/app/services/universe/trading212/filters/__init__.py` | CORE | Aggregatore dei cinque filtri di qualità e della pipeline che li compone. | `ingestion/app/services/universe/trading212/filters/data_coverage.py`, `ingestion/app/services/universe/trading212/filters/historical_data.py`, `ingestion/app/services/universe/trading212/filters/liquidity.py`, `ingestion/app/services/universe/trading212/filters/market_cap.py`, `ingestion/app/services/universe/trading212/filters/pipeline.py`, `ingestion/app/services/universe/trading212/filters/price.py` |
| `ingestion/app/services/universe/trading212/filters/data_coverage.py` | CORE | Filtro che verifica la copertura dei campi istituzionali richiesti (market cap, prezzo, volume, sector/industry, ratio finanziari, ecc.) per uno strumento. | `ingestion/app/services/universe/trading212/config.py` |
| `ingestion/app/services/universe/trading212/filters/historical_data.py` | CORE | Filtro che richiede almeno `min_trading_days` di storico prezzi a 5 anni per ammettere uno strumento nell'universo. | `ingestion/app/services/universe/trading212/config.py`, `ingestion/app/services/market_data/yfinance/_facade.py` |
| `ingestion/app/services/universe/trading212/filters/liquidity.py` | CORE | Filtro che richiede volume medio in dollari e in azioni sopra soglia, differenziata per fascia di market cap. | `ingestion/app/services/universe/trading212/config.py` |
| `ingestion/app/services/universe/trading212/filters/market_cap.py` | CORE | Filtro che respinge strumenti con market cap sotto la soglia minima configurata. | `ingestion/app/services/universe/trading212/config.py` |
| `ingestion/app/services/universe/trading212/filters/pipeline.py` | CORE | Compone una lista ordinata di filtri in un'unica pipeline pass/fail con statistiche per filtro. | `ingestion/app/services/universe/trading212/protocols.py` |
| `ingestion/app/services/universe/trading212/filters/price.py` | CORE | Filtro che esclude penny stock (prezzo troppo basso) e valori di prezzo palesemente errati (troppo alti). | `ingestion/app/services/universe/trading212/config.py` |
| `ingestion/app/services/universe/trading212/protocols.py` | PORTS | Dichiara i sei protocolli che disaccoppiano `UniverseBuilder` dalle implementazioni concrete di filtro, mapping ticker, cache, client T212 e repository. | — |
| `ingestion/app/services/universe/trading212/ticker_mapper.py` | DRIVEN | Adapter che scopre il ticker yfinance corrispondente a un simbolo Trading 212, con cache, tentativi multipli di suffisso e verifica via fetch_info. | `ingestion/app/services/universe/trading212/cache/ticker_cache.py`, `ingestion/app/services/universe/trading212/config.py`, `ingestion/app/services/market_data/yfinance/_facade.py` |
| `ingestion/app/services/universe/universe_build_service.py` | CORE | Use-case di primo livello, invocato da scheduler e CLI, che assembla config, pipeline filtri, ticker mapper, repository e client T212 e lancia il build dell'universo. | `ingestion/app/database.py`, `ingestion/app/repositories/universe/universe_repository.py`, `ingestion/app/schemas/universe/trading212.py`, `ingestion/app/services/_shared/_progress.py`, `ingestion/app/services/universe/trading212/builder.py`, `ingestion/app/services/universe/trading212/cache/ticker_cache.py`, `ingestion/app/services/universe/trading212/client.py`, `ingestion/app/services/universe/trading212/config.py`, `ingestion/app/services/universe/trading212/filters/__init__.py`, `ingestion/app/services/universe/trading212/ticker_mapper.py` |
| `ingestion/app/worker.py` | DRIVING | Entrypoint del demone di ingestion, eseguito come `python -m app.worker`: il processo di lunga durata che possiede la pipeline di dati schedulata. Non espone alcuna API HTTP — le esecuzioni manuali passano da `app.cli`, e lo stato di avanzamento dei job si legge dalla tabella `background_jobs` e dai log. | `ingestion/app/config.py`, `ingestion/app/database.py`, `ingestion/app/metrics.py`, `ingestion/app/services/jobs/scheduler.py`, `ingestion/app/repositories/jobs/background_job_repository.py`, `ingestion/app/services/_shared/_benchmark_bootstrap.py`, `ingestion/app/services/market_data/yfinance.py`, `external:prometheus` |
| `ingestion/baml_client/__init__.py` | DRIVEN | Punto d'ingresso pubblico del pacchetto client BAML generato. | `ingestion/baml_client/types.py`, `ingestion/baml_client/tracing.py`, `ingestion/baml_client/stream_types.py`, `ingestion/baml_client/config.py`, `ingestion/baml_client/sync_client.py`, `ingestion/baml_client/watchers.py`, `ingestion/app/services/macro/macro_calibration.py`, `ingestion/app/services/macro/macro_news_summary.py` |
| `ingestion/baml_client/async_client.py` | DRIVEN | Client asincrono generato — la controparte await-abile di sync_client.py per le stesse due funzioni BAML. | `ingestion/baml_client/stream_types.py`, `ingestion/baml_client/types.py`, `ingestion/baml_client/type_builder.py`, `ingestion/baml_client/parser.py`, `ingestion/baml_client/runtime.py`, `ingestion/baml_client/globals.py` |
| `ingestion/baml_client/config.py` | DRIVEN | Shim di compatibilità per la configurazione del livello di log del client BAML. | `ingestion/baml_client/globals.py` |
| `ingestion/baml_client/globals.py` | DRIVEN | Istanziazione del runtime BAML condiviso da tutto il pacchetto. | `ingestion/baml_client/inlinedbaml.py` |
| `ingestion/baml_client/inlinedbaml.py` | DRIVEN | Copia congelata, al momento della generazione, del testo sorgente dei file .baml del contesto ingestion. | `ingestion/baml_src/` |
| `ingestion/baml_client/parser.py` | DRIVEN | Parsing standalone di una risposta LLM già ottenuta altrove, senza rieseguire la chiamata al modello. | `ingestion/baml_client/stream_types.py`, `ingestion/baml_client/types.py`, `ingestion/baml_client/runtime.py` |
| `ingestion/baml_client/runtime.py` | DRIVEN | Livello di chiamata a basso livello che parla davvero con il motore BAML e, attraverso di esso, con il modello Ollama configurato. | `ingestion/baml_client/types.py`, `ingestion/baml_client/stream_types.py`, `ingestion/baml_client/type_builder.py`, `ingestion/baml_client/globals.py`, `external:ollama` |
| `ingestion/baml_client/stream_types.py` | DRIVEN | Varianti parziali (tutti i campi opzionali) dei due tipi di output BAML, per il consumo incrementale durante lo streaming. | `ingestion/baml_client/types.py` |
| `ingestion/baml_client/sync_client.py` | DRIVEN | Client sincrono generato — l'unica istanza (`b`) che il codice del contesto ingestion importa realmente per invocare le due funzioni BAML. | `ingestion/baml_client/stream_types.py`, `ingestion/baml_client/types.py`, `ingestion/baml_client/type_builder.py`, `ingestion/baml_client/parser.py`, `ingestion/baml_client/runtime.py`, `ingestion/baml_client/globals.py`, `ingestion/app/services/macro/macro_calibration.py`, `ingestion/app/services/macro/macro_news_summary.py` |
| `ingestion/baml_client/tracing.py` | DRIVEN | Superficie pubblica minimale per il tracing e il tagging delle chiamate BAML. | `ingestion/baml_client/globals.py` |
| `ingestion/baml_client/type_builder.py` | DRIVEN | API per estendere a runtime lo schema dei tipi BAML (classi ed enum) senza rigenerare il client. | `ingestion/baml_client/globals.py` |
| `ingestion/baml_client/type_map.py` | DRIVEN | Tabella di lookup nome-qualificato → classe Python per i tipi generati, nelle due varianti complete e parziali. | `ingestion/baml_client/types.py`, `ingestion/baml_client/stream_types.py` |
| `ingestion/baml_client/types.py` | DRIVEN | Contratto tipizzato completo dell'output delle due funzioni BAML del contesto ingestion — il confine di output strutturato che attraversa il modello locale. | `ingestion/app/services/macro/macro_calibration.py`, `ingestion/app/services/macro/macro_news_summary.py` |
| `ingestion/baml_client/watchers.py` | DRIVEN | Definizioni dei tipi per gli hook di osservabilità interna del runtime BAML (eventi di blocco prompt, eventi di variabile, eventi di stream). | `ingestion/baml_client/runtime.py` |
| `ingestion/baml_src/ClassifyMacroRegime.baml` | config | Definisce la funzione BAML che classifica la fase del ciclo economico da un riassunto testuale di indicatori macro e calibra da lì i parametri Black-Litterman δ (avversione al rischio) e τ (scala di incertezza sulle view). | `ingestion/baml_src/clients.baml` |
| `ingestion/baml_src/SummarizeCountryNews.baml` | config | Definisce la funzione BAML che produce il riassunto giornaliero di macro-notizie per paese e ne classifica il sentiment, alimentando la tabella `macro_news_summaries` (Fase 2 della pipeline notizie, secondo il commento in testa al file). | `ingestion/baml_src/clients.baml` |
| `ingestion/baml_src/clients.baml` | config | Configurazione dell'unico client LLM del contesto ingestion e delle policy di retry disponibili alle funzioni BAML del pacchetto. | `external:ollama` |
| `ingestion/baml_src/generators.baml` | config | Configurazione di generazione del codice: dichiara il linguaggio target e la destinazione del client BAML compilato a partire dai file di questo pacchetto. | — |
| `ingestion/tests/__init__.py` | DRIVING | Docstring di pacchetto per l'intera suite di test del demone di ingestion: descrive (in modo in parte aspirazionale, con cartelle unit/integration/e2e) la struttura e i comandi pytest di riferimento. | — |
| `ingestion/tests/_fixtures/__init__.py` | DRIVING | Punto di aggregazione dei seed builder per-dominio (macro, market_data, universe) usati dai test per popolare il database SQLite in-memoria in modo composabile. | `ingestion/tests/_fixtures/macro.py`, `ingestion/tests/_fixtures/market_data.py`, `ingestion/tests/_fixtures/universe.py` |
| `ingestion/tests/_fixtures/_helpers.py` | DRIVING | Utility interna, condivisa dai builder di seed per-dominio, per inserire una riga ORM e forzarne il flush. | — |
| `ingestion/tests/_fixtures/assertions.py` | DRIVING | Helper di asserzione puro, senza DB né I/O, che verifica che un payload sia sicuro per la serializzazione JSON prima di attraversare il confine JSONB o il campo `result` di un job. | — |
| `ingestion/tests/_fixtures/job_service_mock.py` | DRIVING | Mock riutilizzabile per le istanze modulo-livello di `BackgroundJobService`, che lo scheduler istanzia una volta per dominio (`_yfinance_jobs`, `_macro_jobs`, ...) e che aprono sessioni proprie bypassando la sessione di test. | `ingestion/app/services/jobs/background_job.py` |
| `ingestion/tests/_fixtures/macro.py` | DRIVING | Seed builder per il dominio macro: crea un `EconomicIndicator`, una `MacroCalibration` e una `FredObservation` per un singolo paese. | `ingestion/app/models/macro/macro_regime.py`, `ingestion/tests/_fixtures/_helpers.py` |
| `ingestion/tests/_fixtures/market_data.py` | DRIVING | Seed builder per il dominio market_data: crea `Exchange` + `Instrument` + `TickerProfile`, senza `PriceHistory` bulk per restare economico sotto SQLite. | `ingestion/app/models/market_data/yfinance_data.py`, `ingestion/app/models/universe/universe.py`, `ingestion/tests/_fixtures/universe.py`, `ingestion/tests/_fixtures/_helpers.py` |
| `ingestion/tests/_fixtures/universe.py` | DRIVING | Seed builder per il dominio universo: crea `Exchange` + `Instrument` con semantica get-or-create. | `ingestion/app/models/universe/universe.py`, `ingestion/tests/_fixtures/_helpers.py` |
| `ingestion/tests/conftest.py` | DRIVING | Configurazione pytest e fixture condivise per l'intera suite: motore SQLite in-memoria session-scoped, sessione per-test con pattern SAVEPOINT, e fixture di comodo per iniettare `database_manager.get_session` e i seed per-dominio. | `ingestion/app/models/_shared/__init__.py`, `ingestion/app/database.py`, `ingestion/tests/_fixtures/__init__.py`, `ingestion/tests/_fixtures/job_service_mock.py` |
| `ingestion/tests/fixtures/smoke_prices.sql` | DRIVING | Fixture SQL di prezzi sintetici deterministici, pensata per un workflow GitHub Actions di smoke-test che carica dati dopo `alembic upgrade head` e prima di invocare `POST /api/v1/optimize` e `POST /api/v1/backtest`. | — |
| `ingestion/tests/integration/__init__.py` | DRIVING | Docstring di pacchetto per una cartella `integration/` che oggi non contiene ancora test — dichiara l'intento (DB reale, wiring service+repository, integrazioni esterne con mock) senza implementazione. | — |
| `ingestion/tests/repositories/__init__.py` | DRIVING | File vuoto che rende `tests.repositories` un pacchetto Python importabile. | — |
| `ingestion/tests/repositories/test_macro_regime_repository.py` | DRIVING | Test dell'adapter driven `MacroRegimeRepository.upsert_regime_classification`, che persiste l'output del classificatore rule-based di regime macro sulla tabella `macro_calibrations`. | `ingestion/app/repositories/macro/macro_regime_repository.py`, `ingestion/app/models/macro/macro_regime.py` |
| `ingestion/tests/unit/__init__.py` | DRIVING | Docstring di pacchetto per `tests.unit`: dichiara l'intento (test isolati, veloci, senza DB o API esterne, mock per le dipendenze) dell'intera sottocartella. | — |
| `ingestion/tests/unit/_shared/__init__.py` | DRIVING | File vuoto: marker di pacchetto per una sottocartella `unit/_shared/` oggi senza test propri (i test degli helper condivisi vivono in `unit/services/_shared/` e in singoli file a `unit/` root). | — |
| `ingestion/tests/unit/schemas/__init__.py` | DRIVING | File vuoto: marker di pacchetto per una sottocartella `unit/schemas/` oggi senza test propri (i test degli schemi Pydantic vivono in file dedicati a `unit/` root, es. `test_yfinance_schemas.py`). | — |
| `ingestion/tests/unit/services/__init__.py` | DRIVING | File vuoto: marker di pacchetto per `tests.unit.services`, la sottocartella che raggruppa i test per dominio del layer core (`_shared`, `jobs`, `macro`, `market_data`, `scenarios`, `universe`). | — |
| `ingestion/tests/unit/services/_shared/__init__.py` | DRIVING | File vuoto: marker di pacchetto per i test degli helper condivisi del layer core (`_benchmark_bootstrap`, `_json_safe`, `notifications`, `_price_fetcher`, `_progress`, `_sector_resolver`, `trading_calendar`). | — |
| `ingestion/tests/unit/services/_shared/test_benchmark_bootstrap.py` | DRIVING | Test del bootstrap dei benchmark di riferimento (`_shared/_benchmark_bootstrap`), che su un database freddo verifica e reintegra la copertura degli indici di riferimento prima della prima esecuzione schedulata. | `ingestion/app/services/_shared/_benchmark_bootstrap.py` |
| `ingestion/tests/unit/services/_shared/test_json_safe.py` | DRIVING | Test dell'helper puro `_shared/_json_safe.safe_float`, che normalizza valori eterogenei (None, NaN/inf, stringhe, scalari numpy, array) in un `float` JSON-safe o `None`. | `ingestion/app/services/_shared/_json_safe.py` |
| `ingestion/tests/unit/services/_shared/test_notifications_error_branch.py` | DRIVING | Test di gap-fill sul ramo d'errore del webhook di notifica: garantisce che un fallimento di rete sia loggato (WARNING) e non solo silenziosamente inghiottito. | `ingestion/app/services/_shared/notifications.py` |
| `ingestion/tests/unit/services/_shared/test_price_fetcher.py` | DRIVING | Test di `_shared/_price_fetcher.fetch_close_prices`, che documenta e blocca il contratto 'drop silenzioso' per i ticker assenti dal database (a differenza del guard-on-empty, che vive nel chiamante). | `ingestion/app/services/_shared/_price_fetcher.py`, `ingestion/app/repositories/market_data/yfinance_repository.py` |
| `ingestion/tests/unit/services/_shared/test_progress.py` | DRIVING | Test delle factory di callback di progresso in `_shared/_progress` (`_noop`, `make_progress`, `make_cancellable_progress`), usate da ogni step della pipeline per riportare avanzamento e permettere la cancellazione cooperativa. | `ingestion/app/services/_shared/_progress.py` |
| `ingestion/tests/unit/services/_shared/test_sector_resolver.py` | DRIVING | Test di `_shared/_sector_resolver.resolve_sector_map`, il resolver a due livelli (snapshot poi `TickerProfile`) condiviso da Attribution e Dashboard per mappare ticker a settore. | `ingestion/app/services/_shared/_sector_resolver.py`, `ingestion/app/models/market_data/yfinance_data.py` |
| `ingestion/tests/unit/services/_shared/test_trading_calendar.py` | DRIVING | Test di `_shared/trading_calendar` (`parse_period_years`, `get_expected_trading_sessions`, `has_sufficient_history`), che delega al calendario reale `exchange_calendars` per stimare le sessioni di trading attese di un exchange su un periodo. | `ingestion/app/services/_shared/trading_calendar.py` |
| `ingestion/tests/unit/services/jobs/__init__.py` | DRIVING | File vuoto: marker di pacchetto per i test del dominio jobs (`background_job`, `scheduler`). | — |
| `ingestion/tests/unit/services/jobs/test_background_job_lifecycle.py` | DRIVING | Test del ciclo di vita di `BackgroundJobService` e del reaper periodico, colmando i gap lasciati da `test_background_job_service_heartbeat.py`: forwarder di progresso semplice, conflitto su `create_job`, cancel a livello thread via evento, `run_orphan_reaper`. | `ingestion/app/services/jobs/background_job.py`, `ingestion/app/services/jobs/scheduler.py`, `ingestion/app/repositories/jobs/background_job_repository.py`, `ingestion/app/services/_shared/_progress.py`, `ingestion/app/config.py` |
| `ingestion/tests/unit/services/jobs/test_scheduler_registration.py` | DRIVING | Test che `create_scheduler()` registri esattamente i 7 job attesi con i trigger giusti, senza avviare lo scheduler né invocare alcuna funzione registrata (nessun fetch reale). | `ingestion/app/services/jobs/scheduler.py` |
| `ingestion/tests/unit/services/macro/__init__.py` | DRIVING | File vuoto: marker di pacchetto per i test del dominio macro (scraper, regime service, news summary). | — |
| `ingestion/tests/unit/services/macro/test_ilsole_scraper.py` | DRIVING | Test unitario di `IlSoleScraper` (indicatori reali e previsioni per paese): parsing riga, paese vuoto/sconosciuto, `_safe_float`, guardia di regressione su `Accept-Encoding`, e trip del circuit breaker su fallimento di rete ripetuto — nessuna rete reale, nessuna sleep reale. | `ingestion/app/services/macro/scrapers/ilsole_scraper.py`, `ingestion/app/services/macro/scrapers/tradingeconomics_scraper.py`, `ingestion/app/services/infrastructure/circuit_breaker.py`, `ingestion/app/services/infrastructure/retry.py` |
| `ingestion/tests/unit/services/macro/test_macro_news_summary.py` | DRIVING | Test della pipeline `macro_news_summary` (validazione output LLM, riassunto per paese a due paesi, gate di soglia articoli, `run_news_summarize`), che sintetizza le notizie macro per paese tramite una chiamata BAML. | `ingestion/app/services/macro/macro_news_summary.py`, `ingestion/app/repositories/macro/macro_regime_repository.py`, `ingestion/app/database.py`, `ingestion/baml_client/` |
| `ingestion/tests/unit/services/macro/test_macro_regime_service.py` | DRIVING | Test di `MacroRegimeService.fetch_country` e `run_bulk_macro_fetch`, che orchestrano lo scraping di indicatori reali/previsioni/rendimenti obbligazionari per paese e li persistono via repository. | `ingestion/app/services/macro/macro_regime_service.py`, `ingestion/app/repositories/macro/macro_regime_repository.py`, `ingestion/app/database.py` |
| `ingestion/tests/unit/services/market_data/__init__.py` | DRIVING | File vuoto: marker di pacchetto per i test del dominio market_data (yfinance: news, ticker sub-client). | — |
| `ingestion/tests/unit/services/market_data/test_yfinance_macro_news_service.py` | DRIVING | Test del sub-client `yfinance/news/macro_news.MacroNewsFetcher`, che raccoglie e classifica per tema le notizie macro da query di ricerca e ticker fissi. | `ingestion/app/services/market_data/yfinance/news/macro_news.py`, `ingestion/tests/_fixtures/assertions.py` |
| `ingestion/tests/unit/services/market_data/test_yfinance_news_service.py` | DRIVING | Test dei sub-client `yfinance/news/client.NewsClient` e `yfinance/news/aggregator.CountryNewsFetcher`, che recuperano e aggregano notizie per singolo ticker o per paese. | `ingestion/app/services/market_data/yfinance/news/client.py`, `ingestion/app/services/market_data/yfinance/news/aggregator.py`, `ingestion/tests/_fixtures/assertions.py` |
| `ingestion/tests/unit/services/market_data/test_yfinance_ticker_service.py` | DRIVING | Test dei sub-client `yfinance/ticker/{corporate_actions,holders,financials}`, che incapsulano dividendi/split/azioni, holder istituzionali/insider, e bilanci trimestrali/annuali per ticker. | `ingestion/app/services/market_data/yfinance/ticker/corporate_actions.py`, `ingestion/app/services/market_data/yfinance/ticker/holders.py`, `ingestion/app/services/market_data/yfinance/ticker/financials.py` |
| `ingestion/tests/unit/services/scenarios/__init__.py` | DRIVING | File vuoto: marker di pacchetto per una sottocartella `unit/services/scenarios/` riservata a un dominio `services/scenarios` che oggi non ha test propri. | — |
| `ingestion/tests/unit/services/universe/__init__.py` | DRIVING | File vuoto: marker di pacchetto per i test del dominio universo Trading 212 (builder, cache, filtri, mapper, universe_build_service). | — |
| `ingestion/tests/unit/services/universe/test_t212_builder.py` | DRIVING | Test di `UniverseBuilder.build()`, l'orchestratore che unisce scoperta exchange/strumento Trading 212, mapping a ticker yfinance e persistenza via repository, con filtro opzionale sull'universo investibile. | `ingestion/app/services/universe/trading212/builder.py`, `ingestion/app/services/universe/trading212/cache/ticker_cache.py`, `ingestion/app/services/universe/trading212/config.py`, `ingestion/app/services/universe/trading212/ticker_mapper.py` |
| `ingestion/tests/unit/services/universe/test_t212_cache.py` | DRIVING | Test di `TickerMappingCache`, la cache effimera per-build senza TTL né limite di capacità che memorizza le mappature simbolo→ticker yfinance scoperte durante una build. | `ingestion/app/services/universe/trading212/cache/ticker_cache.py` |
| `ingestion/tests/unit/services/universe/test_t212_filters.py` | DRIVING | Test dei filtri dell'universo Trading 212 (`MarketCapFilter`, `PriceFilter`, `LiquidityFilter`, `DataCoverageFilter`, `HistoricalDataFilter`) e di `FilterPipelineImpl`, che applica in sequenza i filtri con short-circuit e statistiche. | `ingestion/app/services/universe/trading212/config.py`, `ingestion/app/services/universe/trading212/filters/data_coverage.py`, `ingestion/app/services/universe/trading212/filters/historical_data.py`, `ingestion/app/services/universe/trading212/filters/liquidity.py`, `ingestion/app/services/universe/trading212/filters/market_cap.py`, `ingestion/app/services/universe/trading212/filters/pipeline.py`, `ingestion/app/services/universe/trading212/filters/price.py` |
| `ingestion/tests/unit/services/universe/test_t212_mapper.py` | DRIVING | Test di `YFinanceTickerMapper`, che mappa un simbolo Trading 212 al ticker yfinance corretto provando suffissi per exchange e verificando ciascun tentativo, con cache di risultato. | `ingestion/app/services/universe/trading212/cache/ticker_cache.py`, `ingestion/app/services/universe/trading212/config.py`, `ingestion/app/services/universe/trading212/ticker_mapper.py` |
| `ingestion/tests/unit/services/universe/test_universe_build_service.py` | DRIVING | Test del wiring di `services/universe/universe_build_service` (`build_trading212_client`, `run_universe_build`), il punto che lo scheduler chiama per lanciare una build dell'universo. | `ingestion/app/services/universe/universe_build_service.py`, `ingestion/app/services/universe/trading212/builder.py`, `ingestion/app/schemas/universe/trading212.py`, `ingestion/app/database.py` |
| `ingestion/tests/unit/test_background_job_liveness_columns.py` | DRIVING | Verifica di schema per le colonne di liveness worker (`worker_pid`, `worker_host`, `last_heartbeat_at`) e per l'indice composito su `background_jobs`. | `ingestion/app/models/jobs/background_job.py` |
| `ingestion/tests/unit/test_background_job_repository.py` | DRIVING | Test dell'adapter driven `BackgroundJobRepository` con SQL dialect-agnostico: atomicità di `claim_or_create`, aritmetica di `cleanup_expired`, `get_latest_by_type` e il predicato di liveness a quattro condizioni di `reconcile_orphans`. | `ingestion/app/repositories/jobs/background_job_repository.py` |
| `ingestion/tests/unit/test_background_job_repository_branches.py` | DRIVING | Test di copertura dei rami residui di `BackgroundJobRepository` non coperti da `test_background_job_repository.py` (fallback di `_ensure_tables_exist`, rami minori a linee specifiche). | `ingestion/app/repositories/jobs/background_job_repository.py` |
| `ingestion/tests/unit/test_background_job_service_heartbeat.py` | DRIVING | Test del thread di heartbeat e dell'evento di cancellazione di `BackgroundJobService`, che mantiene viva la riga `background_jobs` di un job in esecuzione battendo `last_heartbeat_at` a cadenza regolare. | `ingestion/app/services/jobs/background_job.py`, `ingestion/app/services/_shared/_progress.py` |
| `ingestion/tests/unit/test_background_job_service_paths.py` | DRIVING | Test di copertura dei rami di `BackgroundJobService` — il ciclo di vita del job stesso — che tengono un'esecuzione bloccata lontana dal contaminare la pipeline: id malformato ignorato, heartbeat che sopravvive a un errore DB transitorio ma si ferma quando il reaper ha reclamato la riga, evento di cancel sempre settato anche se il target solleva, metriche/webhook di fallimento sulle transizioni giuste. | `ingestion/app/services/jobs/background_job.py` |
| `ingestion/tests/unit/test_cli.py` | DRIVING | Test dell'adapter driving `app.cli` (esecuzione manuale via `python -m app.cli <comando>`), che verifica il contratto da cui dipendono gli script di shell: ogni comando invoca lo stesso step dello scheduler automatico, con le opzioni giuste, e i comandi a singolo step escono non-zero se lo step non ha completato. | `ingestion/app/cli.py`, `ingestion/app/services/jobs/scheduler.py`, `ingestion/app/database.py` |
| `ingestion/tests/unit/test_database_admin_repository_branches.py` | DRIVING | Test di copertura dei rami di `DatabaseAdminRepository`, separando i percorsi eseguibili su SQLite (health check, guardia allowlist sulle tabelle) da quelli Postgres-only (query `information_schema`, `pg_total_relation_size`, `TRUNCATE ... CASCADE`) documentati ma non eseguiti. | `ingestion/app/repositories/_shared/database_admin_repository.py` |
| `ingestion/tests/unit/test_database_lazy_init.py` | DRIVING | Test della guardia di inizializzazione pigra di `DatabaseManager.get_session` (issue #562, #563): il manager chiama `initialize()` solo se non è già stato inizializzato. | `ingestion/app/database.py` |
| `ingestion/tests/unit/test_database_manager.py` | DRIVING | Test di `app/database.py` — l'unico percorso del demone verso Postgres — che verifica creazione dell'engine, impostazioni di pool da config, idempotenza dell'inizializzazione, rollback e recupero da disconnessione. | `ingestion/app/database.py`, `ingestion/app/config.py` |
| `ingestion/tests/unit/test_date_parsing_branches.py` | DRIVING | Test di copertura dei rami per `app/utils/date_parsing.parse_reference_date`, che normalizza date scritte in formati eterogenei (report FRED/ilSole/TradingEconomics) in oggetti `date`. | `ingestion/app/utils/date_parsing.py` |
| `ingestion/tests/unit/test_fetch_earnings_deprecation.py` | DRIVING | Test di regressione (issue #591) che garantisce che `YFinanceDataService.fetch_and_store` non invochi mai il percorso deprecato `Ticker.earnings` di yfinance 1.3.0, e che il protocollo non lo esponga più. | `ingestion/app/services/market_data/yfinance/protocols/interfaces.py`, `ingestion/app/services/market_data/yfinance/ticker/financials.py`, `ingestion/app/services/market_data/yfinance_data_service.py` |
| `ingestion/tests/unit/test_fetch_fred_series_progress.py` | DRIVING | Test TDD sul contratto di progresso per-serie di `MacroRegimeService.fetch_fred_series`: la firma deve accettare `on_progress` e il callback deve essere invocato una volta per ogni serie FRED richiesta. | `ingestion/app/services/macro/macro_regime_service.py` |
| `ingestion/tests/unit/test_fred_progress.py` | DRIVING | Blocca il contratto di progresso per-serie di `fetch_fred_series` con un test minimale e mirato (issue #569), complementare a `test_fetch_fred_series_progress.py`. | `ingestion/app/services/macro/macro_regime_service.py` |
| `ingestion/tests/unit/test_fred_scraper_catalog.py` | DRIVING | Test di appartenenza al catalogo per i sotto-dizionari `FRED_SERIES`, a guardia della deriva tra la lista `_REQUIRED_FRED_SERIES` di preflight e il registro dello scraper. | `ingestion/app/services/macro/scrapers/fred_scraper.py` |
| `ingestion/tests/unit/test_infrastructure_shims.py` | DRIVING | Test di parità per il layer di re-export `market_data/yfinance/infrastructure/{cache,circuit_breaker,rate_limiter,retry}` (issue #823), che mantiene stabili i vecchi percorsi di import yfinance mentre le primitive di resilienza sono ora condivise. | `ingestion/app/services/infrastructure/cache.py`, `ingestion/app/services/infrastructure/circuit_breaker.py`, `ingestion/app/services/infrastructure/rate_limiter.py`, `ingestion/app/services/infrastructure/retry.py`, `ingestion/app/services/market_data/yfinance/infrastructure/cache.py`, `ingestion/app/services/market_data/yfinance/infrastructure/circuit_breaker.py`, `ingestion/app/services/market_data/yfinance/infrastructure/rate_limiter.py`, `ingestion/app/services/market_data/yfinance/infrastructure/retry.py` |
| `ingestion/tests/unit/test_ingestion_error_logging.py` | DRIVING | Test (issue #850) che ogni drop per-elemento precedentemente silenzioso ora sia loggato prima di continuare: un elemento malformato è saltato/degradato e loggato, mai inghiottito senza traccia. | `ingestion/app/services/macro/scrapers/fred_scraper.py`, `ingestion/app/services/macro/scrapers/ilsole_scraper.py`, `ingestion/app/services/market_data/yfinance/news/aggregator.py` |
| `ingestion/tests/unit/test_lru_cache.py` | DRIVING | Test della primitiva condivisa `LRUCache` (issue #823): scadenza TTL, eviction LRU a capacità piena, rispetto del TTL da `contains`/`keys()`, interleaving thread-safe. | `ingestion/app/services/infrastructure/cache.py` |
| `ingestion/tests/unit/test_macro_calibration.py` | DRIVING | Test unitario del servizio e dell'endpoint di calibrazione del regime macro, che affida a BAML il giudizio (fase del ciclo, delta, tau, confidenza, razionale) a partire da indicatori deterministici. | `ingestion/app/services/macro/macro_calibration.py`, `ingestion/baml_client/` |
| `ingestion/tests/unit/test_macro_calibration_service_branches.py` | DRIVING | Test di copertura dei rami per `app/services/macro/macro_calibration.py`, mappando gli intervalli di codice non coperti da `test_macro_calibration.py` a condizioni concrete: costruzione del summary macro, calcolo dello spread 10Y-2Y, cache-hit, override del summary, esito e fallimento dell'upsert, `run_bulk_calibrate` con progresso e commit per paese. | `ingestion/app/services/macro/macro_calibration.py`, `ingestion/baml_client/` |
| `ingestion/tests/unit/test_macro_news_summary_repository.py` | DRIVING | Test dei metodi repository per `MacroNewsSummary`: inserimento e aggiornamento del riassunto giornaliero delle notizie macro per paese. | `ingestion/app/repositories/macro/macro_regime_repository.py` |
| `ingestion/tests/unit/test_macro_news_summary_service.py` | DRIVING | Test unitario più esteso del servizio `macro_news_summary`, che copre la mappatura query/ticker→paese, il filtraggio degli articoli nuovi per paese, la formattazione e il gate di completezza della pipeline mattutina, oltre alla validazione dell'output LLM. | `ingestion/app/services/macro/macro_news_summary.py`, `ingestion/app/models/macro/macro_regime.py`, `ingestion/baml_client/` |
| `ingestion/tests/unit/test_macro_regime_repository_branches.py` | DRIVING | Test di copertura dei rami per `MacroRegimeRepository`, che dichiara esplicitamente quali metodi upsert (basati su `sqlalchemy.dialects.postgresql.insert` con `ON CONFLICT`) non sono eseguibili su SQLite e restano documentati ma non testati oltre le loro guardie SQLite-safe. | `ingestion/app/repositories/macro/macro_regime_repository.py`, `ingestion/app/models/macro/macro_regime.py` |
| `ingestion/tests/unit/test_macro_regime_service_branches.py` | DRIVING | Test di copertura dei rami residui di `MacroRegimeService` non coperti dai due file di test dedicati esistenti: property `fred_scraper` già impostata, `fetch_and_store` con paesi di default/eccezione per paese, `fetch_fred_series` incrementale, `fetch_macro_news`, `run_bulk_macro_fetch`/`run_bulk_fred_fetch`/`run_macro_news_fetch` felici e con errore. | `ingestion/app/services/macro/macro_regime_service.py` |
| `ingestion/tests/unit/test_macro_regime_service_fred_key.py` | DRIVING | Test del messaggio diagnostico consapevole dell'ambiente Docker quando `FRED_API_KEY` non è configurata (issue #564). | `ingestion/app/services/macro/macro_regime_service.py`, `ingestion/app/config.py` |
| `ingestion/tests/unit/test_market_data_service_branches.py` | DRIVING | Test di copertura dei rami per `yfinance_data_service.py` e `reference_index_seeder.py`, mirati a normalizzazione timezone, fallback su fallimento della query di staleness, guardie sullo storico insufficiente, e casi limite del parsing link delle notizie. | `ingestion/app/services/market_data/yfinance_data_service.py`, `ingestion/app/services/market_data/reference_index_seeder.py` |
| `ingestion/tests/unit/test_new_tables_migration.py` | DRIVING | Test della migrazione Alembic `y5z6a7b8c9d0`, che aggiunge sei nuove tabelle (`factor_scores`, `factor_validation_reports`, `optimization_runs`, `backtest_runs`, `risk_limits`, `rebalancing_policies`). | `ingestion/alembic/versions/` |
| `ingestion/tests/unit/test_notifications.py` | DRIVING | Test unitario del servizio di notifica webhook (`_shared/notifications.notify_failure`), che avvisa un canale esterno (es. Discord) quando un job schedulato fallisce. | `ingestion/app/services/_shared/notifications.py` |
| `ingestion/tests/unit/test_price_fetcher.py` | DRIVING | Test TDD (issue #382, fase 'red' scritta prima dell'implementazione) di `_price_fetcher.fetch_close_prices`, isolato interamente via mock del repository, senza chiamate DB reali. | `ingestion/app/services/_shared/_price_fetcher.py`, `ingestion/app/repositories/market_data/yfinance_repository.py` |
| `ingestion/tests/unit/test_reference_index_seeder.py` | DRIVING | Test del contratto d'ordine del callback di progresso in `seed_reference_indices` (issue #572): il sentinella pre-loop `on_progress(total=N)` deve essere emesso prima di qualunque `on_progress(current=...)` per singolo ticker. | `ingestion/app/services/market_data/reference_index_seeder.py` |
| `ingestion/tests/unit/test_repository_base_branches.py` | DRIVING | Test di copertura dei rami per `repositories/_shared/base.py`: le operazioni CRUD di `BaseRepository` e il ramo di early-return a righe vuote di `RepositoryBase._upsert`. | `ingestion/app/repositories/_shared/base.py`, `ingestion/app/models/jobs/background_job.py` |
| `ingestion/tests/unit/test_scheduler_liveness_settings.py` | DRIVING | Test (issue #590) dell'esposizione via settings dei parametri del reaper di liveness: cadenza di heartbeat e timeout di orfanaggio, con override da variabili d'ambiente. | `ingestion/app/config.py`, `ingestion/app/worker.py` |
| `ingestion/tests/unit/test_scheduler_pipeline.py` | DRIVING | Test puramente unitario delle funzioni di orchestrazione pipeline in `app.services.jobs.scheduler` (i wrapper `run_*_step` che CLI e cron invocano), senza client HTTP, DB o thread reali. | `ingestion/app/services/jobs/scheduler.py`, `ingestion/app/services/jobs/background_job.py`, `ingestion/app/config.py` |
| `ingestion/tests/unit/test_sector_resolver.py` | DRIVING | Test end-to-end (con `db_session` reale) del resolver di settore a due livelli condiviso da Attribution e Dashboard (issue #427), incluso il fix per cui il join su `TickerProfile` deve avvenire su `Instrument.yfinance_ticker`, non su `Instrument.ticker`. | `ingestion/app/models/market_data/yfinance_data.py`, `ingestion/app/models/universe/universe.py`, `ingestion/tests/_fixtures/universe.py` |
| `ingestion/tests/unit/test_seed_reference_indices_expanded_migration.py` | DRIVING | Test della migrazione Alembic `a7b8c9d0e1f2` (issue #430), che semina 9 benchmark di riferimento aggiuntivi (QQQ, IWM, EFA, EEM, AGG, VGK, ...) oltre a SPY già seminato dalla migrazione predecessore. | `ingestion/alembic/versions/` |
| `ingestion/tests/unit/test_seed_reference_indices_migration.py` | DRIVING | Test della migrazione Alembic `x4y5z6a7b8c9`, che semina il primo benchmark di riferimento SPY su NYSE. | `ingestion/alembic/versions/` |
| `ingestion/tests/unit/test_sentiment_repository_branches.py` | DRIVING | Test di copertura dei rami per `SentimentRepository`, incluse le tre strategie a cascata di `get_macro_news_fallback` (match diretto per source_ticker, match per settore via ETF, ricerca testuale ilike su titolo/contenuto). | `ingestion/app/repositories/macro/sentiment_repository.py`, `ingestion/app/models/macro/macro_regime.py`, `ingestion/app/models/market_data/yfinance_data.py`, `ingestion/tests/_fixtures/universe.py` |
| `ingestion/tests/unit/test_shared_infrastructure.py` | DRIVING | Test delle primitive di resilienza condivise: `CircuitBreaker`, `RateLimiter`, `retry_with_backoff` — il meccanismo comune usato da tutti gli scraper per rispettare i rate-limit delle fonti esterne. | `ingestion/app/services/infrastructure/circuit_breaker.py`, `ingestion/app/services/infrastructure/rate_limiter.py`, `ingestion/app/services/infrastructure/retry.py` |
| `ingestion/tests/unit/test_trading212_client.py` | DRIVING | Test unitario di `Trading212Client`, il client HTTP thin usato per costruire l'universo investibile (demo/live), con ogni chiamata HTTP mockata via `requests`. | `ingestion/app/services/universe/trading212/client.py` |
| `ingestion/tests/unit/test_trading212_filters_branches.py` | DRIVING | Test di copertura dei rami per i sotto-moduli filtro Trading 212 e per `config.py`: helper `to_dict`, factory method, property `.name`, formattazione di market cap sotto 1 miliardo, categorie di data-coverage mancanti oltre soglia, client yfinance lazy in `HistoricalDataFilter`. | `ingestion/app/services/universe/trading212/config.py`, `ingestion/app/services/universe/trading212/filters/data_coverage.py`, `ingestion/app/services/universe/trading212/filters/historical_data.py`, `ingestion/app/services/universe/trading212/filters/liquidity.py`, `ingestion/app/services/universe/trading212/filters/market_cap.py`, `ingestion/app/services/universe/trading212/filters/pipeline.py` |
| `ingestion/tests/unit/test_trading212_mapper_builder_branches.py` | DRIVING | Test di copertura dei rami residui per `ticker_mapper.py` (~19 rami) e `builder.py` (~13 rami), complementare a `test_t212_mapper.py` e `test_t212_builder.py`. | `ingestion/app/services/universe/trading212/builder.py`, `ingestion/app/services/universe/trading212/cache/ticker_cache.py`, `ingestion/app/services/universe/trading212/config.py`, `ingestion/app/services/universe/trading212/ticker_mapper.py` |
| `ingestion/tests/unit/test_tradingeconomics_scraper.py` | DRIVING | Test di `TradingEconomicsIndicatorsScraper` (issue #317): propagazione di `ParseStructureError`, trip del circuit breaker, gestione del fallimento del fetch, casi limite di `_extract_number`, uscita anticipata per paese non supportato — nessun DB, nessuna chiamata HTTP reale. | `ingestion/app/services/macro/scrapers/tradingeconomics_scraper.py`, `ingestion/app/services/macro/scrapers/exceptions.py` |
| `ingestion/tests/unit/test_universe_repository_branches.py` | DRIVING | Test di copertura dei rami per `UniverseRepository` (target ≥80%): `save_exchange` (update/create), `mark_delisted`, `get_active_tickers`, `clear_all`, conteggi, `get_instruments`/`get_exchanges` con filtri. | `ingestion/app/repositories/universe/universe_repository.py`, `ingestion/app/models/universe/universe.py`, `ingestion/tests/_fixtures/universe.py` |
| `ingestion/tests/unit/test_upsert_updated_at.py` | DRIVING | Test di regressione (issue #311) che, su `ON CONFLICT DO UPDATE`, `updated_at` sia stampato con l'ora corrente del server e non con il valore stantio di `excluded.updated_at`. | `ingestion/app/repositories/macro/macro_regime_repository.py`, `ingestion/app/models/macro/macro_regime.py` |
| `ingestion/tests/unit/test_worker.py` | DRIVING | Test dell'entrypoint del demone `python -m app.worker`, che blocca su un evento di shutdown già settato in ogni test così lo startup corre a completamento e lo shutdown procede subito, senza aprire socket, database o scheduler reali. | `ingestion/app/worker.py`, `ingestion/app/database.py`, `ingestion/app/metrics.py`, `ingestion/app/services/jobs/scheduler.py`, `ingestion/app/repositories/jobs/background_job_repository.py` |
| `ingestion/tests/unit/test_yfinance_analysis_eps.py` | DRIVING | Test di `AnalysisClient.fetch_eps_trend`/`fetch_eps_revisions`, il sub-client che espone i pannelli DataFrame `Ticker.eps_trend`/`Ticker.eps_revisions` di yfinance 1.3.0. | `ingestion/app/services/market_data/yfinance/protocols/interfaces.py`, `ingestion/app/services/market_data/yfinance/ticker/analysis.py` |
| `ingestion/tests/unit/test_yfinance_cycle3_price_quality.py` | DRIVING | Test di regressione (post-#598) sul contratto degli otto kwarg di qualità-prezzo inoltrati da `fetch_history` a `ticker.history`, e sulla continuità dei prezzi intorno a uno split azionario, hermetico e senza chiamate Yahoo reali. | `ingestion/app/services/market_data/yfinance/_facade.py`, `ingestion/app/services/market_data/yfinance/infrastructure/cache.py`, `ingestion/app/services/market_data/yfinance/infrastructure/circuit_breaker.py`, `ingestion/app/services/market_data/yfinance/infrastructure/rate_limiter.py` |
| `ingestion/tests/unit/test_yfinance_cycle4_cached_property.py` | DRIVING | Test di tortura (issue #601) sulla migrazione delle 13 property sub-client di `YFinanceClient` a `functools.cached_property`, incluso il comportamento sotto race a 50 thread e la pulizia di `reset_instance`. | `ingestion/app/services/market_data/yfinance/_facade.py` |
| `ingestion/tests/unit/test_yfinance_cycle5_search_expansion.py` | DRIVING | Test (issue #604) di `SearchClient.search`, che espone `news_count` e include condizionalmente le chiavi `lists`/`research`/`nav` nel risultato via flag opt-in. | `ingestion/app/services/market_data/yfinance/market/search.py`, `ingestion/app/services/market_data/yfinance/protocols/interfaces.py` |
| `ingestion/tests/unit/test_yfinance_data_service_branches.py` | DRIVING | Test di copertura dei rami residui di `YFinanceDataService`: normalizzazione timezone in `_is_fresh`, `_timed` con timeout esplicito, fallback su fallimento della query di staleness, percorso incrementale a range di date, guardie sullo storico insufficiente, rami delle notizie, `_aggregate_outcome` su eccezione. | `ingestion/app/services/market_data/yfinance_data_service.py` |
| `ingestion/tests/unit/test_yfinance_data_service_cycle2.py` | DRIVING | Test di wiring a livello service per valuation/eps_trend/eps_revisions: che `YFinanceDataService.fetch_and_store` instradi i tre nuovi fetcher attraverso `repo.upsert_financial_statements` con le triple `statement_type`/`period_type`/`currency_code` documentate. | `ingestion/app/services/market_data/yfinance_data_service.py`, `ingestion/app/repositories/market_data/yfinance_repository.py` |
| `ingestion/tests/unit/test_yfinance_facade_paths.py` | DRIVING | Test di copertura dei rami per `YFinanceClient` (facade), incentrato sulle degradazioni sotto ogni step di ingestion: rate-limit che deve far scattare il circuit breaker, ticker con troppo pochi campi che deve tornare `None` invece di un dict parziale, bulk download fallito che deve ripiegare su fetch per-ticker senza perdere l'intero batch. | `ingestion/app/services/market_data/yfinance/_facade.py` |
| `ingestion/tests/unit/test_yfinance_metadata_valuation.py` | DRIVING | Test di `MetadataClient.fetch_valuation_measures`, il sub-client che espone il pannello a 9 metriche `Ticker.valuation` di yfinance 1.3.0. | `ingestion/app/services/market_data/yfinance/protocols/interfaces.py`, `ingestion/app/services/market_data/yfinance/ticker/metadata.py` |
| `ingestion/tests/unit/test_yfinance_parallel_fetch.py` | DRIVING | Test (issue #575) del percorso di fetch bulk parallelo yfinance, che patcha `database_manager.get_session` a livello modulo così ogni worker riceve una sessione mock distinta e nessuno tocca Postgres. | `ingestion/app/services/market_data/yfinance_data_service.py`, `ingestion/app/database.py` |
| `ingestion/tests/unit/test_yfinance_protocol_cycle3.py` | DRIVING | Test di regressione sulla firma del protocollo `YFinanceClientProtocol` (issue #597): `fetch_history` deve esporre gli otto kwarg di qualità-prezzo e `bulk_download` deve avere `auto_adjust=True` di default con `max_retries` allineato alla firma della facade. | `ingestion/app/services/market_data/yfinance/protocols/interfaces.py` |
| `ingestion/tests/unit/test_yfinance_repository_branches.py` | DRIVING | Test di copertura dei rami per `YFinanceRepository` e i suoi helper modulo-livello, separando i metodi `upsert_*` (Postgres-only, `ON CONFLICT`, non eseguibili su SQLite) dalle query SELECT pure, testabili con la sessione reale. | `ingestion/app/repositories/market_data/yfinance_repository.py`, `ingestion/app/models/market_data/yfinance_data.py` |
| `ingestion/tests/unit/test_yfinance_schemas.py` | DRIVING | Test di validazione (issue #573) per lo schema Pydantic della request di fetch yfinance, `YFinanceFetchRequest`, in particolare il campo `workers`. | `ingestion/app/schemas/market_data/yfinance_data.py` |
| `ingestion/tests/unit/test_yfinance_timeout_and_throughput.py` | DRIVING | Test del watchdog di timeout yfinance e dei miglioramenti di throughput incrementale: `call_with_timeout` su chiamata che si blocca, skip a zero chiamate di rete quando ogni categoria è già fresca, propagazione dei worker da settings allo scheduler. | `ingestion/app/services/market_data/yfinance_data_service.py`, `ingestion/app/services/jobs/scheduler.py` |

<!-- FINE GENERATO: ingestion -->

---

## 6. `fund/` — il fondo a quattro agenti

I quattro agenti sono nodi guidati da un **orchestratore scritto a mano** sopra lo stato
condiviso `FundState` (§9), con LangGraph adottato solo come substrato durevole per checkpoint
e interrupt. Condividono lo stato tipizzato e si passano il lavoro attraverso di esso — mai
direttamente. Non chiamano endpoint HTTP grezzi: chiamano **use-case** interni come strumenti,
il che mantiene il service layer come unico confine di orchestrazione.

```
  agente macro         agente allocazione     agente rischio        agente esecuzione
  ────────────         ──────────────────     ──────────────        ─────────────────
  prob. di regime      prior di equilibrio    misura di coda        portafoglio "aim"
  turbolenza /         + view → posterior     coerente              (una mossa parziale,
  absorption ratio     → pesi, per MV /       + rischio di          non tutto il divario)
  E[r] da segnali      BL / risk budgeting      percorso            + traiettoria di
  macro                                       + attribuzione Euler    impatto e timing
                                              + ricerca dello
                                                scenario peggiore
        │                     │                      │                     │
        ▼                     ▼                      ▼                     ▼
   macro_view ────────► allocation ─────────► risk_verdict ────────► proposed_orders

  Tutti e quattro leggono e scrivono UN solo FundState tipizzato.
  Nessuno di loro chiama un altro.

Ogni operazione che l'agente di esecuzione propone attraversa, in quest'ordine:

  1. Pre-check deterministico  — condizioni di contesto valutate SENZA chiamare un LLM
                                 (kill-switch, limiti duri). I flag deterministici
                                 prevalgono sempre sull'instradamento probabilistico.
  2. Validazione a regole      — limiti di posizione, esposizione, concentrazione.
                                 Motore a regole CLASSICO, NON dimostrazione formale.
  3. Gate di approvazione      — interrupt() dell'orchestratore; firma una persona.
                                 Senza broker configurato questo è l'ULTIMO passo:
                                 approvare significa "approvato perché tu lo esegua".
  4. Adapter del broker        — SOLO se un broker è configurato. Solo qui un ordine
                                 vero lascia il sistema.

  Ogni transizione di stato ha un checkpoint → traccia di audit completa e rigiocabile.

I passi 1-3 non dipendono da un broker. Senza broker configurato il sistema si ferma a una
lista di ordini firmata e l'utente li esegue da sé; l'agente di esecuzione fa comunque il
suo lavoro — dimensionamento, scheduling e impatto atteso restano consigli per chi preme il
pulsante.
```

L'output LLM di ogni agente attraversa un **confine di output strutturato e tipizzato (BAML)**,
così il resto del sistema riceve oggetti validati e tipizzati — mai testo libero.

§6.1-§6.3 descrivono come i quattro agenti sono *cablati*. Il cablaggio è metà del sistema:
**§6.4 dice cosa ciascuno calcola** e di quale corpo di teoria è l'implementazione. Un agente di
cui si conosce il campo di output ma non il contenuto è uno slot, non un disegno.

### 6.1 Come comunicano gli agenti (il contratto a livello di codice)

Il requisito è che ogni agente resti una **unità chiaramente indipendente e facilmente
manutenibile** pur restando pienamente connessa alle altre. La risposta è un pattern
**blackboard / mediator**, *non* un protocollo di rete (§6.2 per il perché A2A è rimandato):

> **Gli agenti non si parlano mai direttamente. Leggono e scrivono soltanto un unico oggetto
> di stato condiviso e tipizzato, e un orchestratore decide chi gira quando.**

Tre regole lo impongono:

1. **Nessun agente importa un altro agente.** `risk.py` non importa mai `allocation.py`. Zero
   import incrociati è ciò che rende ogni agente indipendente: cambiarne uno non può romperne
   un altro.
2. **Gli agenti comunicano solo attraverso l'oggetto di stato tipizzato.** Un agente legge i
   campi che gli servono e scrive i campi che produce. Quell'oggetto di stato *è* il protocollo.
   Le interne di un agente si possono riscrivere liberamente finché il contratto di stato è
   rispettato.
3. **Ogni agente implementa una sola interfaccia (una port).** L'orchestratore tratta tutti e
   quattro in modo identico e può sostituirne uno con un fake nei test cambiando una riga.

**Il contratto di stato.** `FundState` vive in `domain/fund/` e porta un campo per produttore.
Ogni campo è opzionale e vuoto finché il suo produttore non ha girato, ed è questo a rendere una
deliberazione parziale uno stato legittimo invece che un errore:

| Campo | Scritto da | Contiene |
|---|---|---|
| `macro_view` | agente macro | la lettura del regime e le view che ne derivano |
| `allocation` | agente allocazione | i pesi proposti e come sono stati derivati |
| `risk_verdict` | agente rischio | la valutazione di quell'allocazione contro i limiti |
| `proposed_orders` | agente esecuzione | gli ordini che porterebbero lì il portafoglio |
| `approvals` | il gate umano | ciò che una persona ha esplicitamente autorizzato |

**Il contratto d'agente.** `ports/agent.py` dichiara una sola operazione: prendi lo stato,
restituisci lo stato. Nient'altro. Ogni agente, visto da fuori, è identico agli altri.

**Cosa può toccare un agente.** Legge i campi che gli servono, calcola con i propri strumenti e
restituisce uno stato con il proprio campo riempito. Non scrive mai il campo di un altro agente
e non importa mai un altro agente. L'agente di rischio legge `allocation` e `macro_view` e
scrive `risk_verdict`; non ha alcun modo di mettere le mani su come i primi due sono stati
prodotti.

`FundState` è immutabile, tipizzato e serializzabile, quindi funge anche da **registro di
audit**: lo stesso oggetto che disaccoppia gli agenti è quello di cui si fa checkpoint a ogni
passo. Un oggetto solo consegna sia il disaccoppiamento sia la verificabilità.

**Perché questa forma:** agenti che si chiamano a vicenda si saldano insieme (cambiane uno,
rompili tutti); un protocollo di rete (A2A) compra indipendenza al prezzo di una tassa da
sistemi distribuiti che qui non serve ancora. Il pattern blackboard dà indipendenza *e*
connessione a costo infrastrutturale zero — ed è esattamente quello che LangGraph implementa
nativamente (nodi + `State` tipizzato + archi), con checkpoint e gate di interrupt sopra.

**Il ritorno in manutenzione:** ogni agente è indipendentemente **testabile** (dai un
`FundState`, asserisci il `FundState` in uscita — nessun altro agente serve), **sostituibile**
(riscrivi il file, rispetta il contratto), **leggibile** (un file = tutto il lavoro di un
agente) e **rimpiazzabile più avanti da un'implementazione remota** (§6.2) senza che
l'orchestratore o gli altri agenti se ne accorgano.

### 6.2 A2A / MCP come adapter futuro (rimandato, non adottato)

Il protocollo A2A (Agent2Agent) è stato valutato e **rimandato**. A2A esiste per far scoprire e
chiamare fra loro agenti *opachi*, costruiti da *fornitori, framework o linguaggi diversi*, su
JSON-RPC/HTTP. I nostri quattro agenti sono una squadra sola, un repo solo, un linguaggio solo,
e condividono tipi e database: nessuna delle premesse di A2A vale, e la lista "overkill quando"
della specifica stessa (accoppiamento stretto, sincronia, stessa infrastruttura, nessun
isolamento multi-tenant) descrive esattamente la nostra situazione. Adottarlo adesso vorrebbe
dire costruire un protocollo da sistemi distribuiti perché le nostre funzioni possano chiamarsi
fra loro.

Poiché ogni agente sta già dietro la port `Agent` (§6.1), adottare A2A più avanti è **un solo
adapter**, non una riscrittura: un adapter che soddisfa `Agent.run()` chiamando un agente remoto
via A2A e restituendo un `FundState`. I motivi veri per riaprire la questione: (a) un agente di
terze parti o di proprietà esterna deve entrare nel grafo; (b) un agente deve essere scritto in
un altro linguaggio; (c) squadre separate possiedono agenti separati dietro un contratto stabile;
(d) gli agenti vengono pubblicati per essere scoperti dall'esterno. Nessuno è vero oggi.

Correlato: **MCP** (Model Context Protocol) è lo standard complementare per l'accesso *agente →
strumenti* (A2A è *agente → agente*). Se il requisito "gli agenti agiscono attraverso le nostre
API" dovesse mai richiedere un'interfaccia strumenti standardizzata, MCP è lo standard più
pertinente da valutare, prima di A2A. Per ora gli agenti chiamano use-case interni direttamente
attraverso il service layer (§6.3).

### 6.3 Come gli agenti usano gli strumenti (sicuro anche con modelli locali)

"Gli agenti compiono azioni" significa chiamare strumenti — per esempio l'agente di rischio che
calcola il rischio su un'allocazione proposta. Il disegno è modellato da un fatto duro: **i
modelli piccoli self-hosted (7-30B) sono inaffidabili nel tool-calling nativo** — saltano
chiamate, allucinano nomi di strumenti ed emettono argomenti JSON malformati. Quindi **non**
diamo all'LLM un ciclo autonomo di tool-calling (ReAct). Invece:

**1. Gli strumenti sono guidati dal codice per default — non è l'LLM a decidere quando girano.**
Il codice dell'orchestratore o dell'agente invoca lo strumento in modo deterministico e passa il
*risultato* all'LLM perché lo interpreti. In concreto per l'agente di rischio: il calcolo del
rischio (una chiamata `portopt`) **gira sempre** sull'allocazione proposta; il modello locale
riceve soltanto i numeri e scrive il giudizio `RiskVerdict`. La matematica del rischio non
dipende mai dal fatto che il modello scelga di chiamarla, e un modello debole non può saltarla in
silenzio.

**2. Quando esiste una scelta vera, l'LLM emette un *intento* tipizzato, non una tool call
nativa.** Il modello restituisce un **oggetto tipizzato BAML** con un campo enum `tool` più
argomenti tipizzati — vincolato dalla grammatica, quindi un modello debole *non può* produrre un
nome di strumento invalido o argomenti non parsabili. Il codice valida e smista. **Non** usiamo
l'API di function-calling nativa di Ollama (duplica il ruolo di BAML ed è meno affidabile sui
modelli piccoli). Questo è "scegli da un menù", non tool-calling a cielo aperto.

**3. Uno strumento è una chiamata Python in-process dietro una port.** Ogni strumento è uno
use-case, una chiamata `portopt` o un metodo di port — niente rete, niente shell, niente
sottoprocessi. Il registro è un **piccolo insieme fisso, registrato a mano a build time**
(calcola rischio, lancia ottimizzazione, scarica dati, proponi ordini). Nessuna scoperta a
runtime, nessun MCP. Gli agenti raggiungono gli strumenti attraverso gli stessi use-case del
service layer che l'API espone, così il percorso agente e il percorso HTTP condividono una sola
superficie d'azione.

**4. Il fallimento è valida → riprova → fallisci in sicurezza.** Pydantic valida l'intento
tipizzato; in caso di fallimento il passo riprova N volte con l'errore di validazione rimesso nel
prompt; se continua a fallire, il passo dell'agente **si interrompe e fa emergere il
fallimento** nel log di audit — mai un tiro a indovinare silenzioso. Un fallimento di intento può
opzionalmente scalare al gate umano attraverso l'interrupt di LangGraph, riusando il substrato
HITL di §9.

**Confine di fiducia.** Gli strumenti accessibili all'LLM sono **di sola lettura o di puro
calcolo** (scarica dati, calcola rischio, lancia ottimizzazione). Le azioni con effetti
collaterali — piazzare un ordine, spendere denaro — **non sono mai** strumenti chiamabili
dall'LLM: passano dal guardrail deterministico e dal gate di approvazione umana. L'LLM propone;
non preme mai un grilletto con effetti sul mondo reale.

### 6.4 Cosa calcola ciascun agente

Ogni agente è l'implementazione di uno specifico corpo di teoria di portafoglio, e il suo campo
di stato è il portatore tipizzato dell'output di quella teoria. Questa sottosezione dice, per
agente, cosa viene calcolato, cosa deve essere **calcolo deterministico** e cosa è lasciato al
modello — la divisione è quella che §6.3 già impone, applicata campo per campo. I riferimenti ai
capitoli sono a `docs/theory/`.

Una regola vale per tutti e quattro: **nessun numero che entra in una decisione è scritto da un
modello linguistico.** Probabilità di regime, pesi, misure di rischio e piani di trading sono
tutti output di solver o stimatori invocati da codice. Il compito del modello è scegliere fra
metodi con un nome, fornire il giudizio che la teoria stessa richiede, ed emettere l'oggetto
tipizzato. È la regola 1 di §6.3 letta come regola di disegno e non come aggiramento di
robustezza.

#### L'agente macro — `agents/macro.py` → `macro_view`

*Teoria: capitoli 21, 26, 28, 29.*

Risponde a una domanda sola: **in che stato si trova il mercato, e cosa implica quello stato sui
rendimenti attesi e sulla struttura di covarianza?** Esiste perché l'insieme delle opportunità
non è costante — medie, volatilità e, cosa decisiva, correlazioni cambiano con lo stato, e un
allocatore fermo sulla frontiera incondizionata è fermo sulla media delle frontiere per regime.

| Grandezza | Cos'è | Prodotta da |
|---|---|---|
| Probabilità di regime filtrate | la probabilità di ciascuno stato data l'informazione disponibile adesso, aggiornata ricorsivamente da un filtro markoviano a cambio di regime; gli stati sono interpretabili economicamente (crollo, crescita lenta, rialzo, ripresa) con medie, volatilità **e segni di correlazione** distinti | stimatore deterministico |
| Indicatori di stress osservabili | turbolenza finanziaria (una distanza di Mahalanobis sul vettore di rendimenti contemporanei), scomposta in sorpresa di magnitudine e sorpresa di correlazione; absorption ratio come misura di fragilità; rango effettivo come dimensionalità effettiva | stimatore deterministico |
| Premi al rischio attesi condizionati | pendenza della struttura a termine → probabilità di recessione e crescita reale; spread creditizi scomposti in perdita attesa ed excess bond premium; scalatura dell'esposizione a varianza inversa; fattori dinamici compressi da molte serie macro; combinazione di molte previsioni deboli | stimatore deterministico |
| Punteggio out-of-sample e vincoli | R² out-of-sample contro la media storica, più vincoli di segno, troncamento a non-negatività del premio e ancoraggio alla valutazione | stimatore deterministico |
| La view vera e propria | la matrice di mappatura, il vettore delle view e **l'incertezza sulle view** — la tripla che un aggiornamento bayesiano richiede | modello, a partire da quanto sopra |

Tre conseguenze vincolano il contratto di stato.

**Una view senza una confidenza dichiarata non è una view.** Il posterior che l'agente di
allocazione calcola è una media pesata fra il prior di equilibrio e le view, e i pesi sono
l'incertezza del prior e l'incertezza *della view*. Un `MacroView` che porta solo una direzione è
inutilizzabile a valle: deve portare la dispersione che dice quanto pesarci sopra.

**Previsione e preferenza sono campi diversi e devono restare campi diversi.** Il trattamento
standard di una covarianza a due stati mescola una covarianza di stato calmo e una di stato
stressato usando una probabilità dello stato stressato, e separatamente pesa i due stati con due
parametri di avversione al rischio. La probabilità è una *previsione* e appartiene all'agente
macro; l'avversione è un *atteggiamento* e appartiene all'obiettivo che l'utente configura.
Collassarle in un numero solo permetterebbe a un modello di cambiare in silenzio la postura di
rischio dell'utente.

**Lo stimatore di regime è condiviso, ma è uno strumento, non un agente.** Lo stesso stato
latente che condiziona l'allocazione fra classi di attivo condiziona anche la selezione
trasversale: il momentum è profittevole dopo mercati in salita e crolla all'uscita dai rimbalzi
di mercato orso, il premio value è controciclico, il premio size inverte segno fra regimi.
Entrambi gli agenti hanno bisogno di quello stato. La regola di §6.1 sul divieto di import
incrociati si preserva mettendo lo stimatore nel service layer come use-case che entrambi gli
agenti chiamano (regola 3 di §6.3), non facendo leggere a un agente il campo dell'altro.

#### L'agente di allocazione — `agents/allocation.py` → `allocation`

*Teoria: capitoli 2, 3, 17, 30.*

Trasforma una view condizionata in un vettore di pesi. Esiste perché il passo da "il premio
sembra alto" a "tienine questa quantità" non è né ovvio né unico: esiste un menù documentato di
costruzioni con profili di errore di stima diversi, e scegliere fra queste è la decisione che
l'agente sta prendendo.

| Costruzione | Quando si applica | Cosa richiede |
|---|---|---|
| Media-varianza vincolata | esiste un vettore di rendimenti attesi affidabile | rendimenti attesi, covarianza, insieme dei vincoli |
| Posterior di Black-Litterman | esistono view, ma parziali e incerte; l'equilibrio è l'ancora | pesi di mercato (ottimizzati all'indietro in rendimenti di equilibrio), covarianza, la tripla delle view |
| Risk budgeting / equal risk contribution | i rendimenti attesi non sono affidabili; si alloca rischio invece di capitale | covarianza e un vettore di budget |
| Politiche a pesi diretti | molti attivi, caratteristiche disponibili; l'anello debole è il modello di rendimento | caratteristiche, un piccolo vettore di parametri, un obiettivo |

Quattro conseguenze vincolano il disegno.

**I pesi escono da un solver.** L'ottimizzazione è una chiamata `portopt` che gira sempre (regola
1 di §6.3); il modello sceglie la costruzione e registra il razionale nella metà "come sono stati
derivati" del campo `allocation`. Quel campo non è decorazione: è ciò che rende la scelta
verificabile, perché la scelta è dove stava il giudizio.

**I vincoli fanno parte dello stimatore, non dell'interfaccia.** Il divieto di vendita allo
scoperto e i limiti di posizione sono formalmente equivalenti a uno shrinkage della matrice di
covarianza; cambiano la stima, non solo la presentazione. Un ottimizzatore plug-in è un
massimizzatore di errore — mette peso positivo grande su qualunque cosa abbia l'errore di stima
positivo più grande — e l'insieme dei vincoli, lo shrinkage e la struttura fattoriale sono i tre
rimedi con un nome. Appartengono alla derivazione registrata.

**Il risk budgeting è ben posto solo sotto condizioni.** Il portafoglio a budget di rischio
esiste ed è unico quando la matrice di covarianza è definita positiva e ogni budget è
strettamente positivo; la sua volatilità sta fra quella del portafoglio a minima varianza e
quella del portafoglio a budget sui pesi. Sono precondizioni verificabili, non speranze.

**Una deliberazione è una decisione a un periodo.** La politica dinamica ottima si scompone in una
domanda miope e una domanda di copertura intertemporale, e il secondo termine esiste solo quando
un orizzonte e le dinamiche dello stato sono dentro il problema. `FundState` oggi non porta alcun
orizzonte, quindi ciò che l'agente di allocazione produce è il termine miope. È una scelta
legittima e comune; dev'essere una scelta dichiarata.

#### L'agente di rischio — `agents/risk.py` → `risk_verdict`

*Teoria: capitoli 4, 20, 23, 24.*

**Perché esiste.** L'agente di allocazione ha ottimizzato contro momenti stimati, dentro il
campione, usando una misura che ha scelto lui. L'agente di rischio valuta quella proposta sulle
cose che l'ottimizzatore non ha ottimizzato, ed è un *nodo separato che non può riscrivere
l'allocazione* proprio perché il valore dell'esercizio sta nell'essere una prospettiva
indipendente e complementare — la stessa ragione per cui un programma di stress testing dev'essere
indipendente dal modello di cui sta testando l'output. Un agente in grado di modificare
l'allocazione si correggerebbe i compiti da solo.

| Valutazione | Cosa aggiunge, che l'ottimizzatore non aveva |
|---|---|
| Misura di coda coerente | il Value-at-Risk non è subadditivo, quindi un limite espresso in VaR può penalizzare un'operazione che *diversifica*; la famiglia conditional / expected shortfall è coerente e risponde a "quando si rompe, quanto fa male", non a "quanto spesso" |
| Rischio di percorso | il drawdown è funzione del massimo corrente, quindi non è ricavabile in alcun modo dalla distribuzione marginale; la famiglia conditional drawdown è convessa, si riduce a un programma lineare e interpola fra drawdown medio e massimo |
| Attribuzione | i contributi di Eulero sono l'unica allocazione del rischio totale compatibile con il RORAC e sommano esattamente a esso, localizzando il rischio su posizioni, fattori o sotto-portafogli |
| Diagnostica di concentrazione | sotto subadditività il contributo di Eulero di un componente non supera mai il suo rischio stand-alone; indici di diversificazione vicini al 100% segnalano libri quasi comonotoni; il factor risk impact generalizza il coefficiente di determinazione |
| Stress | una ricerca sistematica dello scenario plausibile peggiore, più l'identificazione dei fattori chiave che guidano quella perdita |

Quattro conseguenze vincolano il disegno.

**I limiti andrebbero espressi su una misura coerente.** Altrimenti il guardrail può rifiutare
un'operazione che riduce il rischio. In un contesto gaussiano i limiti in volatilità, VaR ed
expected shortfall differiscono solo per una costante moltiplicativa e i contributi *relativi*
coincidono, quindi un insieme di limiti si può riesprimere fra i tre — ma solo lì.

**Riportare i contributi in expected shortfall, non quelli in VaR.** I primi sono elementari
attese condizionate su un evento di probabilità positiva; i secondi condizionano su un evento di
misura nulla, richiedono uno smoothing a kernel con scelta di bandwidth e sono molto volatili. È
un'istruzione concreta su quale numero il verdetto porta.

**Lo stress è una ricerca, non un brainstorming.** Gli scenari scelti a mano falliscono in due
direzioni: includono cose implausibili e mancano cose plausibili. La forma disciplinata fissa una
regione di plausibilità come ellissoide di raggio di Mahalanobis limitato, poi *minimizza* il
profitto atteso condizionato su quella regione; il raggio gioca per lo stress test il ruolo che
il livello di confidenza gioca per il VaR. I fattori non fissati in uno scenario parziale
prendono la loro attesa condizionata, che è dimostrabilmente il completamento più plausibile. Il
reverse stress testing è la stessa macchineria girata all'indietro a partire da un esito che
minaccia la sopravvivenza. Tutto questo è ottimizzazione deterministica, che è esattamente ciò
che a un modello locale debole non va chiesto di improvvisare.

**All'agente serve un percorso di rendimenti, non solo i momenti.** Le misure di drawdown sono
definite su un percorso campionario di rendimenti cumulati di portafoglio. È un requisito reale
su `ports/market_data.py`, non un dettaglio implementativo.

#### L'agente di esecuzione — `agents/execution.py` → `proposed_orders`

*Teoria: capitoli 11, 22.*

Trasforma un portafoglio obiettivo in ordini. Il suo contenuto centrale è un'affermazione che
l'architettura deve dire chiaramente: **`proposed_orders` non è `allocation` meno le posizioni
correnti.**

| Grandezza | Cos'è |
|---|---|
| L'obiettivo verso cui operare | non il portafoglio ottimo corrente, ma un portafoglio *aim*: l'ottimo calcolato come se ogni segnale predittivo fosse scontato per la propria velocità di ritorno alla media, così i segnali a decadimento lento pesano di più |
| Quanto muoversi | una frazione fissa della distanza dall'aim a ogni periodo — la frazione cresce con l'avversione al rischio e cala con i costi di transazione, ed è indipendente dalla posizione corrente e dalla sua storia |
| Turnover atteso | prevedibile ex ante dal mix di segnali: scala con la tracking error obiettivo, con la radice del numero di titoli e con la radice di uno meno l'autocorrelazione trasversale del segnale |
| La traiettoria | il piano che minimizza l'implementation shortfall contro il rischio di timing; sotto impatto lineare decade iperbolicamente con una emivita fissata da liquidità e avversione al rischio **e non dall'orizzonte imposto**; l'impatto empirico è concavo, più vicino a una legge a radice quadrata, il che cambia la forma del piano |
| La stima di costo | impatto permanente (l'inferenza di un market maker dal flusso ordini — il suo reciproco è la profondità di mercato) più impatto temporaneo più spread |

Quattro conseguenze vincolano il disegno.

**Il trading parziale è il punto.** Inseguire completamente un obiettivo che si muove non è
ottimo quando operare costa; la politica ottima muove una frazione fissa verso un obiettivo
pesato per la persistenza. Un agente di esecuzione che emette la differenza piena sta
implementando una strategia diversa e peggiore.

**Lo scheduling è a livello di portafoglio.** Con più titoli, il costo atteso si separa per
titolo solo se le matrici di impatto sono diagonali, ma il termine di rischio accoppia tutto il
sistema attraverso la matrice di covarianza. Gli ordini non si possono pianificare un ticker alla
volta, a meno che la covarianza non sia diagonale, e non lo è.

**Un intento diventa molti ordini figli.** La traiettoria è una sequenza, il che moltiplica
l'esposizione agli ordini duplicati già registrata in §12 e interagisce con la trappola di
idempotenza al resume di §9 (un nodo di esecuzione interrotto riparte dal suo inizio). Qualunque
schema di idempotenza si adotti deve avere per chiave l'ordine figlio, non solo l'intento.

**Senza broker configurato questo lavoro è consiglio — e adesso ha un nome.** Il dimensionamento
è la mossa parziale verso l'aim; lo scheduling è la traiettoria; l'impatto di mercato atteso è la
somma dei termini permanente, temporaneo e di spread. È questo che contiene la frase sul
"consiglio per chi preme il pulsante".

#### Il guardrail — `agents/guardrail.py`

*Teoria: capitolo 23.*

Il guardrail non è il quinto agente: è il gate deterministico dei passi 1 e 2 qui sopra, e non
contiene alcun modello. Ciò che il capitolo 23 gli dà è la *forma* che i limiti dovrebbero avere.

- **I limiti di esposizione e concentrazione si esprimono correttamente come vincoli sui
  contributi al rischio** — il peso di ogni posizione moltiplicato per il suo rischio marginale,
  tenuto entro un budget, con budget e pesi non negativi e ciascuno normalizzato a uno. Sorvegliare
  un libro ex ante significa controllare congiuntamente peso, beta e volatilità di portafoglio,
  perché il contributo è il loro prodotto.
- **Un budget di rischio nullo non significa "non tenerne affatto".** Un contributo nullo ammette
  una soluzione con posizione non nulla ogni volta che le correlazioni sono negative. "Nessuna
  esposizione a X" va implementato togliendo X dall'universo, non mettendo il suo budget a zero.
- **Un insieme di limiti può essere infeasible**, e le condizioni sono esplicite: matrice di
  covarianza definita positiva e budget strettamente positivi. Infeasible è un terzo esito,
  distinto da pass e da fail, e lo strato deterministico è dove deve emergere.
- **I contributi si calcolano a monte.** Verificare un limite sui contributi richiede la matrice
  di covarianza, che non è qualcosa che un validatore di campo abbia in mano. Coerentemente con la
  regola 1 di §6.3, i contributi sono prodotti da calcolo deterministico prima del gate; il gate
  valida numeri. Questo mantiene il guardrail uno strato di regole classico, come §11 richiede.

<!-- GENERATO: fund -->

_Da qui in giù la sezione è generata da `scriptorium arch` il 2026-08-19: non modificarla a mano, si riscrive a ogni assemblaggio. Il record completo — responsabilità estese e citazioni riga per riga — sta in `data/architecture-spec/`, da cui questo è renderizzato._

### 6.1 Albero

75 file, ciascuno annotato col suo ruolo esagonale. Ruoli: config 3 · CORE 40 · DRIVEN 23 · DRIVING 4 · PORTS 5

```
└── fund/
    ├── app/
    │   ├── adapters/
    │   │   ├── broker/  ┃ DRIVEN · planned
    │   │   │   └── trading212/
    │   │   │       ├── __init__.py  ┃ DRIVEN · planned
    │   │   │       ├── adapter.py  ┃ DRIVEN · planned
    │   │   │       ├── client.py  ┃ DRIVEN · planned
    │   │   │       ├── config.py  ┃ DRIVEN · planned
    │   │   │       ├── errors.py  ┃ DRIVEN · planned
    │   │   │       └── mapping.py  ┃ DRIVEN · planned
    │   │   ├── checkpointer/  ┃ DRIVEN · planned
    │   │   ├── llm/  ┃ DRIVEN · planned
    │   │   ├── persistence/
    │   │   │   ├── __init__.py  ┃ DRIVEN · planned
    │   │   │   ├── _shared/
    │   │   │   │   ├── __init__.py  ┃ DRIVEN · planned
    │   │   │   │   └── session.py  ┃ DRIVEN · planned
    │   │   │   ├── market_data_repository.py  ┃ DRIVEN · planned
    │   │   │   └── models/
    │   │   │       ├── __init__.py  ┃ DRIVEN · planned
    │   │   │       ├── macro.py  ┃ DRIVEN · planned
    │   │   │       ├── market_data.py  ┃ DRIVEN · planned
    │   │   │       └── universe.py  ┃ DRIVEN · planned
    │   │   └── portfolio/
    │   │       ├── __init__.py  ┃ DRIVEN · planned
    │   │       ├── csv_import.py  ┃ DRIVEN · planned
    │   │       ├── manual_entry.py  ┃ DRIVEN · planned
    │   │       ├── models.py  ┃ DRIVEN · planned
    │   │       └── repository.py  ┃ DRIVEN · planned
    │   ├── agents/
    │   │   ├── allocation.py  ┃ CORE · planned
    │   │   ├── execution.py  ┃ CORE · planned
    │   │   ├── guardrail.py  ┃ CORE · planned
    │   │   ├── macro.py  ┃ CORE · planned
    │   │   ├── orchestrator.py  ┃ CORE · planned
    │   │   ├── prompts/  ┃ CORE · planned
    │   │   └── risk.py  ┃ CORE · planned
    │   ├── domain/
    │   │   ├── _shared/  ┃ CORE · planned
    │   │   ├── fund/
    │   │   │   ├── __init__.py  ┃ CORE · planned
    │   │   │   ├── allocation.py  ┃ CORE · planned
    │   │   │   ├── approval.py  ┃ CORE · planned
    │   │   │   ├── fund_state.py  ┃ CORE · planned
    │   │   │   ├── macro_view.py  ┃ CORE · planned
    │   │   │   └── risk_verdict.py  ┃ CORE · planned
    │   │   ├── portfolio/
    │   │   │   ├── __init__.py  ┃ CORE · planned
    │   │   │   ├── holding.py  ┃ CORE · planned
    │   │   │   └── portfolio.py  ┃ CORE · planned
    │   │   └── trading/
    │   │       ├── __init__.py  ┃ CORE · planned
    │   │       ├── guardrail_rule.py  ┃ CORE · planned
    │   │       ├── kill_switch.py  ┃ CORE · planned
    │   │       └── order.py  ┃ CORE · planned
    │   ├── http/
    │   │   ├── app.py  ┃ DRIVING · planned
    │   │   ├── deps.py  ┃ DRIVING · planned
    │   │   ├── routers/  ┃ DRIVING · planned
    │   │   └── sse.py  ┃ DRIVING · planned
    │   ├── main.py  ┃ config · planned
    │   ├── ports/
    │   │   ├── agent.py  ┃ PORTS · planned
    │   │   ├── broker.py  ┃ PORTS · planned
    │   │   ├── checkpointer.py  ┃ PORTS · planned
    │   │   ├── llm.py  ┃ PORTS · planned
    │   │   └── market_data.py  ┃ PORTS · planned
    │   ├── services/
    │   │   ├── _shared/
    │   │   │   ├── __init__.py  ┃ CORE · planned
    │   │   │   ├── _idempotency.py  ┃ CORE · planned
    │   │   │   ├── _json_safe.py  ┃ CORE · planned
    │   │   │   ├── _progress.py  ┃ CORE · planned
    │   │   │   ├── _solver_guard.py  ┃ CORE · planned
    │   │   │   ├── audit_log.py  ┃ CORE · planned
    │   │   │   └── broker_gate.py  ┃ CORE · planned
    │   │   ├── fund/
    │   │   │   ├── __init__.py  ┃ CORE · planned
    │   │   │   ├── approve_trade.py  ┃ CORE · planned
    │   │   │   └── run_deliberation.py  ┃ CORE · planned
    │   │   └── optimization/
    │   │       ├── __init__.py  ┃ CORE · planned
    │   │       ├── assess_risk.py  ┃ CORE · planned
    │   │       ├── estimate_regime.py  ┃ CORE · planned
    │   │       ├── propose_orders.py  ┃ CORE · planned
    │   │       ├── run_optimization.py  ┃ CORE · planned
    │   │       └── solver_result.py  ┃ CORE · planned
    │   └── tools/
    │       ├── intent.py  ┃ CORE · planned
    │       ├── registry.py  ┃ CORE · planned
    │       └── run.py  ┃ CORE · planned
    ├── baml_client/  ┃ DRIVEN · planned
    ├── baml_src/  ┃ config · planned
    └── tests/  ┃ config · planned
```

### 6.2 I file

fund/ è l'unico bounded context, fra i quattro, in cui il core (domain + services + agents + tools) è realmente non banale, quindi la regola di dipendenza verso l'interno va applicata con disciplina file per file. domain/ (fund, trading, _shared) non importa nulla: FundState, MacroView, Allocation, RiskVerdict, Approval, Order, Position, GuardrailRule, KillSwitch sono dataclass/pydantic pure. ports/ (agent.py, broker.py, llm.py, market_data.py, checkpointer.py) sono le uniche interfacce che core e driven possono condividere: ogni servizio, agente o tool che ha bisogno del mondo esterno (LLM, broker, dati di mercato, persistenza dello stato) dichiara la propria dipendenza come porta, mai come import diretto di un adapter. services/ (optimization, fund, _shared) e agents/ (orchestrator, macro, allocation, risk, execution, guardrail, prompts) e tools/ (registry, intent, run) sono tutti CORE: si chiamano fra loro, chiamano domain/ e chiamano ports/, mai adapters/. L'unica eccezione esplicitamente ammessa è services/optimization/ che importa direttamente la libreria portopt (bounded context 'optimizer/'): è un'importazione core-a-core legittima perché portopt è essa stessa un dominio puro senza DB/HTTP/LLM (§2 dell'architettura), non un adapter — per questo è annotata come external:portopt anziché come violazione. adapters/ (persistence, broker, llm, checkpointer) sono DRIVEN: implementano le rispettive ports/ e sono gli unici file che parlano con sistemi reali (external:postgres, external:trading212, external:ollama) o con codice generato (baml_client). http/ (app.py, routers/, sse.py) e main.py sono DRIVING: chiamano services/ (mai domain/ o agents/ direttamente) e non contengono logica di dominio. deps.py è il composition root: è l'unico punto in cui un file del lato driving referenzia direttamente gli adapters/ per costruirli e iniettarli nei services/ tramite le ports/ — questo è ammesso perché la regola vietata riguarda solo core→driven, non driving→driven. baml_src/ è config (sorgente dei prompt/funzioni compilato da BAML) e baml_client/ è driven (client generato, 'non modificare a mano', usato solo da adapters/llm/). tests/ è config e per costruzione attraversa i confini con fake che sostituiscono le ports/, cosa esplicitamente prevista dall'architettura per rendere ogni agente testabile in isolamento.

| File | Ruolo | Contiene | Si collega a |
|---|---|---|---|
| `fund/app/adapters/broker/` | DRIVEN ·  pianificato | Client httpx reale verso l'API REST di Trading 212, implementazione concreta di ports/broker.py. | `fund/app/ports/broker.py`, `external:trading212` |
| `fund/app/adapters/broker/trading212/__init__.py` | DRIVEN ·  pianificato | Punto di ingresso pubblico del package trading212: ri-esporta la classe che implementa `ports/broker.py` per questo venue. | `fund/app/adapters/broker/trading212/adapter.py` |
| `fund/app/adapters/broker/trading212/adapter.py` | DRIVEN ·  pianificato | L'implementazione concreta di `ports/broker.py` per Trading 212: il file che compone client, mapping e config in un adapter registrabile. | `fund/app/ports/broker.py`, `fund/app/adapters/broker/trading212/client.py`, `fund/app/adapters/broker/trading212/mapping.py`, `fund/app/adapters/broker/trading212/config.py`, `fund/app/adapters/broker/trading212/errors.py` |
| `fund/app/adapters/broker/trading212/client.py` | DRIVEN ·  pianificato | Wrapper httpx sottile sulla REST API pubblica di Trading 212: solo trasporto HTTP, nessuna conoscenza del dominio. | `fund/app/adapters/broker/trading212/config.py`, `external:trading212` |
| `fund/app/adapters/broker/trading212/config.py` | DRIVEN ·  pianificato | Oggetto di configurazione del venue: ambiente (demo/live), base URL corrispondente e chiave API. | — |
| `fund/app/adapters/broker/trading212/errors.py` | DRIVEN ·  pianificato | Eccezioni tipizzate proprie del package, che traducono i fallimenti di trasporto e di venue in un vocabolario che il core può intercettare senza conoscere httpx o le specificità di Trading 212. | `fund/app/adapters/broker/trading212/client.py` |
| `fund/app/adapters/broker/trading212/mapping.py` | DRIVEN ·  pianificato | Traduzione fra il vocabolario di dominio (`Order`, `Holding`/`Portfolio`) e la forma JSON specifica delle API di Trading 212. | `fund/app/domain/trading/`, `fund/app/domain/portfolio/` |
| `fund/app/adapters/checkpointer/` | DRIVEN ·  pianificato | PostgresSaver, implementazione concreta di ports/checkpointer.py basata su langgraph-checkpoint-postgres. | `fund/app/ports/checkpointer.py`, `external:postgres` |
| `fund/app/adapters/llm/` | DRIVEN ·  pianificato | Client Ollama + BAML, implementazione concreta di ports/llm.py. | `fund/app/ports/llm.py`, `fund/baml_client/`, `external:ollama` |
| `fund/app/adapters/persistence/__init__.py` | DRIVEN ·  pianificato | Punto di ingresso pubblico del package persistence: la superficie che il resto di fund/ importa per ottenere l'adapter di lettura sulla base dati dell'ingestion. | `fund/app/adapters/persistence/market_data_repository.py` |
| `fund/app/adapters/persistence/_shared/__init__.py` | DRIVEN ·  pianificato | Punto di ingresso del sotto-package _shared: raccoglie l'infrastruttura di sessione condivisa dall'unico repository del package. | `fund/app/adapters/persistence/_shared/session.py` |
| `fund/app/adapters/persistence/_shared/session.py` | DRIVEN ·  pianificato | Fabbrica della connessione al Postgres condiviso con l'ingestion: apre l'engine SQLAlchemy e la sessione che ogni lettura di questo package usa per interrogare le tabelle che il daemon di ingestion popola. | `external:postgres` |
| `fund/app/adapters/persistence/market_data_repository.py` | DRIVEN ·  pianificato | L'unico repository del package: implementa fund/app/ports/market_data.py leggendo prezzi, dati fondamentali e serie macro dalle tabelle che l'ingestion scrive, per servire gli use-case e gli agenti del fund. | `fund/app/ports/market_data.py`, `fund/app/adapters/persistence/_shared/session.py`, `fund/app/adapters/persistence/models/macro.py`, `fund/app/adapters/persistence/models/market_data.py`, `fund/app/adapters/persistence/models/universe.py`, `external:postgres` |
| `fund/app/adapters/persistence/models/__init__.py` | DRIVEN ·  pianificato | Punto di ingresso del sotto-package models: raccoglie le classi ORM di sola lettura mappate sulle tabelle scritte dall'ingestion. | `fund/app/adapters/persistence/models/macro.py`, `fund/app/adapters/persistence/models/market_data.py`, `fund/app/adapters/persistence/models/universe.py` |
| `fund/app/adapters/persistence/models/macro.py` | DRIVEN ·  pianificato | Mapping SQLAlchemy di sola lettura sulle tabelle macro scritte dall'ingestion — rendimenti obbligazionari, indicatori economici, osservazioni FRED, calibrazioni di regime, notizie macro — lo stesso schema fisico di ingestion/app/models/macro/macro_regime.py, ridichiarato nel bounded context fund invece che importato. | `external:postgres` |
| `fund/app/adapters/persistence/models/market_data.py` | DRIVEN ·  pianificato | Mapping SQLAlchemy di sola lettura sulle tabelle di mercato e di bilancio scritte dall'ingestion — prezzi storici, profili, bilanci, dividendi, split, raccomandazioni degli analisti, azionariato, notizie sul titolo — lo stesso schema fisico di ingestion/app/models/market_data/yfinance_data.py. | `external:postgres`, `fund/app/adapters/persistence/models/universe.py` |
| `fund/app/adapters/persistence/models/universe.py` | DRIVEN ·  pianificato | Mapping SQLAlchemy di sola lettura sulle tabelle dell'universo investibile scritte dall'ingestion — borse e strumenti, incluso lo stato di delisting — lo stesso schema fisico di ingestion/app/models/universe/universe.py. | `external:postgres` |
| `fund/app/adapters/portfolio/__init__.py` | DRIVEN ·  pianificato | Punto di ingresso pubblico del pacchetto adapters/portfolio/: raccoglie ed espone le due vie di importazione delle posizioni (CSV, inserimento manuale) e lo strato di persistenza (modello ORM e repository) come un'unica superficie. | `fund/app/adapters/portfolio/csv_import.py`, `fund/app/adapters/portfolio/manual_entry.py`, `fund/app/adapters/portfolio/models.py`, `fund/app/adapters/portfolio/repository.py` |
| `fund/app/adapters/portfolio/csv_import.py` | DRIVEN ·  pianificato | Adattatore DRIVEN che implementa la prima delle due vie di default per popolare le posizioni: importa in blocco un file CSV (un estratto broker o un foglio compilato a mano) e lo traduce in Holding di dominio. | `fund/app/domain/portfolio/holding.py`, `fund/app/domain/portfolio/portfolio.py`, `fund/app/adapters/portfolio/repository.py` |
| `fund/app/adapters/portfolio/manual_entry.py` | DRIVEN ·  pianificato | Adattatore DRIVEN che implementa la seconda via di default per popolare le posizioni: accetta una singola voce inserita a mano (strumento + quantità) invece di un file, e la applica come aggiunta, aggiornamento o rimozione sul Portfolio corrente. | `fund/app/domain/portfolio/holding.py`, `fund/app/domain/portfolio/portfolio.py`, `fund/app/adapters/portfolio/repository.py` |
| `fund/app/adapters/portfolio/models.py` | DRIVEN ·  pianificato | Mappatura ORM (SQLAlchemy) della tabella (o delle tabelle) in cui il fondo persiste le proprie posizioni — lo schema di storage su cui repository.py si appoggia, di proprietà di fund/, distinto dalle tabelle che persistence/ legge dal database di ingestion. | `external:postgres`, `fund/app/adapters/portfolio/repository.py` |
| `fund/app/adapters/portfolio/repository.py` | DRIVEN ·  pianificato | Repository che implementa la lettura del Portfolio corrente e la scrittura/aggiornamento delle singole Holding contro le tabelle di models.py, ed è il punto unico in cui csv_import.py e manual_entry.py convergono e da cui una futura sincronizzazione broker leggerebbe/scriverebbe. | `fund/app/adapters/portfolio/models.py`, `fund/app/domain/portfolio/holding.py`, `fund/app/domain/portfolio/portfolio.py`, `fund/app/adapters/portfolio/csv_import.py`, `fund/app/adapters/portfolio/manual_entry.py`, `fund/app/adapters/broker/`, `external:postgres` |
| `fund/app/agents/allocation.py` | CORE ·  pianificato | Il nodo del grafo dei quattro agenti che traduce una view condizionata (letta da `FundState.macro_view`) in un vettore di pesi di portafoglio (scritto in `FundState.allocation`), scegliendo fra il menu di costruzioni che la teoria mette a disposizione — media-varianza vincolata, Black-Litterman, risk budgeting/ERC, politiche parametriche dirette — e delegando ogni calcolo numerico a `portopt` attraverso il layer di servizio. | `fund/app/ports/agent.py`, `fund/app/domain/fund/`, `fund/app/services/optimization/`, `fund/app/services/fund/`, `fund/app/ports/llm.py`, `fund/app/agents/prompts/`, `fund/app/agents/orchestrator.py` |
| `fund/app/agents/execution.py` | CORE ·  pianificato | L'agente di esecuzione del fondo — il quarto e ultimo nodo che l'orchestratore esegue nella sequenza macro → allocation → risk → execution, e l'unico produttore del campo FundState.proposed_orders. Implementa il port Agent (fund/app/ports/agent.py: prende lo stato, restituisce lo stato) e traduce il vettore di pesi obiettivo scritto dall'agente di allocazione in ordini eseguibili, rendendo concreta l'affermazione che l'architettura enuncia esplicitamente su di sé — proposed_orders non è allocation meno le posizioni correnti. Il suo contenuto teorico è la politica di trading dinamico a costi di transazione di Gârleanu-Pedersen (capitolo 11) — un aim portfolio pesato per la persistenza dei segnali e una regola di avvicinamento parziale, mai l'intera differenza — composta con lo scheduling a minimo costo di esecuzione di Kyle, Almgren-Chriss, Bertsimas-Lo, Obizhaeva-Wang e la legge di potenza di Bouchaud-Farmer-Lillo/Marzo-Ritelli-Zagaglia (capitolo 22). | `fund/app/ports/agent.py`, `fund/app/domain/fund/fund_state.py`, `fund/app/domain/fund/allocation.py`, `fund/app/domain/trading/order.py`, `fund/app/domain/portfolio/holding.py`, `fund/app/domain/portfolio/portfolio.py`, `fund/app/ports/llm.py`, `fund/app/tools/registry.py`, `fund/app/tools/run.py`, `fund/app/tools/intent.py`, `fund/app/agents/prompts/`, `fund/app/agents/orchestrator.py`, `fund/app/services/optimization/propose_orders.py`, `optimizer/execution/` |
| `fund/app/agents/guardrail.py` | CORE ·  pianificato | Il cancello deterministico che chiude i passi 1 e 2 della pipeline di trade — pre-check senza LLM e validazione a regole — prima del gate di approvazione umana. Non è un quinto agente e non contiene alcun modello: dichiara i validatori Pydantic v2 (per-campo e cross-campo) che leggono FundState insieme alle regole di dominio GuardrailRule e KillSwitch, e decidono pass, fail o infeasible. | `fund/app/domain/trading/`, `fund/app/domain/fund/`, `fund/app/agents/risk.py`, `fund/app/agents/execution.py`, `fund/app/agents/orchestrator.py` |
| `fund/app/agents/macro.py` | CORE ·  pianificato | L'agente macro del fondo — il primo nodo che l'orchestratore esegue nella sequenza macro → allocation → risk → execution, e l'unico produttore del campo FundState.macro_view. Implementa il port Agent (fund/app/ports/agent.py: prende lo stato, restituisce lo stato) e risponde a una sola domanda — in che stato si trova il mercato, e che cosa implica quello stato per i rendimenti attesi e per la struttura di covarianza — traducendo in un MacroView tipizzato le probabilità di regime filtrate, gli indicatori di stress osservabili e i premi al rischio condizionati attesi che i capitoli 21, 26, 28 e 29 della teoria stabiliscono, insieme alla view — matrice di mappatura, vettore di view e incertezza della view — che l'agente di allocazione ingerirà nel proprio aggiornamento alla Black-Litterman. | `fund/app/ports/agent.py`, `fund/app/domain/fund/fund_state.py`, `fund/app/domain/fund/macro_view.py`, `fund/app/ports/llm.py`, `fund/app/agents/prompts/`, `fund/app/tools/registry.py`, `fund/app/tools/run.py`, `fund/app/services/optimization/estimate_regime.py`, `fund/app/agents/orchestrator.py`, `optimizer/regimes/` |
| `fund/app/agents/orchestrator.py` | CORE ·  pianificato | L'orchestratore hand-rolled del fondo: il mediatore in puro Python — nessun DSL a grafo — che fa girare i quattro agenti (macro, allocazione, rischio, esecuzione) in sequenza sull'unico oggetto FundState condiviso, e che monta quello stesso stato come singolo canale del substrato durevole LangGraph per checkpoint e interrupt del gate di approvazione umana. | `fund/app/ports/agent.py`, `fund/app/agents/macro.py`, `fund/app/agents/allocation.py`, `fund/app/agents/risk.py`, `fund/app/agents/execution.py`, `fund/app/agents/guardrail.py`, `fund/app/domain/fund/fund_state.py`, `fund/app/domain/fund/approval.py`, `fund/app/ports/checkpointer.py`, `fund/app/ports/broker.py` |
| `fund/app/agents/prompts/` | CORE ·  pianificato | Binding delle funzioni BAML usate da ciascun agente (macro, allocation, risk, execution). | `fund/app/ports/llm.py` |
| `fund/app/agents/risk.py` | CORE ·  pianificato | L'agente di rischio del fondo — il terzo nodo che l'orchestratore esegue nella sequenza macro → allocation → risk → execution, e l'unico produttore del campo FundState.risk_verdict. Implementa il port Agent (fund/app/ports/agent.py: prende lo stato, restituisce lo stato) e porta nella deliberazione una prospettiva indipendente e complementare a quella dell'agente di allocazione — che ha ottimizzato sui momenti stimati, dentro il campione, usando una misura che ha scelto essa stessa: il rischio agente valuta la proposta su ciò che l'ottimizzatore non ha ottimizzato. Traduce in un RiskVerdict tipizzato cinque assi distinti che i capitoli 4, 20, 23 e 24 della teoria stabiliscono: la misura di coda coerente, il rischio di traiettoria (drawdown), l'attribuzione di Eulero, la diagnostica di concentrazione e la ricerca sistematica dello scenario di stress peggiore. | `fund/app/ports/agent.py`, `fund/app/domain/fund/fund_state.py`, `fund/app/domain/fund/allocation.py`, `fund/app/domain/fund/macro_view.py`, `fund/app/domain/fund/risk_verdict.py`, `fund/app/ports/llm.py`, `fund/app/ports/market_data.py`, `fund/app/agents/prompts/`, `fund/app/tools/registry.py`, `fund/app/tools/run.py`, `fund/app/services/optimization/assess_risk.py`, `fund/app/agents/orchestrator.py`, `optimizer/attribution/`, `optimizer/optimization/` |
| `fund/app/domain/_shared/` | CORE ·  pianificato | Primitive di dominio condivise fra domain/fund/ e domain/trading/ (value object, tipi comuni, eccezioni di dominio). | — |
| `fund/app/domain/fund/__init__.py` | CORE ·  pianificato | Superficie pubblica del pacchetto domain/fund/ — il barrel module che raggruppa in un unico spazio dei nomi importabile i cinque tipi che compongono il contratto di stato del fondo: FundState, MacroView, Allocation, RiskVerdict, Approval. | `fund/app/domain/fund/fund_state.py`, `fund/app/domain/fund/macro_view.py`, `fund/app/domain/fund/allocation.py`, `fund/app/domain/fund/risk_verdict.py`, `fund/app/domain/fund/approval.py` |
| `fund/app/domain/fund/allocation.py` | CORE ·  pianificato | Allocation, il tipo che porta i pesi proposti e la loro derivazione — l'output tipizzato dell'agente di allocazione (agents/allocation.py), fondato sui capitoli 2, 3, 17, 30 della teoria. | `fund/app/domain/fund/fund_state.py` |
| `fund/app/domain/fund/approval.py` | CORE ·  pianificato | Approval, il tipo che registra ciò che una persona ha esplicitamente autorizzato al gate di approvazione umana — il terzo passo, prima dell'eventuale adapter broker, della pipeline deterministica che ogni trade proposto attraversa. | `fund/app/domain/fund/fund_state.py`, `fund/app/domain/trading/order.py` |
| `fund/app/domain/fund/fund_state.py` | CORE ·  pianificato | FundState, l'oggetto blackboard condiviso: il contratto di stato tipizzato, congelato e serializzabile che i quattro agenti leggono e scrivono e che l'orchestratore checkpointa a ogni transizione. | `fund/app/domain/fund/macro_view.py`, `fund/app/domain/fund/allocation.py`, `fund/app/domain/fund/risk_verdict.py`, `fund/app/domain/fund/approval.py`, `fund/app/domain/trading/order.py` |
| `fund/app/domain/fund/macro_view.py` | CORE ·  pianificato | MacroView, il tipo che porta la lettura di regime e le view formate da essa — l'output tipizzato dell'agente macro (agents/macro.py), fondato sui capitoli 21, 26, 28, 29 della teoria. | `fund/app/domain/fund/fund_state.py` |
| `fund/app/domain/fund/risk_verdict.py` | CORE ·  pianificato | RiskVerdict, il tipo che porta la valutazione indipendente dell'allocazione proposta — l'output tipizzato dell'agente di rischio (agents/risk.py), fondato sui capitoli 4, 20, 23, 24 della teoria. | `fund/app/domain/fund/fund_state.py`, `fund/app/domain/fund/allocation.py` |
| `fund/app/domain/portfolio/__init__.py` | CORE ·  pianificato | Punto di ingresso del pacchetto di dominio «portfolio»: raccoglie ed espone Holding e Portfolio come superficie pubblica del pacchetto. | `fund/app/domain/portfolio/holding.py`, `fund/app/domain/portfolio/portfolio.py` |
| `fund/app/domain/portfolio/holding.py` | CORE ·  pianificato | Il modello di dominio Holding: il portatore tipizzato di una singola posizione posseduta dal fondo — lo strumento e la quantità detenuta, così come il dominio stesso li possiede, non come li riporterebbe un broker. | `fund/app/domain/portfolio/portfolio.py`, `fund/app/domain/trading/order.py` |
| `fund/app/domain/portfolio/portfolio.py` | CORE ·  pianificato | Il modello di dominio Portfolio: l'aggregato delle Holding possedute dal fondo in questo momento — lo stato dei possedimenti di cui il dominio stesso è proprietario, distinto da ciò che un broker riporterebbe. | `fund/app/domain/portfolio/holding.py`, `fund/app/domain/trading/order.py`, `fund/app/services/optimization/propose_orders.py`, `fund/app/agents/execution.py` |
| `fund/app/domain/trading/__init__.py` | CORE ·  pianificato | Punto di ingresso del pacchetto di dominio «trading»: raccoglie ed espone Order, GuardrailRule e KillSwitch come superficie pubblica del pacchetto. | `fund/app/domain/trading/order.py`, `fund/app/domain/trading/guardrail_rule.py`, `fund/app/domain/trading/kill_switch.py` |
| `fund/app/domain/trading/guardrail_rule.py` | CORE ·  pianificato | Il modello di dominio GuardrailRule: una definizione dichiarativa di limite (posizione, esposizione, concentrazione) espressa su contributi di rischio, con i validator Pydantic v2 che ne fanno il motore di regole classico dello step 2 del gate. | `fund/app/domain/trading/order.py`, `fund/app/domain/trading/kill_switch.py`, `fund/app/domain/fund/risk_verdict.py`, `fund/app/agents/guardrail.py`, `fund/app/services/fund/approve_trade.py` |
| `fund/app/domain/trading/kill_switch.py` | CORE ·  pianificato | Il modello di dominio KillSwitch: uno stato di arresto deterministico e incondizionato (scattato/non scattato più la causa) verificato senza alcuna chiamata LLM prima di ogni passo dell'agente, che prevale sempre sull'instradamento probabilistico. | `fund/app/domain/trading/guardrail_rule.py`, `fund/app/domain/fund/fund_state.py`, `fund/app/agents/guardrail.py`, `fund/app/services/fund/run_deliberation.py` |
| `fund/app/domain/trading/order.py` | CORE ·  pianificato | Il modello di dominio Order: il portatore tipizzato di un singolo ordine (o child order) proposto dall'agente di esecuzione, dal suo stato «proposto» fino a «piazzato/consigliato», compreso il posto per la chiave di idempotenza che un broker richiederà. | `fund/app/domain/fund/fund_state.py`, `fund/app/domain/portfolio/holding.py`, `fund/app/agents/execution.py`, `fund/app/agents/guardrail.py`, `fund/app/services/optimization/propose_orders.py`, `fund/app/services/fund/approve_trade.py`, `fund/app/ports/broker.py` |
| `fund/app/http/app.py` | DRIVING ·  pianificato | Application factory FastAPI del fondo. | `fund/app/http/routers/`, `fund/app/http/deps.py` |
| `fund/app/http/deps.py` | DRIVING ·  pianificato | Composition root: cablaggio a iniezione di dipendenze fra adapters/ e services/, esposto ai routers. | `fund/app/services/optimization/`, `fund/app/services/fund/`, `fund/app/adapters/persistence/`, `fund/app/adapters/broker/`, `fund/app/adapters/llm/`, `fund/app/adapters/checkpointer/` |
| `fund/app/http/routers/` | DRIVING ·  pianificato | Router HTTP: optimize, fund, trades, stream. | `fund/app/services/optimization/`, `fund/app/services/fund/`, `fund/app/http/sse.py`, `fund/app/http/deps.py` |
| `fund/app/http/sse.py` | DRIVING ·  pianificato | Streaming SSE nativo FastAPI per gli eventi di deliberazione per-nodo e il progresso di portopt. | `fund/app/services/fund/`, `fund/app/services/optimization/` |
| `fund/app/main.py` | config ·  pianificato | Entrypoint uvicorn del servizio fund. | `fund/app/http/app.py`, `external:uvicorn` |
| `fund/app/ports/agent.py` | PORTS ·  pianificato | L'interfaccia unica che ogni agente del fondo (macro, allocazione, rischio, esecuzione) deve implementare: il contratto che rende i quattro agenti intercambiabili agli occhi dell'orchestratore. | `fund/app/domain/fund/fund_state.py`, `fund/app/agents/macro.py`, `fund/app/agents/allocation.py`, `fund/app/agents/risk.py`, `fund/app/agents/execution.py`, `fund/app/agents/orchestrator.py` |
| `fund/app/ports/broker.py` | PORTS ·  pianificato | L'interfaccia opzionale attraverso cui il core piazza, cancella e sincronizza ordini presso un broker reale — l'unica porta del fondo per cui, per decisione esplicita di §6, nessun adapter è registrato di default. | `fund/app/domain/trading/order.py`, `fund/app/domain/portfolio/portfolio.py`, `fund/app/domain/portfolio/holding.py`, `fund/app/services/fund/approve_trade.py`, `fund/app/agents/execution.py`, `fund/app/adapters/broker/trading212/` |
| `fund/app/ports/checkpointer.py` | PORTS ·  pianificato | Il confine che disaccoppia l'orchestratore hand-rolled dal meccanismo concreto di persistenza dello stato del grafo: dichiara come FundState viene salvato ad ogni transizione e come una deliberazione sospesa in attesa di approvazione umana viene ripresa, senza che il core sappia che dietro c'è LangGraph/PostgresSaver su Postgres. | `fund/app/domain/fund/fund_state.py`, `fund/app/agents/orchestrator.py`, `fund/app/services/fund/run_deliberation.py`, `fund/app/services/fund/approve_trade.py`, `fund/app/adapters/checkpointer/` |
| `fund/app/ports/llm.py` | PORTS ·  pianificato | Il confine tipizzato di structured-output tra il core del fondo e i modelli linguistici locali: dichiara come un agente richiede un'inferenza e riceve indietro un oggetto tipizzato e validato (BAML-backed), senza che il core importi mai Ollama o il client BAML generato. | `fund/app/domain/fund/fund_state.py`, `fund/app/agents/macro.py`, `fund/app/agents/allocation.py`, `fund/app/agents/risk.py`, `fund/app/agents/execution.py`, `fund/app/agents/prompts/`, `fund/app/tools/intent.py`, `fund/app/adapters/llm/` |
| `fund/app/ports/market_data.py` | PORTS ·  pianificato | L'interfaccia astratta di sola lettura verso i dati di mercato, fondamentali e macroeconomici: la porta attraverso cui il core del fondo — servizi e agenti — legge prezzi, fondamentali e serie macro senza mai conoscere Postgres, lo schema concreto o il fatto che quei dati siano stati raccolti dal demone di ingestion. | `fund/app/services/optimization/estimate_regime.py`, `fund/app/services/optimization/run_optimization.py`, `fund/app/services/optimization/assess_risk.py`, `fund/app/agents/macro.py`, `fund/app/agents/risk.py`, `fund/app/agents/allocation.py`, `fund/app/agents/execution.py`, `fund/app/tools/registry.py`, `fund/app/tools/run.py`, `fund/app/domain/portfolio/portfolio.py`, `fund/app/adapters/persistence/market_data_repository.py` |
| `fund/app/services/_shared/__init__.py` | CORE ·  pianificato | Aggregatore del pacchetto _shared dei servizi fund: il punto unico da cui services/fund/ (run_deliberation, approve_trade) e services/optimization/ (run_optimization, assess_risk, estimate_regime, propose_orders) importano gli helper cross-cutting. | `fund/app/services/_shared/_json_safe.py`, `fund/app/services/_shared/_progress.py`, `fund/app/services/_shared/_solver_guard.py`, `fund/app/services/_shared/_idempotency.py`, `fund/app/services/_shared/audit_log.py`, `fund/app/services/_shared/broker_gate.py` |
| `fund/app/services/_shared/_idempotency.py` | CORE ·  pianificato | Derivazione della chiave di idempotenza per un intent d'ordine e per ciascun ordine figlio della traiettoria di esecuzione, tenuta nel core sopra la porta broker come richiesto da §6/§9. | `fund/app/domain/trading/order.py`, `fund/app/services/optimization/propose_orders.py`, `fund/app/services/fund/approve_trade.py`, `fund/app/ports/broker.py` |
| `fund/app/services/_shared/_json_safe.py` | CORE ·  pianificato | Utility pura di sanificazione numerica per i campi di FundState prima che attraversino il confine di serializzazione verso checkpoint, audit trail o evento SSE. | — |
| `fund/app/services/_shared/_progress.py` | CORE ·  pianificato | Costruttori di eventi di progresso condivisi tra la deliberazione a quattro agenti e le chiamate a portopt, nella forma tipizzata per-nodo che il trasporto SSE richiede. | `fund/app/services/fund/run_deliberation.py`, `fund/app/services/optimization/run_optimization.py`, `fund/app/domain/fund/fund_state.py` |
| `fund/app/services/_shared/_solver_guard.py` | CORE ·  pianificato | Guardia condivisa che avvolge ogni chiamata a portopt raggiungibile da un passo agente con un budget di wall-clock e normalizza lo stato terminale del solver in un verdetto tipizzato pass/fail/infeasible/timeout. | `fund/app/services/optimization/run_optimization.py`, `fund/app/services/optimization/assess_risk.py`, `fund/app/services/optimization/propose_orders.py`, `fund/app/services/optimization/solver_result.py` |
| `fund/app/services/_shared/audit_log.py` | CORE ·  pianificato | Registrazione di audit per ogni decisione presa da un agente LLM: modello, parametri e hash del prompt insieme all'output completo, prima che il passo venga checkpointato. | `fund/app/domain/fund/fund_state.py`, `fund/app/services/fund/run_deliberation.py`, `fund/app/ports/checkpointer.py`, `fund/app/ports/llm.py` |
| `fund/app/services/_shared/broker_gate.py` | CORE ·  pianificato | Verifica se un adapter broker è registrato dietro ports/broker.py e comunica alla use-case di approvazione se fermarsi al gate umano o proseguire verso il collocamento reale. | `fund/app/ports/broker.py`, `fund/app/services/fund/approve_trade.py` |
| `fund/app/services/fund/__init__.py` | CORE ·  pianificato | Punto d'ingresso del sotto-pacchetto `fund` del service/use-case layer: raccoglie i due use-case che governano una deliberazione del fondo. | `fund/app/services/fund/run_deliberation.py`, `fund/app/services/fund/approve_trade.py` |
| `fund/app/services/fund/approve_trade.py` | CORE ·  pianificato | Use-case che realizza il gate umano di approvazione: trasforma la decisione di una persona su uno o più ordini proposti in un'azione sul `FundState` sospeso. | `fund/app/domain/fund/approval.py`, `fund/app/domain/trading/order.py`, `fund/app/agents/orchestrator.py`, `fund/app/ports/checkpointer.py`, `fund/app/ports/broker.py` |
| `fund/app/services/fund/run_deliberation.py` | CORE ·  pianificato | Use-case che avvia e conduce un intero ciclo di deliberazione del fondo: dalle posizioni correnti alla lista di ordini proposti, passando per i quattro agenti. | `fund/app/agents/orchestrator.py`, `fund/app/domain/fund/fund_state.py`, `fund/app/domain/portfolio/portfolio.py`, `fund/app/ports/checkpointer.py` |
| `fund/app/services/optimization/__init__.py` | CORE ·  pianificato | Punto d'ingresso del pacchetto che documenta i quattro use-case di ottimizzazione della fund (stima del regime, ottimizzazione, valutazione del rischio, proposta ordini) come la "small fixed tool registry" di §6.3, condivisa da agenti e router HTTP. | `fund/app/services/optimization/estimate_regime.py`, `fund/app/services/optimization/run_optimization.py`, `fund/app/services/optimization/assess_risk.py`, `fund/app/services/optimization/propose_orders.py`, `fund/app/services/optimization/solver_result.py` |
| `fund/app/services/optimization/assess_risk.py` | CORE ·  pianificato | Use-case che calcola, per un'allocazione proposta, le misure di rischio che l'ottimizzatore non ha ottimizzato — misura di coda coerente, rischio di percorso, attribuzione di Euler, diagnostica di concentrazione e stress test — sempre eseguito in codice, mai a discrezione del modello linguistico dell'agente di rischio. | `fund/app/ports/market_data.py`, `optimizer/attribution/_euler.py`, `optimizer/attribution/_factor_impact.py`, `optimizer/attribution/_expected_loss.py`, `optimizer/uncertainty_set/_factory.py`, `fund/app/services/optimization/solver_result.py` |
| `fund/app/services/optimization/estimate_regime.py` | CORE ·  pianificato | Use-case condiviso che stima lo stato di mercato latente e gli indicatori osservabili di stress — l'unico strumento che sia l'agente macro sia l'agente di allocazione invocano, evitando che l'uno importi l'altro. | `fund/app/ports/market_data.py`, `optimizer/regimes/_markov_switching.py`, `optimizer/regimes/_turbulence.py`, `optimizer/regimes/_absorption.py`, `optimizer/regimes/_nowcast.py`, `optimizer/regimes/_cross_sectional.py` |
| `fund/app/services/optimization/propose_orders.py` | CORE ·  pianificato | Use-case che trasforma un'allocazione target in una lista di ordini candidati, calcolando il portafoglio-obiettivo (aim), la frazione di distanza da percorrere e la traiettoria di esecuzione — mai la semplice differenza fra allocazione e posizioni correnti. | `fund/app/domain/portfolio/portfolio.py`, `fund/app/services/optimization/run_optimization.py`, `optimizer/execution/__init__.py`, `optimizer/execution/_trajectory.py`, `optimizer/execution/_cost.py`, `fund/app/domain/trading/order.py` |
| `fund/app/services/optimization/run_optimization.py` | CORE ·  pianificato | Use-case che invoca `portopt` per calcolare i pesi ottimi secondo la costruzione scelta dall'agente di allocazione — mean-variance vincolata, posteriore Black-Litterman, risk budgeting/ERC o policy dirette — la stessa chiamata che il tool registry di §6.3 espone anche ai router HTTP. | `optimizer/optimization/_factory.py`, `optimizer/optimization/_risk_budgeting.py`, `optimizer/optimization/_dr_cvar.py`, `optimizer/optimization/_robust.py`, `optimizer/views/_uncertainty.py`, `optimizer/moments/_factory.py`, `fund/app/services/optimization/solver_result.py` |
| `fund/app/services/optimization/solver_result.py` | CORE ·  pianificato | Tipo di valore che porta l'esito di ogni chiamata a un solutore convesso — stato terminale, pesi (se presenti), diagnostica — condiviso da run_optimization.py, assess_risk.py e, tramite quest'ultimo, da propose_orders.py. | — |
| `fund/app/tools/intent.py` | CORE ·  pianificato | Definisce ToolIntent, il tipo tipizzato che un agente riceve dall'LLM quando deve scegliere fra i tool del registro fisso (§6.3): un campo enum `tool` — che discrimina sui valori dichiarati in `registry.py` — più un blocco di argomenti tipizzati specifico per la variante scelta. È il contratto dati che attraversa il confine di structured-output BAML, non l'implementazione di quel confine. | `fund/app/tools/registry.py`, `fund/app/tools/run.py`, `fund/app/ports/llm.py`, `fund/app/agents/allocation.py` |
| `fund/app/tools/registry.py` | CORE ·  pianificato | Enum/registro dei tool disponibili agli agenti, hand-registered a build time. | `fund/app/services/optimization/`, `fund/app/services/fund/`, `fund/app/ports/market_data.py` |
| `fund/app/tools/run.py` | CORE ·  pianificato | Il dispatcher del registro fisso di tool: riceve il ToolIntent tipizzato (l'oggetto BAML enum-tool + argomenti tipizzati prodotto da tools/intent.py), lo valida, lo instrada verso il caso d'uso del service layer che tools/registry.py associa a quel nome di tool, ed è il punto in cui vive interamente il ciclo di fallimento validate → dispatch → retry → fail-safe del §6.3 punto 4. | `fund/app/tools/registry.py`, `fund/app/tools/intent.py`, `fund/app/agents/macro.py`, `fund/app/agents/allocation.py`, `fund/app/agents/risk.py`, `fund/app/agents/execution.py`, `fund/app/services/optimization/run_optimization.py`, `fund/app/services/fund/run_deliberation.py`, `fund/app/services/fund/approve_trade.py` |
| `fund/baml_client/` | DRIVEN ·  pianificato | Client Python generato da BAML, da non modificare a mano. | `external:ollama` |
| `fund/baml_src/` | config ·  pianificato | Sorgenti BAML (definizioni di funzione e schema tipizzati) per l'inferenza degli agenti del fondo. | `fund/baml_client/` |
| `fund/tests/` | config ·  pianificato | Suite di test del bounded context fund. | `fund/app/domain/fund/`, `fund/app/ports/agent.py`, `fund/app/agents/orchestrator.py`, `fund/app/services/fund/`, `fund/app/services/optimization/` |

<!-- FINE GENERATO: fund -->

---

## 7. `frontend/` — il client Angular

Un client **Angular** dell'API HTTP di `fund`. Il nucleo è cieco alla sua esistenza: il
frontend è un adapter driving come lo è la CLI, e nessuna decisione di dominio dipende da come
viene disegnata.

Il layout segue la convenzione che gli altri frontend Angular 21 del workspace già
condividono — `pages/`, `services/`, `models/`, `interceptors/`, `shared/`, con
`app.config.ts` e `app.routes.ts` alla radice di `src/app/` — così chi si sposta fra i
progetti ritrova la stessa forma. Componenti standalone dappertutto; nessun NgModule.

### 7.1 Le 25 pagine e le 8 sezioni di navigazione

Le pagine non sono inventate qui: sono specificate una per una in `docs/frontend/`, e ogni
documento porta il proprio wireframe, la tabella dei controlli, gli stati (vuoto, caricamento,
errore), le note di accessibilità e le fonti di dominio che giustificano ciò che la pagina
mostra. Le 25 pagine sono raggruppate in 8 sezioni di navigazione; ogni sezione è una cartella
sotto `pages/` e ogni rotta è caricata in lazy.

**Il guscio è uno solo ed è condiviso.** Topbar, sidebar a 8 sezioni, breadcrumb e selettore di
ambito del portafoglio vivono in `shared/shell/` e ospitano un solo `<router-outlet>`. Sono
identici su ogni pagina — è la pagina 01 a specificarli — e nessuna pagina ne ridisegna un
pezzo per conto proprio. Questa è la regola che ha fatto nascere la pipeline `scriptorium ui`:
prima che esistesse, lo stesso componente compariva sotto nomi diversi in pagine diverse (197
nomi di regione per quello che erano 32 regioni vere, 12 nomi per una sola toolbar).

**Il lessico delle regioni è chiuso.** Un nome di regione che non è nel lessico è un errore di
specifica, non una variante. La stessa disciplina vale per i controlli: ogni controllo porta
sia l'etichetta che appare a schermo sia il componente condiviso che lo realizza, e sono due
campi distinti proprio perché confonderli è l'errore che genera duplicazione — due pagine con
etichette diverse per lo stesso componente sembrano due componenti.

**I wireframe hanno una specifica di stile unica**
(`docs/frontend/template/01 Wireframe — specifica di stile.md`): un alfabeto chiuso di glifi e
una larghezza unica di 86 colonne, derivata dal renderer del guscio e non scelta a occhio.
Nessun limite di altezza: una pagina densa ha diritto a un wireframe lungo, e imporre un tetto
produceva wireframe amputati invece che pagine più semplici.

### 7.2 Convenzioni Angular

Sono quelle del workspace, e valgono qui senza eccezioni:

| Convenzione | Regola |
|---|---|
| Componenti | standalone (è il default: **non** scrivere `standalone: true`) |
| Change detection | `ChangeDetectionStrategy.OnPush` su ogni componente |
| Input / output | funzioni `input()` e `output()`, non decoratori |
| Dependency injection | `inject()`, non iniezione da costruttore |
| Control flow | `@if`, `@for`, `@switch` — non `*ngIf`, `*ngFor` |
| Stato | signal per lo stato locale, `computed()` per lo stato derivato |
| Binding di classe e stile | `class` e `style`, non `ngClass` e `ngStyle` |
| CSS | Tailwind v4 |

**I modelli TypeScript rispecchiano i tipi di dominio.** `models/` contiene i corrispettivi TS
di `FundState`, `Allocation`, `RiskVerdict` e `Order`. Sono uno specchio, non una seconda
sorgente di verità: quando il contratto di stato di §6.1 cambia, cambiano questi — e non il
contrario.

**Lo streaming è SSE.** La deliberazione dei quattro agenti è lunga e a passi, quindi il
frontend la segue via Server-Sent Events sull'endpoint di stream (§9), non con polling. Il
servizio che la consuma espone lo stato come signal, così la pagina si aggiorna senza logica di
sottoscrizione sparsa nei componenti.

### 7.3 Cosa il frontend non fa

Non calcola. Non ci sono numeri di rischio, pesi o probabilità di regime prodotti nel browser:
ogni cifra che appare a schermo arriva dall'API, che a sua volta la prende da un solver o da uno
stimatore. Il frontend può formattare e aggregare per la visualizzazione, mai derivare una
grandezza che entra in una decisione.

Non decide. Il gate di approvazione umana di §6 è un `interrupt()` dell'orchestratore: il
frontend ne è l'interfaccia — mostra cosa si sta approvando e raccoglie la firma — ma
l'autorizzazione vive in `approvals` dentro `FundState`, non nello stato del componente.

<!-- GENERATO: frontend -->

_Da qui in giù la sezione è generata da `scriptorium arch` il 2026-08-19: non modificarla a mano, si riscrive a ogni assemblaggio. Il record completo — responsabilità estese e citazioni riga per riga — sta in `data/architecture-spec/`, da cui questo è renderizzato._

### 7.1 Albero

164 file, ciascuno annotato col suo ruolo esagonale. Ruoli: config 1 · DRIVING 163

```
└── frontend/
    ├── angular.json  ┃ config · planned
    └── src/
        ├── app/
        │   ├── app.config.ts  ┃ DRIVING · planned
        │   ├── app.routes.ts  ┃ DRIVING · planned
        │   ├── interceptors/
        │   │   ├── correlation-id.interceptor.ts  ┃ DRIVING · planned
        │   │   └── error.interceptor.ts  ┃ DRIVING · planned
        │   ├── models/
        │   │   ├── allocation.model.ts  ┃ DRIVING · planned
        │   │   ├── approval.model.ts  ┃ DRIVING · planned
        │   │   ├── audit-decision.model.ts  ┃ DRIVING · planned
        │   │   ├── backtest-validation.model.ts  ┃ DRIVING · planned
        │   │   ├── fund-state.model.ts  ┃ DRIVING · planned
        │   │   ├── guardrail.model.ts  ┃ DRIVING · planned
        │   │   ├── macro-view.model.ts  ┃ DRIVING · planned
        │   │   ├── optimization-run.model.ts  ┃ DRIVING · planned
        │   │   ├── order.model.ts  ┃ DRIVING · planned
        │   │   ├── portfolio.model.ts  ┃ DRIVING · planned
        │   │   ├── risk-attribution.model.ts  ┃ DRIVING · planned
        │   │   ├── risk-monitoring.model.ts  ┃ DRIVING · planned
        │   │   ├── risk-verdict.model.ts  ┃ DRIVING · planned
        │   │   ├── stress-testing.model.ts  ┃ DRIVING · planned
        │   │   └── turbulence.model.ts  ┃ DRIVING · planned
        │   ├── pages/
        │   │   ├── advanced-signals/
        │   │   │   ├── advanced-signals.routes.ts  ┃ DRIVING · planned
        │   │   │   ├── alternative-data-sentiment/
        │   │   │   │   ├── alternative-data-sentiment.css  ┃ DRIVING · planned
        │   │   │   │   ├── alternative-data-sentiment.html  ┃ DRIVING · planned
        │   │   │   │   └── alternative-data-sentiment.ts  ┃ DRIVING · planned
        │   │   │   ├── factor-timing-rotation/
        │   │   │   │   ├── factor-timing-rotation.css  ┃ DRIVING · planned
        │   │   │   │   ├── factor-timing-rotation.html  ┃ DRIVING · planned
        │   │   │   │   └── factor-timing-rotation.ts  ┃ DRIVING · planned
        │   │   │   └── market-regimes/
        │   │   │       ├── market-regimes.css  ┃ DRIVING · planned
        │   │   │       ├── market-regimes.html  ┃ DRIVING · planned
        │   │   │       └── market-regimes.ts  ┃ DRIVING · planned
        │   │   ├── approvals/
        │   │   │   ├── approval-gate/
        │   │   │   │   ├── approval-gate.css  ┃ DRIVING · planned
        │   │   │   │   ├── approval-gate.html  ┃ DRIVING · planned
        │   │   │   │   └── approval-gate.ts  ┃ DRIVING · planned
        │   │   │   ├── approvals.routes.ts  ┃ DRIVING · planned
        │   │   │   └── guardrail-killswitch/
        │   │   │       ├── guardrail-killswitch.css  ┃ DRIVING · planned
        │   │   │       ├── guardrail-killswitch.html  ┃ DRIVING · planned
        │   │   │       └── guardrail-killswitch.ts  ┃ DRIVING · planned
        │   │   ├── build/
        │   │   │   ├── build-step-tracker.service.ts  ┃ DRIVING · planned
        │   │   │   ├── build.routes.ts  ┃ DRIVING · planned
        │   │   │   ├── objective-constraints/
        │   │   │   │   ├── objective-constraints.page.css  ┃ DRIVING · planned
        │   │   │   │   ├── objective-constraints.page.html  ┃ DRIVING · planned
        │   │   │   │   └── objective-constraints.page.ts  ┃ DRIVING · planned
        │   │   │   ├── run-solver-diagnostics/
        │   │   │   │   ├── run-solver-diagnostics.page.css  ┃ DRIVING · planned
        │   │   │   │   ├── run-solver-diagnostics.page.html  ┃ DRIVING · planned
        │   │   │   │   └── run-solver-diagnostics.page.ts  ┃ DRIVING · planned
        │   │   │   ├── signals-factors/
        │   │   │   │   ├── signals-factors.page.css  ┃ DRIVING · planned
        │   │   │   │   ├── signals-factors.page.html  ┃ DRIVING · planned
        │   │   │   │   └── signals-factors.page.ts  ┃ DRIVING · planned
        │   │   │   ├── universe-data/
        │   │   │   │   ├── universe-data.page.css  ┃ DRIVING · planned
        │   │   │   │   ├── universe-data.page.html  ┃ DRIVING · planned
        │   │   │   │   └── universe-data.page.ts  ┃ DRIVING · planned
        │   │   │   └── views-builder/
        │   │   │       ├── views-builder.page.css  ┃ DRIVING · planned
        │   │   │       ├── views-builder.page.html  ┃ DRIVING · planned
        │   │   │       └── views-builder.page.ts  ┃ DRIVING · planned
        │   │   ├── dashboard/
        │   │   │   ├── dashboard.css  ┃ DRIVING · planned
        │   │   │   ├── dashboard.html  ┃ DRIVING · planned
        │   │   │   └── dashboard.ts  ┃ DRIVING · planned
        │   │   ├── fund/
        │   │   │   ├── allocation-agent/
        │   │   │   │   ├── allocation-agent.css  ┃ DRIVING · planned
        │   │   │   │   ├── allocation-agent.html  ┃ DRIVING · planned
        │   │   │   │   └── allocation-agent.ts  ┃ DRIVING · planned
        │   │   │   ├── deliberation/
        │   │   │   │   ├── deliberation.css  ┃ DRIVING · planned
        │   │   │   │   ├── deliberation.html  ┃ DRIVING · planned
        │   │   │   │   └── deliberation.ts  ┃ DRIVING · planned
        │   │   │   ├── execution-agent/
        │   │   │   │   ├── execution-agent.css  ┃ DRIVING · planned
        │   │   │   │   ├── execution-agent.html  ┃ DRIVING · planned
        │   │   │   │   └── execution-agent.ts  ┃ DRIVING · planned
        │   │   │   ├── fund.routes.ts  ┃ DRIVING · planned
        │   │   │   ├── macro-agent/
        │   │   │   │   ├── macro-agent.css  ┃ DRIVING · planned
        │   │   │   │   ├── macro-agent.html  ┃ DRIVING · planned
        │   │   │   │   └── macro-agent.ts  ┃ DRIVING · planned
        │   │   │   └── risk-agent/
        │   │   │       ├── risk-agent.css  ┃ DRIVING · planned
        │   │   │       ├── risk-agent.html  ┃ DRIVING · planned
        │   │   │       └── risk-agent.ts  ┃ DRIVING · planned
        │   │   ├── report/
        │   │   │   ├── report.css  ┃ DRIVING · planned
        │   │   │   ├── report.html  ┃ DRIVING · planned
        │   │   │   └── report.ts  ┃ DRIVING · planned
        │   │   ├── results/
        │   │   │   ├── backtest-validation/
        │   │   │   │   ├── backtest-validation.css  ┃ DRIVING · planned
        │   │   │   │   ├── backtest-validation.html  ┃ DRIVING · planned
        │   │   │   │   └── backtest-validation.ts  ┃ DRIVING · planned
        │   │   │   ├── results-frontier/
        │   │   │   │   ├── results-frontier.css  ┃ DRIVING · planned
        │   │   │   │   ├── results-frontier.html  ┃ DRIVING · planned
        │   │   │   │   └── results-frontier.ts  ┃ DRIVING · planned
        │   │   │   ├── results.routes.ts  ┃ DRIVING · planned
        │   │   │   └── review-rebalancing/
        │   │   │       ├── review-rebalancing.css  ┃ DRIVING · planned
        │   │   │       ├── review-rebalancing.html  ┃ DRIVING · planned
        │   │   │       └── review-rebalancing.ts  ┃ DRIVING · planned
        │   │   └── risk/
        │   │       ├── risk-attribution/
        │   │       │   ├── risk-attribution.css  ┃ DRIVING · planned
        │   │       │   ├── risk-attribution.html  ┃ DRIVING · planned
        │   │       │   └── risk-attribution.ts  ┃ DRIVING · planned
        │   │       ├── risk-monitoring/
        │   │       │   ├── risk-monitoring.css  ┃ DRIVING · planned
        │   │       │   ├── risk-monitoring.html  ┃ DRIVING · planned
        │   │       │   └── risk-monitoring.ts  ┃ DRIVING · planned
        │   │       ├── risk.routes.ts  ┃ DRIVING · planned
        │   │       ├── stress-testing/
        │   │       │   ├── stress-testing.css  ┃ DRIVING · planned
        │   │       │   ├── stress-testing.html  ┃ DRIVING · planned
        │   │       │   └── stress-testing.ts  ┃ DRIVING · planned
        │   │       └── turbulence-systemic/
        │   │           ├── turbulence-systemic.css  ┃ DRIVING · planned
        │   │           ├── turbulence-systemic.html  ┃ DRIVING · planned
        │   │           └── turbulence-systemic.ts  ┃ DRIVING · planned
        │   ├── services/
        │   │   ├── api-client.ts  ┃ DRIVING · planned
        │   │   ├── fund.service.ts  ┃ DRIVING · planned
        │   │   ├── optimize.service.ts  ┃ DRIVING · planned
        │   │   ├── stream.service.ts  ┃ DRIVING · planned
        │   │   └── trades.service.ts  ┃ DRIVING · planned
        │   └── shared/
        │       ├── action-button-row/
        │       │   ├── action-button-row.css  ┃ DRIVING · planned
        │       │   ├── action-button-row.html  ┃ DRIVING · planned
        │       │   └── action-button-row.ts  ┃ DRIVING · planned
        │       ├── confirm-dialog/
        │       │   ├── confirm-dialog.css  ┃ DRIVING · planned
        │       │   ├── confirm-dialog.html  ┃ DRIVING · planned
        │       │   └── confirm-dialog.ts  ┃ DRIVING · planned
        │       ├── cross-page-link/
        │       │   ├── cross-page-link.css  ┃ DRIVING · planned
        │       │   ├── cross-page-link.html  ┃ DRIVING · planned
        │       │   └── cross-page-link.ts  ┃ DRIVING · planned
        │       ├── directives/
        │       │   ├── aria-live-announcer.directive.ts  ┃ DRIVING · planned
        │       │   └── focus-trap.directive.ts  ┃ DRIVING · planned
        │       ├── empty-state/
        │       │   ├── empty-state.css  ┃ DRIVING · planned
        │       │   ├── empty-state.html  ┃ DRIVING · planned
        │       │   └── empty-state.ts  ┃ DRIVING · planned
        │       ├── entity-card/
        │       │   ├── entity-card.css  ┃ DRIVING · planned
        │       │   ├── entity-card.html  ┃ DRIVING · planned
        │       │   └── entity-card.ts  ┃ DRIVING · planned
        │       ├── error-state/
        │       │   ├── error-state.css  ┃ DRIVING · planned
        │       │   ├── error-state.html  ┃ DRIVING · planned
        │       │   └── error-state.ts  ┃ DRIVING · planned
        │       ├── hero-stat-card/
        │       │   ├── hero-stat-card.css  ┃ DRIVING · planned
        │       │   ├── hero-stat-card.html  ┃ DRIVING · planned
        │       │   └── hero-stat-card.ts  ┃ DRIVING · planned
        │       ├── info-card/
        │       │   ├── info-card.css  ┃ DRIVING · planned
        │       │   ├── info-card.html  ┃ DRIVING · planned
        │       │   └── info-card.ts  ┃ DRIVING · planned
        │       ├── pipes/
        │       │   ├── compact-currency.pipe.ts  ┃ DRIVING · planned
        │       │   └── weight-percent.pipe.ts  ┃ DRIVING · planned
        │       ├── refresh-control/
        │       │   ├── refresh-control.css  ┃ DRIVING · planned
        │       │   ├── refresh-control.html  ┃ DRIVING · planned
        │       │   └── refresh-control.ts  ┃ DRIVING · planned
        │       ├── retry-button/
        │       │   ├── retry-button.css  ┃ DRIVING · planned
        │       │   ├── retry-button.html  ┃ DRIVING · planned
        │       │   └── retry-button.ts  ┃ DRIVING · planned
        │       ├── section-card-grid/
        │       │   ├── section-card-grid.css  ┃ DRIVING · planned
        │       │   ├── section-card-grid.html  ┃ DRIVING · planned
        │       │   └── section-card-grid.ts  ┃ DRIVING · planned
        │       ├── shell/
        │       │   ├── breadcrumb/
        │       │   │   ├── breadcrumb.css  ┃ DRIVING · planned
        │       │   │   ├── breadcrumb.html  ┃ DRIVING · planned
        │       │   │   └── breadcrumb.ts  ┃ DRIVING · planned
        │       │   ├── nav-config.ts  ┃ DRIVING · planned
        │       │   ├── portfolio-scope-switcher/
        │       │   │   ├── portfolio-scope-switcher.css  ┃ DRIVING · planned
        │       │   │   ├── portfolio-scope-switcher.html  ┃ DRIVING · planned
        │       │   │   └── portfolio-scope-switcher.ts  ┃ DRIVING · planned
        │       │   ├── shell.css  ┃ DRIVING · planned
        │       │   ├── shell.html  ┃ DRIVING · planned
        │       │   ├── shell.ts  ┃ DRIVING · planned
        │       │   ├── sidebar/
        │       │   │   ├── sidebar.css  ┃ DRIVING · planned
        │       │   │   ├── sidebar.html  ┃ DRIVING · planned
        │       │   │   └── sidebar.ts  ┃ DRIVING · planned
        │       │   └── topbar/
        │       │       ├── topbar.css  ┃ DRIVING · planned
        │       │       ├── topbar.html  ┃ DRIVING · planned
        │       │       └── topbar.ts  ┃ DRIVING · planned
        │       ├── skeleton-block/
        │       │   ├── skeleton-block.css  ┃ DRIVING · planned
        │       │   ├── skeleton-block.html  ┃ DRIVING · planned
        │       │   └── skeleton-block.ts  ┃ DRIVING · planned
        │       └── status-badge/
        │           ├── status-badge.css  ┃ DRIVING · planned
        │           ├── status-badge.html  ┃ DRIVING · planned
        │           └── status-badge.ts  ┃ DRIVING · planned
        └── main.ts  ┃ DRIVING · planned
```

### 7.2 I file

Tutti i 14 file di frontend/src sono layer=driving: per ARCHITECTURE.md §3 (righe 62-68) e §4 (righe 219-225), frontend/ è l'adattatore driving più esterno del sistema — un client REST+SSE dell'API HTTP del fondo di cui il core (fund/, optimizer/) è cieco ("the core would run unchanged if the frontend were deleted") — e non possiede domain/, ports/ o adapters/ propri. Non esistendo alcun file 'core' in questo bounded context, la regola "un file core non può mai connects_to un file driven direttamente" non è violabile dall'interno: nessun file qui è core, quindi il vincolo è rispettato per assenza di soggetto, non per disciplina interna. È rispettata a livello di piattaforma dal fatto che frontend/ conosce solo il contratto HTTP/SSE esposto da fund/app/http/ (routers optimize·fund·trades·stream, SSE nativa FastAPI con reconnessione Last-Event-ID) — mai l'implementazione interna dei quattro agenti, del guardrail Pydantic o del solver portopt. L'unico punto che attraversa il confine di rete è frontend/src/api/, referenziato come external:fund-http-api (non un vero sistema esterno indipendente, ma il modo corretto di rappresentare l'attraversamento del confine bounded-context verso fund/, che è fuori scope di questa passata). Internamente il grafo rispetta un ordine a strati driving-verso-driving analogo a quello del core: main.* (composition root) monta components/shell/ (guscio persistente — topbar, sidebar a 8 sezioni, breadcrumb, ScopeSwitcher di portafoglio — identico su tutte le 24 pagine) e instrada verso le otto cartelle di features/<sezione>, ciascuna corrispondente 1:1 a un nav_section reale (Dashboard·Costruisci·Risultati·Fondo·Approvazioni·Rischio·Segnali avanzati·Report). Ogni modulo feature compone hooks/ (stato/dati, unico punto che parla con api/), components/ (presentazione pura, mai chiamante diretto di api/) e lib/ (funzioni pure, foglia senza dipendenze in uscita — formattazione, invarianti cross-pagina come la verifica Σxᵢ=1, costanti di design token). Nessuna dipendenza punta mai a fund/ o optimizer/ via import di codice: l'unico canale è HTTP+SSE tramite api/. Due citazioni della bozza precedente in ARCHITECTURE.md (01 Guscio e navigazione.md::144-150 e ::162-172, e ::147) puntavano a righe del file sbagliate rispetto al claim che dovevano supportare (il file ha solo 163 righe; le righe 144-150/147 cadono nella definizione matematica del portafoglio, non nella sincronizzazione topbar↔pannello); qui sono state corrette a 123-128 (interazioni: cambiare portafoglio aggiorna il selettore in topbar) e 138-148 (definizione formale N-vettore/Σxᵢ=1), verificate leggendo il file per intero.

| File | Ruolo | Contiene | Si collega a |
|---|---|---|---|
| `frontend/angular.json` | config ·  pianificato | Il manifesto del workspace Angular CLI del progetto `frontend`: dichiara il progetto e i target `architect` (build, serve, test) con cui la CLI compila, serve in sviluppo e testa il client standalone dell'API HTTP del fondo, secondo la stessa convenzione già condivisa dagli altri frontend Angular 21 del workspace più ampio a cui §4 fa riferimento. È l'unica voce di checklist che copre l'intero gruppo di file di tooling elencato accanto ad esso nell'albero di §4 — `package.json`, `tsconfig*.json`, `Dockerfile` — che non hanno una voce di checklist propria: qui vanno descritte le responsabilità di build/serve/test/budget dell'intero gruppo, non solo quelle strettamente racchiuse nel JSON. | `frontend/src/main.ts`, `frontend/src/app/app.routes.ts`, `frontend/src/environments/`, `external:angular-cli` |
| `frontend/src/app/app.config.ts` | DRIVING ·  pianificato | Il composition root del bootstrap standalone di Angular: l'oggetto `ApplicationConfig` unico da cui `main.ts` avvia l'applicazione, senza alcun `NgModule` a fare da contenitore per i provider. Riunisce in un solo punto tutto ciò che l'adattatore driving 'frontend' deve cablare prima che qualunque pagina venga renderizzata: il router e il client HTTP tipizzato verso l'API del fondo. | `frontend/src/app/app.routes.ts`, `frontend/src/app/interceptors/`, `frontend/src/main.ts` |
| `frontend/src/app/app.routes.ts` | DRIVING ·  pianificato | La tabella di instradamento root dell'applicazione Angular standalone: l'array Routes che il router consulta per risolvere ogni URL in un componente di pagina, con un ramo lazy-loaded per ciascuna delle 8 sezioni di navigazione mostrate nella sidebar del guscio (Dashboard, Build, Results, Fund, Approvals, Risk, Advanced Signals, Report). | `frontend/src/app/pages/dashboard`, `frontend/src/app/pages/build`, `frontend/src/app/pages/results`, `frontend/src/app/pages/fund`, `frontend/src/app/pages/approvals`, `frontend/src/app/pages/risk`, `frontend/src/app/pages/advanced-signals`, `frontend/src/app/pages/report`, `frontend/src/app/app.config.ts` |
| `frontend/src/app/interceptors/correlation-id.interceptor.ts` | DRIVING ·  pianificato | L'interceptor funzionale HTTP che allega un identificativo di correlazione generato lato client a ogni richiesta uscente verso l'API del fondo, il contributo del frontend alla tracciabilità end-to-end che l'architettura richiede su ogni decisione del sistema. | `frontend/src/app/app.config.ts`, `frontend/src/app/services/` |
| `frontend/src/app/interceptors/error.interceptor.ts` | DRIVING ·  pianificato | L'interceptor funzionale HTTP che costituisce la metà lato client del contratto errore→stato dell'adattatore API: intercetta ogni risposta HTTP fallita proveniente dall'API del fondo e la traduce in una forma tipizzata e presentabile, così che pagine e servizi non debbano mai interpretare uno stato HTTP grezzo. | `frontend/src/app/app.config.ts`, `frontend/src/app/services/`, `frontend/src/app/shared/`, `frontend/src/app/models/` |
| `frontend/src/app/models/allocation.model.ts` | DRIVING ·  pianificato | Mirror TypeScript di Allocation, l'oggetto che l'agente allocation scrive in FundState: i pesi proposti e la derivazione registrata di come sono stati ottenuti. | `frontend/src/app/pages/fund/allocation-agent/allocation-agent.ts`, `frontend/src/app/models/fund-state.model.ts`, `frontend/src/app/models/portfolio.model.ts`, `frontend/src/app/pages/fund/risk-agent/risk-agent.ts` |
| `frontend/src/app/models/approval.model.ts` | DRIVING ·  pianificato | Mirror TypeScript di Approval, l'unità che compone il campo approvals di FundState: quello che una persona ha esplicitamente autorizzato al gate di approvazione umana. | `frontend/src/app/pages/approvals/approval-gate/approval-gate.ts`, `frontend/src/app/models/fund-state.model.ts`, `frontend/src/app/models/order.model.ts` |
| `frontend/src/app/models/audit-decision.model.ts` | DRIVING ·  pianificato | Mirror TypeScript di una riga del log delle decisioni degli agenti e dello stato di idempotenza/riconciliazione di un intento d'ordine, i due oggetti che la pagina Report & Audit trail espone come audit-by-inspection. | `frontend/src/app/pages/report/report.ts`, `frontend/src/app/models/order.model.ts`, `frontend/src/app/models/portfolio.model.ts` |
| `frontend/src/app/models/backtest-validation.model.ts` | DRIVING ·  pianificato | Mirror TypeScript dell'esito delle sei verifiche di validazione di una strategia — disclosure delle prove, test fuori campione, correzione per test multipli, haircut Sharpe ratio, deflated Sharpe ratio, PBO/CSCV — più la motivazione economica indipendente, che la pagina Backtest & Validation espone. | `frontend/src/app/pages/results/backtest-validation/backtest-validation.ts` |
| `frontend/src/app/models/fund-state.model.ts` | DRIVING ·  pianificato | Mirror TypeScript del contratto di stato condiviso FundState che l'orchestratore del fondo passa fra i quattro agenti e che l'API espone come oggetto di deliberazione e come record di audit checkpointato. | `frontend/src/app/pages/fund/deliberation/deliberation.ts`, `frontend/src/app/services/`, `frontend/src/app/models/macro-view.model.ts`, `frontend/src/app/models/allocation.model.ts`, `frontend/src/app/models/risk-verdict.model.ts`, `frontend/src/app/models/order.model.ts`, `frontend/src/app/models/approval.model.ts` |
| `frontend/src/app/models/guardrail.model.ts` | DRIVING ·  pianificato | Mirror TypeScript dei tipi del guardrail deterministico — GuardrailRule e KillSwitch di fund/app/domain/trading — che la pagina Guardrail & Kill-Switch legge come confine pass/fail dei passi 1-2 della pipeline di approvazione. | `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.ts`, `frontend/src/app/models/risk-attribution.model.ts`, `frontend/src/app/models/order.model.ts` |
| `frontend/src/app/models/macro-view.model.ts` | DRIVING ·  pianificato | Mirror TypeScript di MacroView, l'oggetto tipizzato che l'agente macro scrive in FundState: la lettura di regime, gli indicatori di stress osservabili, i premi al rischio condizionati e la vista Black-Litterman con la sua incertezza. | `frontend/src/app/pages/fund/macro-agent/macro-agent.ts`, `frontend/src/app/models/fund-state.model.ts`, `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.ts` |
| `frontend/src/app/models/optimization-run.model.ts` | DRIVING ·  pianificato | Mirror TypeScript dell'esito di un run di ottimizzazione portopt — PortfolioResult lato backend — con lo stato del solver come verdetto deterministico pass/fail e i punti di frontiera efficiente che ne derivano. | `frontend/src/app/pages/results/results-frontier/results-frontier.ts`, `frontend/src/app/pages/build/run-solver-diagnostics/run-solver-diagnostics.page.ts`, `frontend/src/app/models/portfolio.model.ts` |
| `frontend/src/app/models/order.model.ts` | DRIVING ·  pianificato | Mirror TypeScript di Order, l'unità che compone proposed_orders in FundState: un ordine proposto dall'agente di esecuzione, con la sua traiettoria e il suo stadio nella pipeline di approvazione a quattro passi. | `frontend/src/app/pages/fund/execution-agent/execution-agent.ts`, `frontend/src/app/pages/approvals/approval-gate/approval-gate.ts`, `frontend/src/app/pages/report/report.ts`, `frontend/src/app/models/fund-state.model.ts`, `frontend/src/app/models/approval.model.ts` |
| `frontend/src/app/models/portfolio.model.ts` | DRIVING ·  pianificato | Mirror TypeScript di Holding e Portfolio — fund/app/domain/portfolio, le posizioni possedute dal dominio del fondo, non lette da un broker — il tipo condiviso da ogni pagina che legge il portafoglio effettivamente detenuto. | `frontend/src/app/pages/results/results-frontier/results-frontier.ts`, `frontend/src/app/pages/results/review-rebalancing/review-rebalancing.ts`, `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.ts`, `frontend/src/app/pages/risk/risk-attribution/risk-attribution.ts`, `frontend/src/app/pages/risk/stress-testing/stress-testing.ts`, `frontend/src/app/pages/report/report.ts` |
| `frontend/src/app/models/risk-attribution.model.ts` | DRIVING ·  pianificato | Mirror TypeScript della scomposizione del rischio di un portafoglio già allocato in contributi additivi di Eulero — l'output del sottomodulo optimizer/attribution/ — con gli indici di concentrazione e diversificazione che ne derivano. | `frontend/src/app/pages/risk/risk-attribution/risk-attribution.ts`, `frontend/src/app/models/portfolio.model.ts`, `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.ts` |
| `frontend/src/app/models/risk-monitoring.model.ts` | DRIVING ·  pianificato | Mirror TypeScript delle misure di rischio in condizioni ordinarie calcolate on-demand sui pesi effettivamente detenuti dal fondo — VaR, CVaR/ES correnti e le tre misure da drawdown — che la pagina Risk Monitoring espone come cruscotto post-investimento distinto dal risk_verdict di una singola deliberazione. | `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.ts`, `frontend/src/app/models/portfolio.model.ts` |
| `frontend/src/app/models/risk-verdict.model.ts` | DRIVING ·  pianificato | Mirror TypeScript di RiskVerdict, l'oggetto che l'agente risk scrive in FundState: la valutazione deterministica dell'allocazione proposta contro misure che l'ottimizzatore non ha ottimizzato. | `frontend/src/app/pages/fund/risk-agent/risk-agent.ts`, `frontend/src/app/models/fund-state.model.ts`, `frontend/src/app/models/risk-attribution.model.ts`, `frontend/src/app/models/stress-testing.model.ts` |
| `frontend/src/app/models/stress-testing.model.ts` | DRIVING ·  pianificato | Mirror TypeScript del risultato di uno stress test diretto o inverso — la ricerca sistematica del caso peggiore entro un ellissoide di plausibilità di Mahalanobis — che la pagina Stress Testing & Scenarios espone. | `frontend/src/app/pages/risk/stress-testing/stress-testing.ts`, `frontend/src/app/models/portfolio.model.ts` |
| `frontend/src/app/models/turbulence.model.ts` | DRIVING ·  pianificato | Mirror TypeScript degli indicatori osservabili di stato del mercato — turbulence e sua decomposizione, absorption ratio, rango effettivo, crescita della PC1 — prodotti dal sottomodulo optimizer/regimes/ e mostrati dalla pagina Turbulence & Systemic Risk. | `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.ts`, `frontend/src/app/models/macro-view.model.ts` |
| `frontend/src/app/pages/advanced-signals/advanced-signals.routes.ts` | DRIVING ·  pianificato | La tabella di instradamento figlia della sezione di navigazione Advanced Signals: mappa i tre URL della sezione (alternative-data-sentiment, market-regimes, factor-timing-rotation) ciascuno al proprio componente di pagina. | `frontend/src/app/pages/advanced-signals/alternative-data-sentiment/alternative-data-sentiment.ts`, `frontend/src/app/pages/advanced-signals/market-regimes/market-regimes.ts`, `frontend/src/app/pages/advanced-signals/factor-timing-rotation/factor-timing-rotation.ts`, `frontend/src/app/app.routes.ts` |
| `frontend/src/app/pages/advanced-signals/alternative-data-sentiment/alternative-data-sentiment.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Alternative Data & Sentiment: layout della SectionCardGrid della libreria segnali, del cambio di layout fra le tre tab (Signal Library / Cross-Sectional Sentiment / ML Aggregation), e della coppia sticky ScopeSwitcher+InfoCard. | `frontend/src/app/pages/advanced-signals/alternative-data-sentiment/alternative-data-sentiment.html` |
| `frontend/src/app/pages/advanced-signals/alternative-data-sentiment/alternative-data-sentiment.html` | DRIVING ·  pianificato | Il template della pagina Alternative Data & Sentiment: markup dichiarativo di ScopeSwitcher, delle due InfoCard (copertura/freshness e criterio diagnostico), TabBar, FilterChipBar, SectionCardGrid della libreria segnali, HeroStatCard di dettaglio, ActionButtonRow, e della coppia ParameterPanel/DataTable/ChartPanel sostituita per le tab Cross-Sectional Sentiment e ML Aggregation. | `frontend/src/app/pages/advanced-signals/alternative-data-sentiment/alternative-data-sentiment.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/advanced-signals/alternative-data-sentiment/alternative-data-sentiment.ts` | DRIVING ·  pianificato | Il componente Angular standalone della pagina Alternative Data & Sentiment: rende operativo il criterio diagnostico che distingue sentiment (pressione transitoria seguita da inversione) e informazione (impatto permanente), le procedure con cui testo, attenzione e umore vengono trasformati in segnali quantitativi, la vista di sezione trasversale condizionata al sentiment, e il quadro di aggregazione via machine learning con cui segnali numerosi, rumorosi e correlati vengono combinati in una misura del premio al rischio. | `frontend/src/app/pages/advanced-signals/alternative-data-sentiment/alternative-data-sentiment.html`, `frontend/src/app/pages/advanced-signals/alternative-data-sentiment/alternative-data-sentiment.css`, `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/app.routes.ts`, `frontend/src/app/pages/build/views-builder`, `frontend/src/app/pages/build/signals-factors` |
| `frontend/src/app/pages/advanced-signals/factor-timing-rotation/factor-timing-rotation.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Factor Timing & Rotation: layout del Toolbar sticky, della coppia di InfoCard, delle righe espandibili della DataTable dei segnali e del TimingComparisonPanel affiancato al riquadro del guadagno in utilità attesa. | `frontend/src/app/pages/advanced-signals/factor-timing-rotation/factor-timing-rotation.html` |
| `frontend/src/app/pages/advanced-signals/factor-timing-rotation/factor-timing-rotation.html` | DRIVING ·  pianificato | Il template della pagina Factor Timing & Rotation: markup di Toolbar, InfoCard Regime & Market State, InfoCard Value Spread, ChartPanel del rendimento in eccesso, DataTable dei segnali per fattore, TimingComparisonPanel, StatusBanner, ActionButtonRow. | `frontend/src/app/pages/advanced-signals/factor-timing-rotation/factor-timing-rotation.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/advanced-signals/factor-timing-rotation/factor-timing-rotation.ts` | DRIVING ·  pianificato | Il componente Angular standalone della pagina Factor Timing & Rotation: rende operativa la ciclicità dei premi fattoriali e la sua dipendenza dal regime (momentum condizionato allo stato di mercato, value contro-ciclico), la diagnostica di value spread per il timing di valutazione, il motore a regimi condiviso con l'allocazione, e il confronto — vincolato dall'assenza di near-arbitrage — fra le varianti di portafoglio di timing e il portafoglio ottimo equivalente allo stochastic discount factor. | `frontend/src/app/pages/advanced-signals/factor-timing-rotation/factor-timing-rotation.html`, `frontend/src/app/pages/advanced-signals/factor-timing-rotation/factor-timing-rotation.css`, `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/app.routes.ts`, `frontend/src/app/pages/build/views-builder`, `frontend/src/app/pages/fund/macro-agent/macro-agent.ts`, `frontend/src/app/pages/advanced-signals/market-regimes/market-regimes.ts` |
| `frontend/src/app/pages/advanced-signals/market-regimes/market-regimes.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Market Regimes: layout del Toolbar sticky, delle barre di probabilità di HeroStatCard, dell'area impilata di Probability Path e della griglia dei restanti ChartPanel. | `frontend/src/app/pages/advanced-signals/market-regimes/market-regimes.html` |
| `frontend/src/app/pages/advanced-signals/market-regimes/market-regimes.html` | DRIVING ·  pianificato | Il template della pagina Market Regimes: markup di Toolbar, HeroStatCard, ChartPanel Probability Path, DataTable Regime Statistics, dei ChartPanel Correlation Diagnostic/Macro Nowcast/Momentum State/Size Premium/Value Premium, ContextBridgeCard e InfoCard diagnostica. | `frontend/src/app/pages/advanced-signals/market-regimes/market-regimes.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/advanced-signals/market-regimes/market-regimes.ts` | DRIVING ·  pianificato | Il componente Angular standalone della pagina Market Regimes: rende operativo il modello a cambio di regime di Markov e il suo filtro ricorsivo (probabilità filtrate/smussate), le diagnostiche di correlazione e volatilità condizionate al regime, il nowcast macro e il diffusion index, lo stato di momentum e i premi size/value condizionati al regime, e il ponte dal regime stimato a una view grezza per il modello di Black-Litterman. | `frontend/src/app/pages/advanced-signals/market-regimes/market-regimes.html`, `frontend/src/app/pages/advanced-signals/market-regimes/market-regimes.css`, `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/app.routes.ts`, `frontend/src/app/pages/build/views-builder`, `frontend/src/app/pages/fund/macro-agent/macro-agent.ts` |
| `frontend/src/app/pages/approvals/approval-gate/approval-gate.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Human Approval Gate. | `frontend/src/app/pages/approvals/approval-gate/approval-gate.ts` |
| `frontend/src/app/pages/approvals/approval-gate/approval-gate.html` | DRIVING ·  pianificato | Il template della pagina 'Human Approval Gate': la disposizione a schermo di StatusBanner, FilterChipBar, CollectionStatBar, SectionCardGrid della coda, DetailPanel con StepTracker/InfoCard di verifiche/ActionButtonRow o InfoCard di esito storico, e InfoCard di audit trail. | `frontend/src/app/pages/approvals/approval-gate/approval-gate.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/approvals/approval-gate/approval-gate.ts` | DRIVING ·  pianificato | Il componente Angular standalone della pagina 'Human Approval Gate': lo stato e la logica che rendono operativo il terzo dei quattro passi obbligati della pipeline di approvazione trade — il punto in cui un umano firma o respinge un ordine proposto dall'agente di esecuzione prima che possa avere effetto reale (inoltro al broker) o autorizzare un piazzamento manuale. | `frontend/src/app/pages/approvals/approval-gate/approval-gate.html`, `frontend/src/app/pages/approvals/approval-gate/approval-gate.css`, `frontend/src/app/services/approval-queue.service.ts`, `frontend/src/app/models/order.model.ts`, `frontend/src/app/shared/`, `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.ts`, `frontend/src/app/pages/fund/` |
| `frontend/src/app/pages/approvals/approvals.routes.ts` | DRIVING ·  pianificato | La tabella di instradamento figlia per il gruppo di sidebar APPROVALS, che il guscio mostra con due sotto-voci (Human Approval Gate, Guardrail & Kill-Switch) sotto un'unica etichetta di gruppo — l'analogo, per questa sezione, di quanto results.routes.ts fa per il gruppo RESULTS. | `frontend/src/app/pages/approvals/approval-gate/approval-gate.ts`, `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.ts`, `frontend/src/app/app.routes.ts` |
| `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Guardrail & Kill-Switch. | `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.ts` |
| `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.html` | DRIVING ·  pianificato | Il template della pagina 'Guardrail & Kill-Switch': la disposizione a schermo della HeroStatCard del kill-switch, della pipeline a quattro passi, della griglia dei limiti rigidi, degli indici di diversificazione e dell'audit trail filtrabile. | `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.ts` | DRIVING ·  pianificato | Il componente Angular standalone della pagina 'Guardrail & Kill-Switch': lo stato e la logica del controllo a priorità più alta di tutta la sezione Approvals — il confine deterministico pass/fail (validatori Pydantic v2 e kill-switch) che precede sempre, senza alcuna chiamata LLM, i primi due dei quattro passi obbligati della pipeline di approvazione trade. | `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.html`, `frontend/src/app/pages/approvals/guardrail-killswitch/guardrail-killswitch.css`, `frontend/src/app/services/guardrail.service.ts`, `frontend/src/app/models/guardrail.model.ts`, `frontend/src/app/shared/`, `frontend/src/app/pages/approvals/approval-gate/approval-gate.ts`, `frontend/src/app/pages/risk/` |
| `frontend/src/app/pages/build/build-step-tracker.service.ts` | DRIVING ·  pianificato | Servizio Angular iniettabile che porta lo stato condiviso, basato su signal, del wizard Build a cinque tappe: tappa corrente, tappe già raggiungibili e presenza di un warning bloccante sulla tappa attiva. | `frontend/src/app/pages/build/universe-data/universe-data.page.ts`, `frontend/src/app/pages/build/signals-factors/signals-factors.page.ts`, `frontend/src/app/pages/build/views-builder/views-builder.page.ts`, `frontend/src/app/pages/build/objective-constraints/objective-constraints.page.ts`, `frontend/src/app/pages/build/run-solver-diagnostics/run-solver-diagnostics.page.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/build/build.routes.ts` | DRIVING ·  pianificato | Le rotte figlie del wizard Build a cinque tappe: un array `Route` che associa ciascun passo (universe-data, signals-factors, views-builder, objective-constraints, run-solver-diagnostics) al proprio componente pagina standalone, caricato lazy via `loadComponent`. | `frontend/src/app/pages/build/universe-data/universe-data.page.ts`, `frontend/src/app/pages/build/signals-factors/signals-factors.page.ts`, `frontend/src/app/pages/build/views-builder/views-builder.page.ts`, `frontend/src/app/pages/build/objective-constraints/objective-constraints.page.ts`, `frontend/src/app/pages/build/run-solver-diagnostics/run-solver-diagnostics.page.ts`, `frontend/src/app/pages/build/build-step-tracker.service.ts`, `frontend/src/app/app.routes.ts` |
| `frontend/src/app/pages/build/objective-constraints/objective-constraints.page.css` | DRIVING ·  pianificato | Fogli di stile scoped per la tabella Weight bounds scrollabile orizzontalmente con intestazioni fisse sotto 768px, e per l'evidenza visiva — mai a solo colore — dell'avviso NP-hard e degli errori di validazione inline. | `frontend/src/app/pages/build/objective-constraints/objective-constraints.page.html` |
| `frontend/src/app/pages/build/objective-constraints/objective-constraints.page.html` | DRIVING ·  pianificato | Template che compone Breadcrumb, InfoCard di dipendenza, il radiogroup SectionCardGrid delle nove card obiettivo, i ParameterPanel di parametri/vincoli di peso/cardinalità/drawdown/Advanced solver, l'InfoCard di riepilogo e ActionButtonRow. | `frontend/src/app/pages/build/objective-constraints/objective-constraints.page.ts`, `frontend/src/app/pages/build/objective-constraints/objective-constraints.page.css`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/build/objective-constraints/objective-constraints.page.ts` | DRIVING ·  pianificato | Componente standalone che implementa la quarta tappa del wizard Build, 'Objective & Constraints': selezione esclusiva dell'obiettivo di ottimizzazione fra nove famiglie (Mean-Variance, Minimize CVaR, Minimize Drawdown, Max Return/Drawdown, Equal Risk Contribution/Risk Budgeting, Most Diversified Portfolio, Minimize MAD, Mean-Variance-Skewness, Robust Mean-Variance), i relativi parametri, i vincoli di peso/cardinalità/lotti/drawdown e il pannello Advanced solver. | `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/pages/build/build-step-tracker.service.ts`, `frontend/src/app/pages/build/build.routes.ts`, `frontend/src/app/pages/build/views-builder/views-builder.page.ts`, `frontend/src/app/pages/build/run-solver-diagnostics/run-solver-diagnostics.page.ts` |
| `frontend/src/app/pages/build/run-solver-diagnostics/run-solver-diagnostics.page.css` | DRIVING ·  pianificato | Fogli di stile scoped per l'impilamento verticale di ChartPanel e InfoCard Optimality certificate sotto 1024px, per lo sticky di Breadcrumb+HeroStatCard durante lo scroll e per i contrasti dei badge di stato del run. | `frontend/src/app/pages/build/run-solver-diagnostics/run-solver-diagnostics.page.html` |
| `frontend/src/app/pages/build/run-solver-diagnostics/run-solver-diagnostics.page.html` | DRIVING ·  pianificato | Template che compone Breadcrumb, HeroStatCard (badge di stato, timer, connessione SSE), le InfoCard di Problem classification/Complexity notice/Optimality certificate/Penalty-swarm diagnostics, il ChartPanel di convergenza, l'EventLogPanel e ActionButtonRow. | `frontend/src/app/pages/build/run-solver-diagnostics/run-solver-diagnostics.page.ts`, `frontend/src/app/pages/build/run-solver-diagnostics/run-solver-diagnostics.page.css`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/build/run-solver-diagnostics/run-solver-diagnostics.page.ts` | DRIVING ·  pianificato | Componente standalone che implementa la quinta e ultima tappa del wizard Build, 'Run & Solver Diagnostics': osserva in streaming l'esecuzione asincrona di un run di ottimizzazione, distinguendo il percorso interamente convesso (classificazione LP⊂QP⊂QCQP⊂SOCP, certificato primale/duale/KKT) dal percorso misto-intero (avviso NP, diagnostica di penalità/sciame PSO), con log eventi e riconnessione. | `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/pages/build/build-step-tracker.service.ts`, `frontend/src/app/pages/build/build.routes.ts`, `frontend/src/app/pages/build/objective-constraints/objective-constraints.page.ts`, `frontend/src/app/pages/results/` |
| `frontend/src/app/pages/build/signals-factors/signals-factors.page.css` | DRIVING ·  pianificato | Fogli di stile scoped per la griglia a due colonne (libreria a sinistra a larghezza fissa più pannello di dettaglio) e per i contrasti degli StatusBadge (Draft/Estimated/Stale). | `frontend/src/app/pages/build/signals-factors/signals-factors.page.html` |
| `frontend/src/app/pages/build/signals-factors/signals-factors.page.html` | DRIVING ·  pianificato | Template che compone TabBar, SectionCardGrid della libreria (modelli fattoriali, universo filtrato o libreria segnali a seconda del tab attivo), il pannello di dettaglio (Specification, Variance Explained, Factor Exposures, Smart Beta Tilt oppure Combine & Diagnostics) e ActionButtonRow, riusando i widget di shared/. | `frontend/src/app/pages/build/signals-factors/signals-factors.page.ts`, `frontend/src/app/pages/build/signals-factors/signals-factors.page.css`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/build/signals-factors/signals-factors.page.ts` | DRIVING ·  pianificato | Componente standalone che implementa la seconda tappa del wizard Build, 'Signals & Factors': libreria dei modelli fattoriali (macroeconomici, di stile, statistici via PCA), screener su fondamentali, libreria dei segnali alpha e pannello Combine & Diagnostics su information coefficient/information ratio/legge fondamentale della gestione attiva. | `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/pages/build/build-step-tracker.service.ts`, `frontend/src/app/pages/build/build.routes.ts`, `frontend/src/app/pages/build/universe-data/universe-data.page.ts`, `frontend/src/app/pages/build/views-builder/views-builder.page.ts` |
| `frontend/src/app/pages/build/universe-data/universe-data.page.css` | DRIVING ·  pianificato | Fogli di stile scoped del componente Universe & Data: griglia dei pannelli, comportamento sticky di ActionButtonRow, scroll orizzontale contenuto delle sole tabelle e soglie responsive a 320/768/1024/1440px. | `frontend/src/app/pages/build/universe-data/universe-data.page.html` |
| `frontend/src/app/pages/build/universe-data/universe-data.page.html` | DRIVING ·  pianificato | Template del componente Universe & Data: dichiara la struttura a pannelli — StepTracker, ricerca/aggiunta ticker, tabella dell'universo, finestra campionaria, winsorizzazione, expected returns, covariance estimation, InfoCard di validazione, ActionButtonRow — componendo i widget di shared/ con i binding verso i signal di universe-data.page.ts. | `frontend/src/app/pages/build/universe-data/universe-data.page.ts`, `frontend/src/app/pages/build/universe-data/universe-data.page.css`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/build/universe-data/universe-data.page.ts` | DRIVING ·  pianificato | Componente standalone che implementa la prima tappa del wizard Build, 'Universe & Data': selezione dell'universo investibile, finestra e frequenza campionaria, winsorizzazione e scelta dello stimatore di covarianza (campionaria, shrinkage lineare/non lineare, EWMA, DCC), con le rispettive diagnostiche. | `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/pages/build/build-step-tracker.service.ts`, `frontend/src/app/pages/build/build.routes.ts`, `frontend/src/app/pages/build/signals-factors/signals-factors.page.ts` |
| `frontend/src/app/pages/build/views-builder/views-builder.page.css` | DRIVING ·  pianificato | Fogli di stile scoped per l'affiancamento delle InfoCard P matrix/Ω matrix che passa a impilato sotto 768px, e per l'evidenziazione delle EntityCard con StatusBadge Incomplete/Invalid. | `frontend/src/app/pages/build/views-builder/views-builder.page.html` |
| `frontend/src/app/pages/build/views-builder/views-builder.page.html` | DRIVING ·  pianificato | Template che compone ActionButtonRow/StatusBanner, l'InfoCard dei parametri globali (τ, a, freschezza di x_M/V), la SectionCardGrid delle view esistenti, il ParameterPanel di editing inline e le InfoCard di anteprima (matrice P, matrice Ω, Prior→Posterior, Weights, Uncertainty). | `frontend/src/app/pages/build/views-builder/views-builder.page.ts`, `frontend/src/app/pages/build/views-builder/views-builder.page.css`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/build/views-builder/views-builder.page.ts` | DRIVING ·  pianificato | Componente standalone che implementa la terza tappa del wizard Build, 'Views Builder': l'editor Black-Litterman delle view assolute/relative dell'investitore (matrici P, Q, Ω) con anteprima del confronto fra la distribuzione a priori di equilibrio (Π*, x_M, V) e quella a posteriori (r_BL, x_BL, V_BL). | `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/pages/build/build-step-tracker.service.ts`, `frontend/src/app/pages/build/build.routes.ts`, `frontend/src/app/pages/build/universe-data/universe-data.page.ts`, `frontend/src/app/pages/build/signals-factors/signals-factors.page.ts` |
| `frontend/src/app/pages/dashboard/dashboard.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Dashboard: regole di layout a griglia per i gruppi di regioni del template (le due HeroStatCard di KEY METRICS, i sette EntityCard dello stepper di asset allocation, i sei EntityCard delle condizioni operative, i quattro EntityCard degli agenti del fondo, l'elenco delle approvazioni pendenti, il feed di attività). | `frontend/src/app/pages/dashboard/dashboard.html` |
| `frontend/src/app/pages/dashboard/dashboard.html` | DRIVING ·  pianificato | Il template della pagina Dashboard: markup dichiarativo di tutte le regioni del wireframe — CollectionStatBar, la griglia KEY METRICS con le due HeroStatCard di rendimento e rischio, l'InfoCard di posizione media-varianza, lo stepper a sette fasi del processo di asset allocation, la griglia delle sei condizioni operative, la griglia di stato dei quattro agenti del fondo, le approvazioni pendenti e il feed di attività recente. | `frontend/src/app/pages/dashboard/dashboard.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/dashboard/dashboard.ts` | DRIVING ·  pianificato | Il componente Angular standalone che implementa la pagina Dashboard, la feature 'optimizer dashboard' distinta — nell'architettura a tre feature del frontend — dalla deliberation view e dalle approvals. È il punto in cui il portafoglio attivo, le sue misure di redditività e rischio, il criterio di dominanza media-varianza, il processo di asset allocation in sette fasi e le sei condizioni per un vantaggio operativo reale vengono riuniti in un'unica vista di riepilogo. | `frontend/src/app/pages/dashboard/dashboard.html`, `frontend/src/app/pages/dashboard/dashboard.css`, `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/app.routes.ts`, `frontend/src/app/pages/frontier-results/`, `frontend/src/app/pages/fund-deliberation/`, `frontend/src/app/pages/approval-gate/` |
| `frontend/src/app/pages/fund/allocation-agent/allocation-agent.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Allocation Agent: layout del ParameterPanel a blocchi condizionali, di DataTable e dei due ChartPanel affiancati. | `frontend/src/app/pages/fund/allocation-agent/allocation-agent.html` |
| `frontend/src/app/pages/fund/allocation-agent/allocation-agent.html` | DRIVING ·  pianificato | Il template della pagina Allocation Agent: markup di HeroStatCard, InfoCard di input/diagnostica, ParameterPanel, DataTable, dei due ChartPanel e di StatusBanner/EventLogPanel. | `frontend/src/app/pages/fund/allocation-agent/allocation-agent.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/fund/allocation-agent/allocation-agent.ts` | DRIVING ·  pianificato | Il componente Angular standalone della pagina Allocation Agent: rende operativa la traduzione dello stato macro z_t ricevuto dal FundState in pesi di portafoglio, secondo le due famiglie di metodi — innesto plug-in a due stadi oppure modellazione diretta dei pesi — descritte in frontend/13. | `frontend/src/app/pages/fund/allocation-agent/allocation-agent.html`, `frontend/src/app/pages/fund/allocation-agent/allocation-agent.css`, `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/pages/fund/macro-agent/macro-agent.ts`, `frontend/src/app/pages/fund/risk-agent/risk-agent.ts` |
| `frontend/src/app/pages/fund/deliberation/deliberation.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Fund Deliberation: layout del diagramma AgentTopology e delle tabelle di FundStateInspector e dei due EventLogPanel. | `frontend/src/app/pages/fund/deliberation/deliberation.html` |
| `frontend/src/app/pages/fund/deliberation/deliberation.html` | DRIVING ·  pianificato | Il template della pagina Fund Deliberation: markup dichiarativo di Toolbar, AgentTopology, FundStateInspector, dei due EventLogPanel e di ActionButtonRow. | `frontend/src/app/pages/fund/deliberation/deliberation.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/fund/deliberation/deliberation.ts` | DRIVING ·  pianificato | Il componente Angular standalone della pagina Fund Deliberation: rende visibile la deliberazione dei quattro agenti (macro, allocation, risk, execution) come sequenza di scritture sullo stato condiviso FundState, ricevute in tempo reale via streaming SSE per-nodo, e presenta lo stesso FundState come registro di audit checkpointato a ogni passo. | `frontend/src/app/pages/fund/deliberation/deliberation.html`, `frontend/src/app/pages/fund/deliberation/deliberation.css`, `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/pages/fund/macro-agent/macro-agent.ts`, `frontend/src/app/pages/fund/allocation-agent/allocation-agent.ts`, `frontend/src/app/pages/fund/risk-agent/risk-agent.ts`, `frontend/src/app/pages/fund/execution-agent/execution-agent.ts`, `frontend/src/app/pages/approvals` |
| `frontend/src/app/pages/fund/execution-agent/execution-agent.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Execution & Orders Agent: layout di DataTable, del DetailPanel a scomparsa laterale e degli sparkline di scheduling. | `frontend/src/app/pages/fund/execution-agent/execution-agent.html` |
| `frontend/src/app/pages/fund/execution-agent/execution-agent.html` | DRIVING ·  pianificato | Il template della pagina Execution & Orders Agent: markup di Toolbar, StatusBanner broker, FilterChipBar, DataTable degli ordini, EventLogPanel, DetailPanel a scomparsa laterale con quattro schede. | `frontend/src/app/pages/fund/execution-agent/execution-agent.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/fund/execution-agent/execution-agent.ts` | DRIVING ·  pianificato | Il componente Angular standalone della pagina Execution & Orders Agent: rende la traduzione del portafoglio-obiettivo in una lista di ordini proposti (proposed_orders), con traiettoria di negoziazione e stima del costo di market impact, entro il confine per cui l'agente propone soltanto — non esegue mai — un ordine. | `frontend/src/app/pages/fund/execution-agent/execution-agent.html`, `frontend/src/app/pages/fund/execution-agent/execution-agent.css`, `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/pages/approvals`, `frontend/src/app/pages/fund/deliberation/deliberation.ts`, `frontend/src/app/pages/results` |
| `frontend/src/app/pages/fund/fund.routes.ts` | DRIVING ·  pianificato | La tabella di instradamento figlia della sezione di navigazione Fund: mappa i cinque URL della sezione (deliberation, macro-agent, allocation-agent, risk-agent, execution-agent) ciascuno al proprio componente di pagina. | `frontend/src/app/pages/fund/deliberation/deliberation.ts`, `frontend/src/app/pages/fund/macro-agent/macro-agent.ts`, `frontend/src/app/pages/fund/allocation-agent/allocation-agent.ts`, `frontend/src/app/pages/fund/risk-agent/risk-agent.ts`, `frontend/src/app/pages/fund/execution-agent/execution-agent.ts`, `frontend/src/app/app.routes.ts` |
| `frontend/src/app/pages/fund/macro-agent/macro-agent.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Macro & Regimes Agent: layout dell'HeroStatCard sticky, della griglia a due colonne delle InfoCard superiori e del cruscotto dei segnali. | `frontend/src/app/pages/fund/macro-agent/macro-agent.html` |
| `frontend/src/app/pages/fund/macro-agent/macro-agent.html` | DRIVING ·  pianificato | Il template della pagina Macro & Regimes Agent: markup di HeroStatCard, TabBar, delle InfoCard di probabilità di regime e view bridge, del cruscotto segnali e delle InfoCard di riepilogo stress/handoff. | `frontend/src/app/pages/fund/macro-agent/macro-agent.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/fund/macro-agent/macro-agent.ts` | DRIVING ·  pianificato | Il componente Angular standalone della pagina Macro & Regimes Agent: rende operativa la stima del regime di mercato, il ponte verso le view di Black-Litterman, il cruscotto dei segnali predittivi validati, il riepilogo dello stress test e l'handoff verso lo stato condiviso, organizzati nei sei tab di frontend/12 (Overview, Regime Filter & Nowcast, Predictive Signals, Correlations by Regime, Stress Scenarios, Handoff Log). | `frontend/src/app/pages/fund/macro-agent/macro-agent.html`, `frontend/src/app/pages/fund/macro-agent/macro-agent.css`, `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/pages/build`, `frontend/src/app/pages/advanced-signals`, `frontend/src/app/pages/risk`, `frontend/src/app/pages/fund/deliberation/deliberation.ts` |
| `frontend/src/app/pages/fund/risk-agent/risk-agent.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Risk Agent: layout delle cinque InfoCard delle misure di rischio e della FilterChipBar. | `frontend/src/app/pages/fund/risk-agent/risk-agent.html` |
| `frontend/src/app/pages/fund/risk-agent/risk-agent.html` | DRIVING ·  pianificato | Il template della pagina Risk Agent: markup di Breadcrumb, HeroStatCard, SectionCardGrid degli input a monte, InfoCard di calcolo deterministico, FilterChipBar, SectionCardGrid delle metriche, InfoCard del verdetto/tool call log/trust boundary, ActionButtonRow. | `frontend/src/app/pages/fund/risk-agent/risk-agent.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/fund/risk-agent/risk-agent.ts` | DRIVING ·  pianificato | Il componente Angular standalone della pagina Risk Agent: espone il RiskVerdict prodotto sull'allocazione proposta, con il calcolo del rischio sempre eseguito deterministicamente dal codice e il modello limitato a interpretarne i numeri, entro il confine di fiducia per cui gli strumenti accessibili al modello sono solo read-only o di puro calcolo. | `frontend/src/app/pages/fund/risk-agent/risk-agent.html`, `frontend/src/app/pages/fund/risk-agent/risk-agent.css`, `frontend/src/app/services/`, `frontend/src/app/models/`, `frontend/src/app/shared/`, `frontend/src/app/pages/fund/allocation-agent/allocation-agent.ts`, `frontend/src/app/pages/fund/macro-agent/macro-agent.ts`, `frontend/src/app/pages/approvals` |
| `frontend/src/app/pages/report/report.css` | DRIVING ·  pianificato | Il foglio di stile con scope al solo componente Report & Audit trail. | `frontend/src/app/pages/report/report.ts` |
| `frontend/src/app/pages/report/report.html` | DRIVING ·  pianificato | Il template della pagina 'Report & Audit trail': la disposizione a schermo di TabBar, ActionButtonRow di export, FilterChipBar, InfoCard di postura d'audit, CollectionStatBar, DataTable paginato e DetailPanel a comparsa laterale, con contenuto di DataTable/DetailPanel che cambia secondo il tab attivo. | `frontend/src/app/pages/report/report.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/report/report.ts` | DRIVING ·  pianificato | Il componente Angular standalone che implementa la pagina 'Report & Audit trail' — l'unica voce di sidebar non raggruppata sotto una sezione (a differenza di BUILD/RESULTS/FUND/APPROVALS/RISK/ADVANCED SIGNALS), che rende operativo lo storico delle decisioni prese dai quattro agenti, lo stato differito di riproducibilità/determinismo, il requisito di idempotenza e riconciliazione degli ordini verso il broker e la provenienza/sincronizzazione delle posizioni. | `frontend/src/app/pages/report/report.html`, `frontend/src/app/pages/report/report.css`, `frontend/src/app/services/audit.service.ts`, `frontend/src/app/models/audit-decision.model.ts`, `frontend/src/app/models/order.model.ts`, `frontend/src/app/models/portfolio.model.ts`, `frontend/src/app/shared/`, `frontend/src/app/app.routes.ts`, `frontend/src/app/pages/fund/`, `frontend/src/app/pages/approvals/` |
| `frontend/src/app/pages/results/backtest-validation/backtest-validation.css` | DRIVING ·  pianificato | Il foglio di stile della pagina 'Backtest & Validation'. | `frontend/src/app/pages/results/backtest-validation/backtest-validation.ts` |
| `frontend/src/app/pages/results/backtest-validation/backtest-validation.html` | DRIVING ·  pianificato | Il template della pagina 'Backtest & Validation': la disposizione delle sette sezioni ancorate (Trial Log, OOS Test, Multiple Testing, Haircut SR, Deflated SR, PBO/CSCV, Rationale) sotto lo StatusBadge riassuntivo di HeroStatCard. | `frontend/src/app/pages/results/backtest-validation/backtest-validation.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/results/backtest-validation/backtest-validation.ts` | DRIVING ·  pianificato | Il componente standalone della pagina 'Backtest & Validation': lo stato e la logica che rendono operativo il controllo di validità di una strategia prima che i suoi risultati vengano presi per buoni, attraverso sei verifiche indipendenti (disclosure delle prove, test fuori campione, correzione per test multipli, haircut Sharpe ratio, deflated Sharpe ratio, PBO/CSCV) più la motivazione economica. | `frontend/src/app/pages/results/backtest-validation/backtest-validation.html`, `frontend/src/app/pages/results/backtest-validation/backtest-validation.css`, `frontend/src/app/services/backtest-validation.service.ts`, `frontend/src/app/models/backtest-validation.model.ts`, `frontend/src/app/shared/`, `frontend/src/app/pages/results/results-frontier/results-frontier.ts`, `frontend/src/app/pages/approvals/` |
| `frontend/src/app/pages/results/results-frontier/results-frontier.css` | DRIVING ·  pianificato | Il foglio di stile della pagina 'Results & Frontier'. | `frontend/src/app/pages/results/results-frontier/results-frontier.ts` |
| `frontend/src/app/pages/results/results-frontier/results-frontier.html` | DRIVING ·  pianificato | Il template della pagina 'Results & Frontier': la disposizione a schermo delle regioni CollectionStatBar, ChartPanel della frontiera, InfoCard del portafoglio selezionato, SectionCardGrid dei portafogli notevoli, DataTable dei contributi al rischio e ActionButtonRow. | `frontend/src/app/pages/results/results-frontier/results-frontier.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/results/results-frontier/results-frontier.ts` | DRIVING ·  pianificato | Il componente standalone della pagina 'Results & Frontier': lo stato e la logica che rendono leggibile l'esito dell'ottimizzazione nel piano rischio-rendimento — il portafoglio selezionato, la frontiera efficiente calcolata per il run scelto, i portafogli notevoli (GMVP, tangenza, MDP) e la scomposizione del rischio del portafoglio selezionato nei contributi dei singoli titoli. | `frontend/src/app/pages/results/results-frontier/results-frontier.html`, `frontend/src/app/pages/results/results-frontier/results-frontier.css`, `frontend/src/app/services/optimization-run.service.ts`, `frontend/src/app/models/optimization-run.model.ts`, `frontend/src/app/models/portfolio.model.ts`, `frontend/src/app/shared/`, `frontend/src/app/pages/results/backtest-validation/backtest-validation.ts` |
| `frontend/src/app/pages/results/results.routes.ts` | DRIVING ·  pianificato | La tabella di instradamento figlia per il gruppo di sidebar 'RESULTS', che il guscio mostra con tre sotto-voci (Results & Frontier, Backtest & Validation, Review & Rebalancing) sotto un'unica etichetta di gruppo — l'analogo, per questa sezione, di quanto app.routes.ts fa per l'intera applicazione con le sezioni che raggruppano più pagine. | `frontend/src/app/pages/results/results-frontier/results-frontier.ts`, `frontend/src/app/pages/results/backtest-validation/backtest-validation.ts`, `frontend/src/app/pages/results/review-rebalancing/review-rebalancing.ts`, `frontend/src/app/app.routes.ts` |
| `frontend/src/app/pages/results/review-rebalancing/review-rebalancing.css` | DRIVING ·  pianificato | Il foglio di stile della pagina 'Review & Rebalancing'. | `frontend/src/app/pages/results/review-rebalancing/review-rebalancing.ts` |
| `frontend/src/app/pages/results/review-rebalancing/review-rebalancing.html` | DRIVING ·  pianificato | Il template della pagina 'Review & Rebalancing': la disposizione di Toolbar, classificazione del portafoglio, confronto pesi/frontiera, scomposizione dei costi, verdetto e azioni, con i due pannelli diagnostici collassabili in coda. | `frontend/src/app/pages/results/review-rebalancing/review-rebalancing.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/results/review-rebalancing/review-rebalancing.ts` | DRIVING ·  pianificato | Il componente standalone della pagina 'Review & Rebalancing': lo stato e la logica che rendono operativa la decisione di rivedere o mantenere il portafoglio corrente, confrontandolo con un portafoglio potenzialmente rivisto e restituendo un verdetto esplicito di convenienza economica al netto dei costi di revisione. | `frontend/src/app/pages/results/review-rebalancing/review-rebalancing.html`, `frontend/src/app/pages/results/review-rebalancing/review-rebalancing.css`, `frontend/src/app/services/rebalancing.service.ts`, `frontend/src/app/models/portfolio.model.ts`, `frontend/src/app/shared/`, `frontend/src/app/pages/results/backtest-validation/backtest-validation.ts`, `frontend/src/app/pages/results/results-frontier/results-frontier.ts`, `frontend/src/app/pages/fund/`, `frontend/src/app/pages/approvals/`, `frontend/src/app/pages/build/` |
| `frontend/src/app/pages/risk/risk-attribution/risk-attribution.css` | DRIVING ·  pianificato | Il foglio di stile della pagina 'Risk Attribution'. | `frontend/src/app/pages/risk/risk-attribution/risk-attribution.ts` |
| `frontend/src/app/pages/risk/risk-attribution/risk-attribution.html` | DRIVING ·  pianificato | Il template della pagina 'Risk Attribution': la disposizione di Toolbar, KeyMetricsRow, DataTable dei contributi per componente con riga TOTAL sticky, ChartPanel dei contributi di Eulero, FactorRiskImpactPanel e DetailPanel a comparsa. | `frontend/src/app/pages/risk/risk-attribution/risk-attribution.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/risk/risk-attribution/risk-attribution.ts` | DRIVING ·  pianificato | Il componente standalone della pagina 'Risk Attribution': lo stato e la logica che rendono operativa la scomposizione del rischio di un portafoglio già allocato in contributi additivi di Eulero imputabili a titoli, fattori o sotto-portafogli, con gli indici di concentrazione e diversificazione che ne derivano. | `frontend/src/app/pages/risk/risk-attribution/risk-attribution.html`, `frontend/src/app/pages/risk/risk-attribution/risk-attribution.css`, `frontend/src/app/services/risk-attribution.service.ts`, `frontend/src/app/models/risk-attribution.model.ts`, `frontend/src/app/models/portfolio.model.ts`, `frontend/src/app/shared/`, `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.ts`, `frontend/src/app/pages/approvals/` |
| `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.css` | DRIVING ·  pianificato | Il foglio di stile della pagina 'Risk Monitoring'. | `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.ts` |
| `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.html` | DRIVING ·  pianificato | Il template della pagina 'Risk Monitoring': la disposizione a schermo di Toolbar, KeyMetricsRow, tre ChartPanel (VaR & CVaR trend, underwater/drawdown, curva CDaR), DataTable dello stato delle soglie di drawdown, EventLogPanel e RelatedPagesList. | `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.ts` | DRIVING ·  pianificato | Il componente standalone della pagina 'Risk Monitoring': lo stato e la logica del cruscotto post-investimento del fondo, che legge — senza sceglierli — VaR e CVaR correnti sulla distribuzione marginale delle perdite e drawdown/MaxDD/AvDD/CDaR sulla curva di equity, entrambi calcolati sui pesi effettivamente detenuti dal portafoglio corrente. | `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.html`, `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.css`, `frontend/src/app/services/risk-monitoring.service.ts`, `frontend/src/app/models/risk-monitoring.model.ts`, `frontend/src/app/models/portfolio.model.ts`, `frontend/src/app/shared/`, `frontend/src/app/pages/risk/risk-attribution/risk-attribution.ts`, `frontend/src/app/pages/risk/stress-testing/stress-testing.ts`, `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.ts` |
| `frontend/src/app/pages/risk/risk.routes.ts` | DRIVING ·  pianificato | La tabella di instradamento figlia per il gruppo di sidebar 'RISK', che il guscio mostra con quattro sotto-voci (Risk Monitoring, Risk Attribution, Stress Testing & Scenarios, Turbulence & Systemic Risk) sotto un'unica etichetta di gruppo — l'analogo, per questa sezione, di quanto results.routes.ts fa per il gruppo RESULTS. | `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.ts`, `frontend/src/app/pages/risk/risk-attribution/risk-attribution.ts`, `frontend/src/app/pages/risk/stress-testing/stress-testing.ts`, `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.ts`, `frontend/src/app/app.routes.ts` |
| `frontend/src/app/pages/risk/stress-testing/stress-testing.css` | DRIVING ·  pianificato | Il foglio di stile della pagina 'Stress Testing & Scenarios'. | `frontend/src/app/pages/risk/stress-testing/stress-testing.ts` |
| `frontend/src/app/pages/risk/stress-testing/stress-testing.html` | DRIVING ·  pianificato | Il template della pagina 'Stress Testing & Scenarios': la disposizione di Toolbar, TabBar Forward/Reverse, DataTable dei fattori fissabili, SectionCardGrid di riepilogo scenari, i due ChartPanel di confronto e contributo, ScenarioLibrary e ReverseStressPanel. | `frontend/src/app/pages/risk/stress-testing/stress-testing.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/risk/stress-testing/stress-testing.ts` | DRIVING ·  pianificato | Il componente standalone della pagina 'Stress Testing & Scenarios': lo stato e la logica delle due modalità di lavoro — stress test diretto (raggio di plausibilità k fissato, ricerca sistematica dello scenario di massima perdita entro l'ellissoide Ell_k) e reverse stress test (esito target fissato, ricerca del raggio minimo e dello scenario implicito) — sui fattori di rischio sistematici del portafoglio selezionato. | `frontend/src/app/pages/risk/stress-testing/stress-testing.html`, `frontend/src/app/pages/risk/stress-testing/stress-testing.css`, `frontend/src/app/services/stress-testing.service.ts`, `frontend/src/app/models/stress-testing.model.ts`, `frontend/src/app/models/portfolio.model.ts`, `frontend/src/app/shared/`, `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.ts`, `frontend/src/app/pages/risk/risk-attribution/risk-attribution.ts` |
| `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.css` | DRIVING ·  pianificato | Il foglio di stile della pagina 'Turbulence & Systemic Risk'. | `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.ts` |
| `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.html` | DRIVING ·  pianificato | Il template della pagina 'Turbulence & Systemic Risk': la disposizione di KeyMetricsRow, SectionLabel, SectionCardGrid a sei indicatori, SectionNav, Toolbar, i sei ChartPanel di dettaglio, le due DataTable e l'InfoCard statica di collocazione. | `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.ts`, `frontend/src/app/shared/` |
| `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.ts` | DRIVING ·  pianificato | Il componente standalone della pagina 'Turbulence & Systemic Risk': lo stato e la logica di sola lettura degli indicatori osservabili che misurano quanto lo stato corrente del mercato sia inusuale (turbulence e sua decomposizione in magnitude/correlation surprise) o fragile (absorption ratio, rango effettivo, crescita della PC1, spettro degli autovalori, participation ratio). | `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.html`, `frontend/src/app/pages/risk/turbulence-systemic/turbulence-systemic.css`, `frontend/src/app/services/turbulence.service.ts`, `frontend/src/app/models/turbulence.model.ts`, `frontend/src/app/shared/`, `frontend/src/app/pages/risk/risk-monitoring/risk-monitoring.ts`, `frontend/src/app/pages/fund/` |
| `frontend/src/app/services/api-client.ts` | DRIVING ·  pianificato | Lo strato di client HTTP tipizzato verso l'API del fondo: il punto unico in cui l'applicazione Angular sa come raggiungere fund/app/http/ — URL di base per ambiente, header comuni, e la forma delle richieste/risposte tipizzate sui modelli di frontend/src/app/models/. È lo strato che rende operativa la voce 'client tipizzato generato per l'API HTTP del fondo' descritta nella pagina del guscio applicativo. | `frontend/src/environments/`, `frontend/src/app/interceptors/`, `frontend/src/app/models/`, `frontend/src/app/services/optimize.service.ts`, `frontend/src/app/services/fund.service.ts`, `frontend/src/app/services/trades.service.ts`, `frontend/src/app/services/stream.service.ts` |
| `frontend/src/app/services/fund.service.ts` | DRIVING ·  pianificato | Il servizio tipizzato che parla con fund/app/http/routers/fund.py: avvia una delibera del fondo a quattro agenti e legge il FundState corrente o checkpointato, tenendo quel FundState — il record di audit condiviso dai quattro agenti — come signal di pagina. | `frontend/src/app/services/api-client.ts`, `frontend/src/app/services/stream.service.ts`, `frontend/src/app/models/`, `fund/app/http/routers/fund.py`, `fund/app/domain/fund/fund_state.py`, `frontend/src/app/pages/fund/` |
| `frontend/src/app/services/optimize.service.ts` | DRIVING ·  pianificato | Il servizio tipizzato che parla con fund/app/http/routers/optimize.py: avvia un'ottimizzazione di portafoglio con la configurazione (universo, obiettivo, vincoli) assemblata dalle pagine di Build, e tiene lo stato del run corrente — non solo la sua risposta finale — come signal di pagina. | `frontend/src/app/services/api-client.ts`, `frontend/src/app/services/stream.service.ts`, `frontend/src/app/models/`, `fund/app/http/routers/optimize.py`, `frontend/src/app/pages/build/`, `frontend/src/app/pages/results/` |
| `frontend/src/app/services/stream.service.ts` | DRIVING ·  pianificato | Il servizio dedicato alla sottoscrizione SSE verso fund/app/http/routers/stream.py: incanala gli eventi per-nodo della delibera e il progresso delle chiamate a portopt come flussi tipizzati, gestendo la riconnessione tramite Last-Event-ID, così che optimize.service.ts e fund.service.ts — e i pannelli che li consumano — non reimplementino ciascuno la logica di connessione. | `frontend/src/app/services/api-client.ts`, `frontend/src/app/models/`, `fund/app/http/routers/stream.py`, `fund/app/http/sse.py`, `frontend/src/app/services/optimize.service.ts`, `frontend/src/app/services/fund.service.ts` |
| `frontend/src/app/services/trades.service.ts` | DRIVING ·  pianificato | Il servizio tipizzato che parla con fund/app/http/routers/trades.py: legge la coda degli ordini proposti in attesa di firma lungo i quattro passi obbligati (pre-check deterministico, validazione a regole, gate umano, adapter broker) e registra la decisione umana esplicita, tenendo quella coda come signal di pagina filtrabile. | `frontend/src/app/services/api-client.ts`, `frontend/src/app/models/`, `fund/app/http/routers/trades.py`, `fund/app/domain/trading/order.py`, `frontend/src/app/pages/approvals/` |
| `frontend/src/app/shared/action-button-row/action-button-row.css` | DRIVING ·  pianificato | Foglio di stile con scope a ActionButtonRow. | `frontend/src/app/shared/action-button-row/action-button-row.ts` |
| `frontend/src/app/shared/action-button-row/action-button-row.html` | DRIVING ·  pianificato | Template di ActionButtonRow. | `frontend/src/app/shared/action-button-row/action-button-row.ts`, `frontend/src/app/shared/confirm-dialog/confirm-dialog.ts` |
| `frontend/src/app/shared/action-button-row/action-button-row.ts` | DRIVING ·  pianificato | Il componente standalone ActionButtonRow: la riga di azioni primarie/secondarie in fondo a un pannello — confronta/esporta nella pagina diagnostica, rifiuta/approva nel gate d'approvazione — con testo contestuale opzionale sopra i pulsanti. | `frontend/src/app/shared/confirm-dialog/confirm-dialog.ts`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/confirm-dialog/confirm-dialog.css` | DRIVING ·  pianificato | Foglio di stile con scope a ConfirmDialog. | `frontend/src/app/shared/confirm-dialog/confirm-dialog.ts` |
| `frontend/src/app/shared/confirm-dialog/confirm-dialog.html` | DRIVING ·  pianificato | Template di ConfirmDialog. | `frontend/src/app/shared/confirm-dialog/confirm-dialog.ts` |
| `frontend/src/app/shared/confirm-dialog/confirm-dialog.ts` | DRIVING ·  pianificato | Il componente standalone ConfirmDialog: il dialogo di conferma esplicita (Annulla/Conferma) che ogni azione distruttiva o irreversibile del frontend deve attraversare prima di eseguirsi — interrompere un run, approvare o rifiutare un trade — la controparte lato client dell'obbligo architetturale che nessuna azione con effetto reale parta senza un passaggio umano deterministico. | `frontend/src/app/shared/directives/focus-trap.directive.ts`, `frontend/src/app/shared/action-button-row/action-button-row.ts`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/cross-page-link/cross-page-link.css` | DRIVING ·  pianificato | Foglio di stile con scope a CrossPageLink. | `frontend/src/app/shared/cross-page-link/cross-page-link.ts` |
| `frontend/src/app/shared/cross-page-link/cross-page-link.html` | DRIVING ·  pianificato | Template di CrossPageLink. | `frontend/src/app/shared/cross-page-link/cross-page-link.ts` |
| `frontend/src/app/shared/cross-page-link/cross-page-link.ts` | DRIVING ·  pianificato | Il componente standalone CrossPageLink: il collegamento di navigazione fra sezioni funzionali che porta con sé il portafoglio attivo come contesto, riusato su ogni card/uscita di pagina verso un'altra pagina del frontend. | `frontend/src/app/app.routes.ts`, `frontend/src/app/pages/`, `frontend/src/app/shared/entity-card/entity-card.ts` |
| `frontend/src/app/shared/directives/aria-live-announcer.directive.ts` | DRIVING ·  pianificato | La direttiva standalone AriaLiveAnnouncer: marca un elemento come regione aria-live e ne governa il momento dell'annuncio, il requisito ripetuto testualmente in ogni pagina letta per lo stato di connessione, i cambi di stato di un run e l'esito di un refresh. | `frontend/src/app/shared/refresh-control/refresh-control.ts`, `frontend/src/app/shared/status-badge/status-badge.ts`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/directives/focus-trap.directive.ts` | DRIVING ·  pianificato | La direttiva standalone FocusTrap: intrappola il focus da tastiera dentro un popover o un dialogo finché non viene chiuso e lo restituisce al trigger che lo ha aperto — il comportamento richiesto testualmente da ScopeSwitcher, dal selettore di portafoglio del guscio e da confirm-dialog/. | `frontend/src/app/shared/confirm-dialog/confirm-dialog.ts`, `frontend/src/app/shared/shell/portfolio-scope-switcher/portfolio-scope-switcher.ts`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/empty-state/empty-state.css` | DRIVING ·  pianificato | Foglio di stile con scope a EmptyState. | `frontend/src/app/shared/empty-state/empty-state.ts` |
| `frontend/src/app/shared/empty-state/empty-state.html` | DRIVING ·  pianificato | Template di EmptyState. | `frontend/src/app/shared/empty-state/empty-state.ts` |
| `frontend/src/app/shared/empty-state/empty-state.ts` | DRIVING ·  pianificato | Il componente standalone EmptyState: il segnaposto 'nessun dato ancora disponibile' — 'No portfolio selected', 'Nessun run avviato', nessuna approvazione pendente — riusato da ogni regione di ogni pagina invece di un messaggio ad hoc per sezione. | `frontend/src/app/pages/`, `frontend/src/app/shared/action-button-row/action-button-row.ts` |
| `frontend/src/app/shared/entity-card/entity-card.css` | DRIVING ·  pianificato | Foglio di stile con scope a EntityCard. | `frontend/src/app/shared/entity-card/entity-card.ts` |
| `frontend/src/app/shared/entity-card/entity-card.html` | DRIVING ·  pianificato | Template di EntityCard. | `frontend/src/app/shared/entity-card/entity-card.ts` |
| `frontend/src/app/shared/entity-card/entity-card.ts` | DRIVING ·  pianificato | Il componente standalone EntityCard: la riga/tessera generica di un elenco o di una griglia — un portafoglio in ScopeSwitcher, una fase dello stepper di allocazione, un trade in coda d'approvazione — l'unità ripetuta che section-card-grid/ dispone. | `frontend/src/app/shared/status-badge/status-badge.ts`, `frontend/src/app/shared/cross-page-link/cross-page-link.ts`, `frontend/src/app/models/`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/error-state/error-state.css` | DRIVING ·  pianificato | Foglio di stile con scope a ErrorState. | `frontend/src/app/shared/error-state/error-state.ts` |
| `frontend/src/app/shared/error-state/error-state.html` | DRIVING ·  pianificato | Template di ErrorState. | `frontend/src/app/shared/error-state/error-state.ts`, `frontend/src/app/shared/retry-button/retry-button.ts` |
| `frontend/src/app/shared/error-state/error-state.ts` | DRIVING ·  pianificato | Il componente standalone ErrorState: il messaggio di fallimento annunciato con role='alert', accoppiato a retry-button/, riusato da HeroStatCard, ScopeSwitcher, InfoCard e da ogni pannello che dipenda da una chiamata all'API del fondo. | `frontend/src/app/shared/retry-button/retry-button.ts`, `frontend/src/app/interceptors/error.interceptor.ts`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/hero-stat-card/hero-stat-card.css` | DRIVING ·  pianificato | Foglio di stile con scope a HeroStatCard. | `frontend/src/app/shared/hero-stat-card/hero-stat-card.ts` |
| `frontend/src/app/shared/hero-stat-card/hero-stat-card.html` | DRIVING ·  pianificato | Template di HeroStatCard: markup del pannello prominente con intestazione, corpo dati e riga di azioni. | `frontend/src/app/shared/hero-stat-card/hero-stat-card.ts` |
| `frontend/src/app/shared/hero-stat-card/hero-stat-card.ts` | DRIVING ·  pianificato | Il componente standalone HeroStatCard: il pannello di riepilogo prominente in cima all'area contenuto — il portafoglio attivo nel guscio, lo stato di un run nella pagina diagnostica — riusato identicamente in più pagine invece di essere ridisegnato per ciascuna. | `frontend/src/app/shared/status-badge/status-badge.ts`, `frontend/src/app/shared/skeleton-block/skeleton-block.ts`, `frontend/src/app/shared/empty-state/empty-state.ts`, `frontend/src/app/shared/error-state/error-state.ts`, `frontend/src/app/models/`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/info-card/info-card.css` | DRIVING ·  pianificato | Foglio di stile con scope a InfoCard. | `frontend/src/app/shared/info-card/info-card.ts` |
| `frontend/src/app/shared/info-card/info-card.html` | DRIVING ·  pianificato | Template di InfoCard. | `frontend/src/app/shared/info-card/info-card.ts` |
| `frontend/src/app/shared/info-card/info-card.ts` | DRIVING ·  pianificato | Il componente standalone InfoCard: il pannello generico chiave-valore/diagnostico riusato per ogni contenuto informativo di sola lettura o quasi — connessione API, classificazione del problema, esiti dei controlli automatici, audit trail — invece di un markup ad hoc per pannello. | `frontend/src/app/shared/status-badge/status-badge.ts`, `frontend/src/app/shared/skeleton-block/skeleton-block.ts`, `frontend/src/app/shared/empty-state/empty-state.ts`, `frontend/src/app/shared/error-state/error-state.ts`, `frontend/src/app/models/`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/pipes/compact-currency.pipe.ts` | DRIVING ·  pianificato | La pipe standalone CompactCurrency: formatta in modo uniforme gli importi monetari in notazione compatta ($19.0k, $24.6k) mostrati ripetutamente nelle card di trade/ordine e negli importi di sintesi delle pagine di dominio. | `frontend/src/app/shared/entity-card/entity-card.ts`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/pipes/weight-percent.pipe.ts` | DRIVING ·  pianificato | La pipe standalone WeightPercent: formatta in modo uniforme, su ogni pagina, la percentuale di ricchezza x_i allocata a una scelta di investimento — la grandezza che l'intero frontend esiste per rendere leggibile. | `frontend/src/app/shared/hero-stat-card/hero-stat-card.ts`, `frontend/src/app/shared/entity-card/entity-card.ts`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/refresh-control/refresh-control.css` | DRIVING ·  pianificato | Foglio di stile con scope a RefreshControl. | `frontend/src/app/shared/refresh-control/refresh-control.ts` |
| `frontend/src/app/shared/refresh-control/refresh-control.html` | DRIVING ·  pianificato | Template di RefreshControl. | `frontend/src/app/shared/refresh-control/refresh-control.ts` |
| `frontend/src/app/shared/refresh-control/refresh-control.ts` | DRIVING ·  pianificato | Il componente standalone RefreshControl: l'azione di ricontrollo/ricaricamento riusata da InfoCard (connessione API), da FilterChipBar (elenco filtrato) e da qualunque pagina che ricarichi le proprie sezioni. | `frontend/src/app/shared/directives/aria-live-announcer.directive.ts`, `frontend/src/app/pages/`, `frontend/src/app/shared/shell/` |
| `frontend/src/app/shared/retry-button/retry-button.css` | DRIVING ·  pianificato | Foglio di stile con scope a RetryButton. | `frontend/src/app/shared/retry-button/retry-button.ts` |
| `frontend/src/app/shared/retry-button/retry-button.html` | DRIVING ·  pianificato | Template di RetryButton. | `frontend/src/app/shared/retry-button/retry-button.ts` |
| `frontend/src/app/shared/retry-button/retry-button.ts` | DRIVING ·  pianificato | Il componente standalone RetryButton: l'azione di nuovo tentativo mostrata dentro error-state/ ovunque una chiamata all'API del fondo sia fallita, dal guscio a un run del solutore. | `frontend/src/app/shared/error-state/error-state.ts`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/section-card-grid/section-card-grid.css` | DRIVING ·  pianificato | Foglio di stile con scope a SectionCardGrid. | `frontend/src/app/shared/section-card-grid/section-card-grid.ts` |
| `frontend/src/app/shared/section-card-grid/section-card-grid.html` | DRIVING ·  pianificato | Template di SectionCardGrid. | `frontend/src/app/shared/section-card-grid/section-card-grid.ts` |
| `frontend/src/app/shared/section-card-grid/section-card-grid.ts` | DRIVING ·  pianificato | Il componente standalone SectionCardGrid: il contenitore a griglia responsiva che dispone N istanze di entity-card/ — le sezioni funzionali nel guscio, la coda di approvazione, gli stepper di stato — con il collasso a colonna singola sotto soglia richiesto da ogni pagina letta. | `frontend/src/app/shared/entity-card/entity-card.ts`, `frontend/src/app/shared/empty-state/empty-state.ts`, `frontend/src/app/pages/` |
| `frontend/src/app/shared/shell/breadcrumb/breadcrumb.css` | DRIVING ·  pianificato | Foglio di stile con scope a Breadcrumb. | `frontend/src/app/shared/shell/breadcrumb/breadcrumb.ts` |
| `frontend/src/app/shared/shell/breadcrumb/breadcrumb.html` | DRIVING ·  pianificato | Template di Breadcrumb. | `frontend/src/app/shared/shell/breadcrumb/breadcrumb.ts` |
| `frontend/src/app/shared/shell/breadcrumb/breadcrumb.ts` | DRIVING ·  pianificato | Il componente standalone Breadcrumb: il titolo di pagina corrente con riga divisoria, in cima all'area contenuto di ogni pagina — non un percorso gerarchico a più livelli, forma che nessuna delle pagine lette mostra. | `frontend/src/app/shared/shell/nav-config.ts`, `frontend/src/app/shared/shell/shell.ts` |
| `frontend/src/app/shared/shell/nav-config.ts` | DRIVING ·  pianificato | Il modulo dati statico che dichiara la struttura delle 8 sezioni di navigazione del frontend — le due voci singole Dashboard e Report più i sei gruppi BUILD, RESULTS, FUND, APPROVALS, RISK, ADVANCED SIGNALS con le rispettive pagine figlie e i relativi percorsi. | `frontend/src/app/shared/shell/sidebar/sidebar.ts`, `frontend/src/app/shared/shell/breadcrumb/breadcrumb.ts`, `frontend/src/app/app.routes.ts` |
| `frontend/src/app/shared/shell/portfolio-scope-switcher/portfolio-scope-switcher.css` | DRIVING ·  pianificato | Foglio di stile con scope a PortfolioScopeSwitcher. | `frontend/src/app/shared/shell/portfolio-scope-switcher/portfolio-scope-switcher.ts` |
| `frontend/src/app/shared/shell/portfolio-scope-switcher/portfolio-scope-switcher.html` | DRIVING ·  pianificato | Template di PortfolioScopeSwitcher. | `frontend/src/app/shared/shell/portfolio-scope-switcher/portfolio-scope-switcher.ts`, `frontend/src/app/shared/entity-card/entity-card.ts` |
| `frontend/src/app/shared/shell/portfolio-scope-switcher/portfolio-scope-switcher.ts` | DRIVING ·  pianificato | Il componente standalone PortfolioScopeSwitcher: il selettore compatto '[ ▾ portfolio ]' nella topbar, condiviso da ogni pagina del frontend, distinto dal pannello ScopeSwitcher a piena pagina che vive nel contenuto della pagina 01. | `frontend/src/app/shared/entity-card/entity-card.ts`, `frontend/src/app/shared/status-badge/status-badge.ts`, `frontend/src/app/shared/directives/focus-trap.directive.ts`, `frontend/src/app/shared/shell/topbar/topbar.ts` |
| `frontend/src/app/shared/shell/shell.css` | DRIVING ·  pianificato | Foglio di stile con scope al layout di Shell. | `frontend/src/app/shared/shell/shell.ts` |
| `frontend/src/app/shared/shell/shell.html` | DRIVING ·  pianificato | Template di Shell: markup della griglia a due colonne (sidebar più contenuto) sormontata dalla topbar. | `frontend/src/app/shared/shell/shell.ts` |
| `frontend/src/app/shared/shell/shell.ts` | DRIVING ·  pianificato | Il componente standalone Shell: la cornice applicativa radice — griglia topbar in alto, sidebar laterale, area contenuto — identica su ogni pagina del frontend, montata una sola volta dal componente root dell'applicazione. | `frontend/src/app/shared/shell/topbar/topbar.ts`, `frontend/src/app/shared/shell/sidebar/sidebar.ts`, `frontend/src/app/shared/shell/breadcrumb/breadcrumb.ts`, `frontend/src/app/app.ts` |
| `frontend/src/app/shared/shell/sidebar/sidebar.css` | DRIVING ·  pianificato | Foglio di stile con scope a Sidebar. | `frontend/src/app/shared/shell/sidebar/sidebar.ts` |
| `frontend/src/app/shared/shell/sidebar/sidebar.html` | DRIVING ·  pianificato | Template di Sidebar. | `frontend/src/app/shared/shell/sidebar/sidebar.ts` |
| `frontend/src/app/shared/shell/sidebar/sidebar.ts` | DRIVING ·  pianificato | Il componente standalone Sidebar: la navigazione laterale identica su ogni pagina, con le 8 sezioni funzionali del frontend (Dashboard, BUILD, RESULTS, FUND, APPROVALS, RISK, ADVANCED SIGNALS, Report) e le rispettive pagine figlie. | `frontend/src/app/shared/shell/nav-config.ts`, `frontend/src/app/shared/shell/shell.ts`, `frontend/src/app/app.routes.ts` |
| `frontend/src/app/shared/shell/topbar/topbar.css` | DRIVING ·  pianificato | Foglio di stile con scope a Topbar. | `frontend/src/app/shared/shell/topbar/topbar.ts` |
| `frontend/src/app/shared/shell/topbar/topbar.html` | DRIVING ·  pianificato | Template di Topbar. | `frontend/src/app/shared/shell/topbar/topbar.ts` |
| `frontend/src/app/shared/shell/topbar/topbar.ts` | DRIVING ·  pianificato | Il componente standalone Topbar: la barra superiore identica su ogni pagina — hamburger di apertura sidebar, nome applicazione, selettore di portafoglio compatto, notifiche, cambio tema, menu utente. | `frontend/src/app/shared/shell/portfolio-scope-switcher/portfolio-scope-switcher.ts`, `frontend/src/app/shared/shell/shell.ts` |
| `frontend/src/app/shared/skeleton-block/skeleton-block.css` | DRIVING ·  pianificato | Foglio di stile con scope a SkeletonBlock. | `frontend/src/app/shared/skeleton-block/skeleton-block.ts` |
| `frontend/src/app/shared/skeleton-block/skeleton-block.html` | DRIVING ·  pianificato | Template di SkeletonBlock. | `frontend/src/app/shared/skeleton-block/skeleton-block.ts` |
| `frontend/src/app/shared/skeleton-block/skeleton-block.ts` | DRIVING ·  pianificato | Il componente standalone SkeletonBlock: il segnaposto di caricamento riusato da ogni pannello (HeroStatCard, InfoCard, ScopeSwitcher, EntityCard) durante il primo accesso o un refresh. | `frontend/src/app/pages/`, `frontend/src/app/shared/shell/` |
| `frontend/src/app/shared/status-badge/status-badge.css` | DRIVING ·  pianificato | Foglio di stile con scope a StatusBadge. | `frontend/src/app/shared/status-badge/status-badge.ts` |
| `frontend/src/app/shared/status-badge/status-badge.html` | DRIVING ·  pianificato | Template di StatusBadge. | `frontend/src/app/shared/status-badge/status-badge.ts` |
| `frontend/src/app/shared/status-badge/status-badge.ts` | DRIVING ·  pianificato | Il componente standalone StatusBadge: l'indicatore di stato con icona più parola testuale, mai a solo colore, richiesto identicamente da ogni pagina letta per badge come Valid/Active, Running/Converged/Infeasible/Unbounded/Failed/Stopped, Human Gate/Rule Validation/Pre-check. | `frontend/src/app/models/`, `frontend/src/app/pages/`, `frontend/src/app/shared/shell/` |
| `frontend/src/main.ts` | DRIVING ·  pianificato | Il punto di ingresso dell'applicazione Angular standalone: l'unico file che invoca `bootstrapApplication`, avviando il componente radice `App` con la configurazione esposta da `app.config.ts`. È il file più esterno dell'intero sistema — l'estremità dell'adattatore driving 'frontend' da cui parte l'esecuzione nel browser, prima ancora che router o client HTTP entrino in gioco. | `frontend/src/app/app.config.ts`, `frontend/src/app/app.ts` |

<!-- FINE GENERATO: frontend -->

---

## 8. Trasversale — ciò che non appartiene a un contesto

Quello che resta alla radice del monorepo non è un quinto contesto: è l'impalcatura condivisa.
Sta qui perché ha comunque bisogno di regole, e perché è il posto dove un confine mal disegnato
si nota per primo.

**`ingestion/` e `fund/` condividono il database, non il codice.** Ciascuno possiede il proprio strato
di accesso — repository e modelli — e la stessa tabella è letta attraverso due mappature
distinte. Non è duplicazione per distrazione: è il confine fra contesti delimitati. Un ORM
condiviso creerebbe un accoppiamento che nessuna delle due parti ha chiesto, e la prima
migrazione che serve solo a uno dei due lo dimostrerebbe.

**`optimizer/` non è importabile da `ingestion/`.** È una regola dura già in vigore e va tenuta:
il demone deve restare leggero e a prova di guasto, e lo stack sklearn/skfolio/scipy è
esattamente ciò che lo appesantirebbe. Il fondo, che quello stack ce l'ha, è l'unico che chiama
`portopt`.

**Il deployment è un monolite modulare.** Per una squadra piccola i microservizi costano circa
il 30-50% di sforzo operativo in più a parità di funzionalità. La forma raccomandata è un
orchestratore di agenti come monolite modulare, con tool runner e retrieval esternalizzati solo
se emerge un collo di bottiglia vero. Il primo candidato all'estrazione, se mai servisse, è
l'agente di esecuzione — se avesse bisogno di isolamento dai guasti degli altri. Finché le
giunture restano pulite (§3), estrarlo è un'operazione meccanica.

`docker-compose.yml` mette in piedi cinque servizi: `db`, `adminer`, `scheduler`, `fund`,
`frontend`. `scheduler/` sono wrapper shell sulla CLI di `api`, `scripts/` sono helper di CI, e
`tests/` è la suite della libreria — rispecchia l'albero di `optimizer/`.

<!-- GENERATO: top-level -->

_Da qui in giù la sezione è generata da `scriptorium arch` il 2026-08-19: non modificarla a mano, si riscrive a ogni assemblaggio. Il record completo — responsabilità estese e citazioni riga per riga — sta in `data/architecture-spec/`, da cui questo è renderizzato._

### 8.1 Albero

113 file, ciascuno annotato col suo ruolo esagonale. Ruoli: config 3 · DRIVING 110

```
├── docker-compose.yml  ┃ config
├── pyproject.toml  ┃ config
├── scheduler/
│   ├── fetch.sh  ┃ DRIVING
│   └── refetch_all.sh  ┃ DRIVING
├── scripts/
│   └── check_branch_coverage.py  ┃ config
└── tests/
    ├── __init__.py  ┃ DRIVING
    ├── cluster/
    │   ├── __init__.py  ┃ DRIVING
    │   └── test_factory.py  ┃ DRIVING
    ├── conftest.py  ┃ DRIVING
    ├── distance/
    │   ├── __init__.py  ┃ DRIVING
    │   └── test_factory.py  ┃ DRIVING
    ├── factors/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_alpha.py  ┃ DRIVING
    │   ├── test_config.py  ┃ DRIVING
    │   ├── test_construction.py  ┃ DRIVING
    │   ├── test_diagnostics.py  ┃ DRIVING
    │   ├── test_factor_coverage.py  ┃ DRIVING
    │   ├── test_forward_return_boundary.py  ┃ DRIVING
    │   ├── test_integration.py  ┃ DRIVING
    │   ├── test_mimicking.py  ┃ DRIVING
    │   ├── test_ml_scoring.py  ┃ DRIVING
    │   ├── test_oos_validation.py  ┃ DRIVING
    │   ├── test_pit_history.py  ┃ DRIVING
    │   ├── test_quintile.py  ┃ DRIVING
    │   ├── test_regime.py  ┃ DRIVING
    │   ├── test_scoring.py  ┃ DRIVING
    │   ├── test_sector_bands.py  ┃ DRIVING
    │   ├── test_selection.py  ┃ DRIVING
    │   ├── test_standardization.py  ┃ DRIVING
    │   └── test_validation.py  ┃ DRIVING
    ├── fx/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_converter.py  ┃ DRIVING
    │   ├── test_decomposition.py  ┃ DRIVING
    │   ├── test_pipeline_fx_integration.py  ┃ DRIVING
    │   └── test_rates.py  ┃ DRIVING
    ├── linear_model/
    │   ├── __init__.py  ┃ DRIVING
    │   └── test_factory.py  ┃ DRIVING
    ├── moments/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_config.py  ┃ DRIVING
    │   ├── test_factory.py  ┃ DRIVING
    │   ├── test_partial_fit.py  ┃ DRIVING
    │   ├── test_scaling.py  ┃ DRIVING
    │   └── test_variance.py  ┃ DRIVING
    ├── online/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_online_predict_matches_loop.py  ┃ DRIVING
    │   ├── test_partial_fit_smoke.py  ┃ DRIVING
    │   └── test_pipeline_rejected.py  ┃ DRIVING
    ├── optimization/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_benchmark_tracker.py  ┃ DRIVING
    │   ├── test_config.py  ┃ DRIVING
    │   ├── test_dr_cvar.py  ┃ DRIVING
    │   ├── test_factory.py  ┃ DRIVING
    │   ├── test_herc.py  ┃ DRIVING
    │   ├── test_hrp.py  ┃ DRIVING
    │   ├── test_max_diversification.py  ┃ DRIVING
    │   ├── test_naive.py  ┃ DRIVING
    │   ├── test_nco.py  ┃ DRIVING
    │   ├── test_regime_blended_mean_risk.py  ┃ DRIVING
    │   ├── test_region_constraints.py  ┃ DRIVING
    │   ├── test_risk_budgeting.py  ┃ DRIVING
    │   ├── test_robust.py  ┃ DRIVING
    │   ├── test_schur.py  ┃ DRIVING
    │   └── test_stacking.py  ┃ DRIVING
    ├── pipeline/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_builder.py  ┃ DRIVING
    │   ├── test_delisting_integration.py  ┃ DRIVING
    │   ├── test_orchestrator.py  ┃ DRIVING
    │   └── test_sector_bands_integration.py  ┃ DRIVING
    ├── pre_selection/
    │   ├── __init__.py  ┃ DRIVING
    │   └── test_pipeline.py  ┃ DRIVING
    ├── preprocessing/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_cs_reexports.py  ┃ DRIVING
    │   ├── test_cs_transformers.py  ┃ DRIVING
    │   ├── test_delisting.py  ┃ DRIVING
    │   ├── test_imputation.py  ┃ DRIVING
    │   ├── test_outliers.py  ┃ DRIVING
    │   ├── test_regression_imputer.py  ┃ DRIVING
    │   └── test_validation.py  ┃ DRIVING
    ├── rebalancing/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_config.py  ┃ DRIVING
    │   └── test_rebalancer.py  ┃ DRIVING
    ├── scheduler/
    │   ├── __init__.py  ┃ DRIVING
    │   └── test_shell_wrappers.py  ┃ DRIVING
    ├── scoring/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_config.py  ┃ DRIVING
    │   ├── test_factory.py  ┃ DRIVING
    │   └── test_information_ratio.py  ┃ DRIVING
    ├── synthetic/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_config.py  ┃ DRIVING
    │   └── test_factory.py  ┃ DRIVING
    ├── test_check_branch_coverage.py  ┃ DRIVING
    ├── test_properties.py  ┃ DRIVING
    ├── tuning/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_config.py  ┃ DRIVING
    │   └── test_factory.py  ┃ DRIVING
    ├── uncertainty_set/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_factory.py  ┃ DRIVING
    │   └── test_robust_integration.py  ┃ DRIVING
    ├── universe/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_config.py  ┃ DRIVING
    │   ├── test_factory.py  ┃ DRIVING
    │   └── test_screener.py  ┃ DRIVING
    ├── validation/
    │   ├── __init__.py  ┃ DRIVING
    │   ├── test_config.py  ┃ DRIVING
    │   ├── test_factory.py  ┃ DRIVING
    │   ├── test_forecast.py  ┃ DRIVING
    │   ├── test_metadata_routing.py  ┃ DRIVING
    │   ├── test_online_forecast.py  ┃ DRIVING
    │   └── test_regime_validation.py  ┃ DRIVING
    └── views/
        ├── __init__.py  ┃ DRIVING
        ├── test_config.py  ┃ DRIVING
        ├── test_factory.py  ┃ DRIVING
        └── test_uncertainty.py  ┃ DRIVING
```

### 8.2 I file

Il context top-level non contiene compute di dominio: è l'insieme di file di supporto (test, wrapper di scheduling, CI helper, orchestrazione container) che stanno *intorno* ai quattro bounded context reali (optimizer/, api→ingestion/, fund/, frontend/) senza mai penetrarne il core. `tests/` è driving-side puro verso `optimizer/` (importa solo la libreria pubblica e, per i test dello scheduler, legge i file shell come testo — non importa `ingestion/app`); non aggiunge dipendenze del dominio verso l'esterno, va semmai nella direzione opposta (verifica che il dominio non dipenda da nulla). `scheduler/` è un adapter driving addizionale, ancora più sottile dell'API: non contiene logica, invoca solo `python -m app.cli <comando>` dentro il container o in locale — la sequenza/gating dei job resta nel service layer di `ingestion/` (`app/services/jobs/scheduler.py`), qui non c'è business logic da violare. `scripts/` è tooling CI puro (analizza `coverage.xml`, non importa il dominio). `docker-compose.yml` è config: cablaggio di processi (db, adminer, scheduler, fund, frontend) e variabili d'ambiente, zero logica applicativa — è l'unico file che 'conosce' tutti e quattro i bounded context insieme, ma solo come nomi di servizio/porte, mai come import di codice. Nessuno di questi file è `core`, quindi la regola core-non-tocca-driven non si applica qui in modo diretto: il context_notes vale piuttosto come garanzia che questi file di supporto restino sul perimetro (driving/config) e non si infilino a scavalcare i port dei singoli bounded context.

| File | Ruolo | Contiene | Si collega a |
|---|---|---|---|
| `docker-compose.yml` | config | Definizione Docker Compose dell'ambiente di runtime locale del monorepo: dichiara tre servizi (`db`, `adminer`, `scheduler`), un volume nominato (`postgres_data`) e una rete condivisa (`optimizer_network`). Non contiene logica applicativa — è puro cablaggio infrastrutturale che stabilisce quali processi esistono, come si raggiungono tra loro e quali porte espongono verso l'host. | `ingestion/Dockerfile`, `ingestion/app/worker.py`, `external:postgres`, `external:adminer`, `external:ollama` |
| `pyproject.toml` | config | File di configurazione del pacchetto Python `portopt` (il contesto delimitato di calcolo dell'ottimizzazione, radice del monorepo): build system, metadati di distribuzione, dipendenze runtime, dipendenze opzionali per test/lint/typecheck/dev, e configurazione di tutti gli strumenti di qualità del codice (ruff, mypy, pyright, coverage, pytest). | `optimizer/__init__.py`, `tests` |
| `scheduler/fetch.sh` | DRIVING | Script shell che forza on-demand l'esecuzione della pipeline di ingestione giornaliera (news → summarize → calibrate) esposta dalla CLI dell'ingestion daemon. | `ingestion/app/cli.py`, `docker-compose.yml`, `external:docker` |
| `scheduler/refetch_all.sh` | DRIVING | Script shell che forza il rebuild completo dei dati di mercato e macro (universo → yfinance 5 anni + FRED) esposto dalla CLI dell'ingestion daemon. | `ingestion/app/cli.py`, `docker-compose.yml`, `external:docker` |
| `scripts/check_branch_coverage.py` | config | Script CLI standalone che legge un report `coverage.xml` in formato Cobertura prodotto dalla suite di test di `optimizer/` e ne verifica due metriche aggregate rispetto a una soglia. | — |
| `tests/__init__.py` | DRIVING | Marcatore di pacchetto Python vuoto per l'intera suite di test di optimizer/. | — |
| `tests/cluster/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/cluster. | — |
| `tests/cluster/test_factory.py` | DRIVING | Test di HierarchicalClusteringConfig e build_hierarchical_clustering (clustering gerarchico per HRP/HERC). | `optimizer/cluster/`, `optimizer/distance/` |
| `tests/conftest.py` | DRIVING | Fixture pytest condivise dall'intera suite: rendimenti sintetici, prezzi derivati, mappatura settoriale e benchmark equal-weight, più l'inserimento di ingestion/ nel sys.path. | `ingestion/app/cli.py` |
| `tests/distance/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/distance. | — |
| `tests/distance/test_factory.py` | DRIVING | Test di DistanceConfig e build_distance su tutti e sei gli stimatori di distanza (Pearson, Kendall, Spearman, Covarianza, Distance Correlation, Mutual Information). | `optimizer/distance/` |
| `tests/factors/__init__.py` | DRIVING | Marcatore di pacchetto vuoto per i test del sottomodulo optimizer/factors. | — |
| `tests/factors/test_alpha.py` | DRIVING | Test di factor_scores_to_expected_returns (ponte fattori→rendimenti attesi in stile CAPM esteso) e di compute_gross_alpha. | `optimizer/factors/` |
| `tests/factors/test_config.py` | DRIVING | Test degli enum e delle dataclass di configurazione dei fattori: gruppi, tipi, standardizzazione, scoring composito, selezione, tilt di regime, soglie di regime, validazione, integrazione. | `optimizer/factors/` |
| `tests/factors/test_construction.py` | DRIVING | Test del calcolo dei singoli fattori grezzi (value, profittabilità, momentum, volatilità, beta, illiquidità, ecc.), dell'allineamento point-in-time e di tre regressioni di valuta/proxy di mercato (issue #289, #290, #294). | `optimizer/factors/` |
| `tests/factors/test_diagnostics.py` | DRIVING | Test dei diagnostici di multicollinearità fra fattori (VIF, PCA) e del controllo di survivorship bias. | `optimizer/factors/` |
| `tests/factors/test_factor_coverage.py` | DRIVING | Test della dataclass diagnostica FactorBuildHealth e della gerarchia di eccezioni FactorCoverageError (issue #249). | `optimizer/factors/`, `optimizer/exceptions.py` |
| `tests/factors/test_forward_return_boundary.py` | DRIVING | Test della semantica esatta della finestra di rendimento forward usata in build_factor_scores_history, per evitare look-ahead bias (issue #295). | `optimizer/factors/` |
| `tests/factors/test_integration.py` | DRIVING | Test dell'integrazione fra fattori e ottimizzazione: costruzione di viste Black-Litterman dai punteggi compositi, vincoli lineari di esposizione fattoriale, e compute_net_alpha al netto dei costi di turnover. | `optimizer/factors/`, `optimizer/optimization/` |
| `tests/factors/test_mimicking.py` | DRIVING | Test della costruzione di portafogli long-short che mimano un fattore (factor-mimicking portfolios), della correlazione incrociata fra fattori e della variante beta-neutrale (issue #101). | `optimizer/factors/` |
| `tests/factors/test_ml_scoring.py` | DRIVING | Test dello scoring composito basato su ML (ridge CV e gradient boosting) come alternativa allo scoring equal/IC-weighted. | `optimizer/factors/` |
| `tests/factors/test_oos_validation.py` | DRIVING | Test della validazione out-of-sample dei fattori a blocchi rolling e Combinatorial Purged CV (CPCV), incluso il conteggio dei fold e il purging (issue #100, #239). | `optimizer/factors/`, `optimizer/validation/` |
| `tests/factors/test_pit_history.py` | DRIVING | Test dell'allineamento point-in-time su storici multi-periodo di fondamentali (issue #245), a complemento di test_construction.py che copre il caso a singolo snapshot. | `optimizer/factors/` |
| `tests/factors/test_quintile.py` | DRIVING | Test dell'analisi spread per quintili/decili dei rendimenti condizionati al punteggio fattoriale. | `optimizer/factors/` |
| `tests/factors/test_regime.py` | DRIVING | Test della classificazione di regime macro e dei tilt fattoriali condizionati al regime, incluso lo score composito multi-indicatore, l'override da disoccupazione, il rilevamento di disaccordo fra due classificatori, e i tilt vincolati (issue #79, #237, #239, #240, #241, #250, #279). | `optimizer/factors/` |
| `tests/factors/test_scoring.py` | DRIVING | Test dello scoring composito equal-weight, IC-weighted e ICIR-weighted a partire dai punteggi di gruppo fattoriali. | `optimizer/factors/` |
| `tests/factors/test_sector_bands.py` | DRIVING | Test della matrice SECTOR_REGIME_BANDS e di resolve_sector_bands: bande di peso (floor, cap) per settore condizionate al regime macro. | `optimizer/factors/` |
| `tests/factors/test_selection.py` | DRIVING | Test della selezione titoli da punteggio composito: conteggio fisso, quantile, zona-cuscinetto (buffer) per l'isteresi di turnover, e bilanciamento settoriale. | `optimizer/factors/` |
| `tests/factors/test_standardization.py` | DRIVING | Test di winsorizzazione, standardizzazione z-score/rank-normal, neutralizzazione settoriale e ortogonalizzazione dei fattori. | `optimizer/factors/` |
| `tests/factors/test_validation.py` | DRIVING | Test della validazione statistica dei fattori: IC mensile, IC series, t-stat di Newey-West, correzione Benjamini-Hochberg per test multipli, e il report di validazione aggregato. | `optimizer/factors/` |
| `tests/fx/__init__.py` | DRIVING | Marcatore di pacchetto vuoto per i test del sottomodulo optimizer/fx. | — |
| `tests/fx/test_converter.py` | DRIVING | Test di FxPriceConverter: conversione di prezzi locali multi-valuta in una valuta base comune. | `optimizer/fx/` |
| `tests/fx/test_decomposition.py` | DRIVING | Test di decompose_fx_returns: scomposizione del rendimento totale in componente locale, componente FX e termine incrociato. | `optimizer/fx/` |
| `tests/fx/test_pipeline_fx_integration.py` | DRIVING | Test di integrazione della conversione FX nella pipeline end-to-end (run_full_pipeline/run_full_pipeline_with_selection). | `optimizer/fx/`, `optimizer/pipeline/` |
| `tests/fx/test_rates.py` | DRIVING | Test delle utility sui tassi di cambio: costruzione del ticker di coppia FX, tasso incrociato, allineamento date. | `optimizer/fx/` |
| `tests/linear_model/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/linear_model. | — |
| `tests/linear_model/test_factory.py` | DRIVING | Test di CSLinearRegressionConfig e build_cs_linear_regression (regressione lineare cross-sezionale, es. per stimare beta o esporre fattori). | `optimizer/linear_model/` |
| `tests/moments/__init__.py` | DRIVING | Marcatore di pacchetto vuoto per i test del sottomodulo optimizer/moments. | — |
| `tests/moments/test_config.py` | DRIVING | Test degli enum e della dataclass MomentEstimationConfig per la stima di media, covarianza e shrinkage. | `optimizer/moments/` |
| `tests/moments/test_factory.py` | DRIVING | Test del dispatch delle factory per stimatori di media (build_mu_estimator), covarianza (build_cov_estimator) e prior (build_prior). | `optimizer/moments/` |
| `tests/moments/test_partial_fit.py` | DRIVING | Test di equivalenza round-trip fra due partial_fit consecutivi e un fit unico sul dato concatenato, per gli stimatori exponential-weighted (EWMu, EWCovariance, EWVariance, RegimeAdjustedEW*). | `optimizer/moments/` |
| `tests/moments/test_scaling.py` | DRIVING | Test dello scaling log-normale dei momenti (media e covarianza) su un orizzonte multi-periodo, sia col metodo esatto che con quello lineare (delta-method). | `optimizer/moments/` |
| `tests/moments/test_variance.py` | DRIVING | Test del dispatch e dei preset per gli stimatori di varianza (empirica, EW, regime-adjusted EW). | `optimizer/moments/` |
| `tests/online/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/online. | — |
| `tests/online/test_online_predict_matches_loop.py` | DRIVING | Test di run_online_predict/run_online_score e delle ricerche a griglia online (OnlineGridSearch/OnlineRandomizedSearch) contro un ciclo di warmup manuale. | `optimizer/online/` |
| `tests/online/test_partial_fit_smoke.py` | DRIVING | Smoke test di partial_fit a due batch su MeanRisk con prior EW e regime-adjusted EW. | `optimizer/online/`, `optimizer/moments/` |
| `tests/online/test_pipeline_rejected.py` | DRIVING | Test che il wrapper online rigetti esplicitamente un sklearn Pipeline al posto di uno stimatore con partial_fit nativo. | `optimizer/online/`, `optimizer/exceptions.py` |
| `tests/optimization/__init__.py` | DRIVING | Marcatore di pacchetto vuoto per i test del sottomodulo optimizer/optimization. | — |
| `tests/optimization/test_benchmark_tracker.py` | DRIVING | Test di BenchmarkTrackerConfig e build_benchmark_tracker (minimizzazione della tracking error verso un benchmark). | `optimizer/optimization/` |
| `tests/optimization/test_config.py` | DRIVING | Test degli enum (ObjectiveFunctionType, RiskMeasureType a 15 membri, RatioMeasureType a 19 membri) e della dataclass MeanRiskConfig. | `optimizer/optimization/` |
| `tests/optimization/test_dr_cvar.py` | DRIVING | Test di DRCVaRConfig e build_dr_cvar (CVaR distributionally robust con raggio di Wasserstein). | `optimizer/optimization/` |
| `tests/optimization/test_factory.py` | DRIVING | Test della factory principale build_mean_risk e di build_sector_constraints. | `optimizer/optimization/` |
| `tests/optimization/test_herc.py` | DRIVING | Test di HERCConfig e build_herc (Hierarchical Equal Risk Contribution). | `optimizer/optimization/`, `optimizer/cluster/`, `optimizer/distance/` |
| `tests/optimization/test_hrp.py` | DRIVING | Test di HRPConfig e build_hrp (Hierarchical Risk Parity). | `optimizer/optimization/`, `optimizer/cluster/`, `optimizer/distance/` |
| `tests/optimization/test_max_diversification.py` | DRIVING | Test di MaxDiversificationConfig e build_max_diversification. | `optimizer/optimization/` |
| `tests/optimization/test_naive.py` | DRIVING | Test delle factory per gli ottimizzatori naive/benchmark: EqualWeighted, InverseVolatility, Random. | `optimizer/optimization/` |
| `tests/optimization/test_nco.py` | DRIVING | Test di NCOConfig e build_nco (Nested Clusters Optimization). | `optimizer/optimization/`, `optimizer/cluster/`, `optimizer/distance/` |
| `tests/optimization/test_regime_blended_mean_risk.py` | DRIVING | Test di build_regime_blended_mean_risk e RegimeBlendedMeanRiskConfig: pipeline di ottimizzazione mean-risk che miscela le stime per stato di regime. | `optimizer/optimization/`, `optimizer/factors/` |
| `tests/optimization/test_region_constraints.py` | DRIVING | Test di build_region_linear_constraints e sanitize_group_token: vincoli lineari per regione geografica (issue #533). | `optimizer/optimization/` |
| `tests/optimization/test_risk_budgeting.py` | DRIVING | Test di RiskBudgetingConfig e build_risk_budgeting (equal/custom risk contribution). | `optimizer/optimization/`, `optimizer/moments/` |
| `tests/optimization/test_robust.py` | DRIVING | Test di RobustMeanRiskConfig e build_robust_mean_risk (mean-risk robusto con insiemi di incertezza su media e covarianza). | `optimizer/optimization/`, `optimizer/uncertainty_set/` |
| `tests/optimization/test_schur.py` | DRIVING | Test di SchurComplementaryConfig e build_schur_complementary (interpolazione fra HRP e minima varianza via complemento di Schur). | `optimizer/optimization/`, `optimizer/cluster/`, `optimizer/distance/` |
| `tests/optimization/test_stacking.py` | DRIVING | Test di StackingConfig e build_stacking (ensemble stacking di più ottimizzatori di base). | `optimizer/optimization/` |
| `tests/pipeline/__init__.py` | DRIVING | Marcatore di pacchetto vuoto per i test del sottomodulo optimizer/pipeline. | — |
| `tests/pipeline/test_builder.py` | DRIVING | Test di build_portfolio_pipeline: assemblaggio di una sklearn Pipeline con step di pre-selezione, validazione e ottimizzatore finale. | `optimizer/pipeline/`, `optimizer/pre_selection/`, `optimizer/optimization/` |
| `tests/pipeline/test_delisting_integration.py` | DRIVING | Test di regressione per il fix del survivorship bias in run_full_pipeline quando alcuni ticker sono delisted a metà serie (issue #274). | `optimizer/pipeline/`, `optimizer/preprocessing/` |
| `tests/pipeline/test_orchestrator.py` | DRIVING | Test delle funzioni di orchestrazione della pipeline: optimize, backtest, tune_and_optimize, run_full_pipeline(_with_selection), compute_net_backtest_returns. | `optimizer/pipeline/`, `optimizer/factors/`, `optimizer/fx/`, `optimizer/optimization/`, `optimizer/rebalancing/`, `optimizer/validation/` |
| `tests/pipeline/test_sector_bands_integration.py` | DRIVING | Test di integrazione dell'iniezione di bande settoriali condizionate al regime nell'orchestratore di pipeline, con dati in memoria senza DB. | `optimizer/pipeline/`, `optimizer/factors/` |
| `tests/pre_selection/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/pre_selection. | — |
| `tests/pre_selection/test_pipeline.py` | DRIVING | Test di PreSelectionConfig e build_preselection_pipeline: filtro di rendimenti estremi, winsorizing, correlazione e top-k prima dell'ottimizzazione. | `optimizer/pre_selection/` |
| `tests/preprocessing/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/preprocessing. | — |
| `tests/preprocessing/test_cs_reexports.py` | DRIVING | Smoke test di identità per i re-export dei trasformatori cross-sezionali skfolio. | `optimizer/preprocessing/` |
| `tests/preprocessing/test_cs_transformers.py` | DRIVING | Test comportamentali dei trasformatori cross-sezionali re-esportati (Gaussian/Percentile rank scaler, standard scaler, tanh shrinker, winsorizer). | `optimizer/preprocessing/` |
| `tests/preprocessing/test_delisting.py` | DRIVING | Test di apply_delisting_returns: guardia contro il survivorship bias sostituendo l'ultimo rendimento valido con il rendimento di delisting dichiarato. | `optimizer/preprocessing/` |
| `tests/preprocessing/test_imputation.py` | DRIVING | Test di SectorImputer: imputazione dei rendimenti mancanti con la media del settore di appartenenza. | `optimizer/preprocessing/` |
| `tests/preprocessing/test_outliers.py` | DRIVING | Test di OutlierTreater: trattamento degli outlier basato su z-score con soglie di winsorizing e rimozione distinte. | `optimizer/preprocessing/` |
| `tests/preprocessing/test_regression_imputer.py` | DRIVING | Test di RegressionImputer: imputazione dei rendimenti mancanti tramite regressione sui titoli correlati vicini. | `optimizer/preprocessing/` |
| `tests/preprocessing/test_validation.py` | DRIVING | Test di DataValidator: sanificazione dei rendimenti grezzi (inf→NaN, rendimenti estremi→NaN) prima di ogni stima. | `optimizer/preprocessing/` |
| `tests/rebalancing/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/rebalancing. | — |
| `tests/rebalancing/test_config.py` | DRIVING | Test degli enum e delle dataclass di configurazione del ribilanciamento: frequenza calendariale, tipo di soglia, giorni di trading per frequenza, config calendariale/soglia/ibrida. | `optimizer/rebalancing/` |
| `tests/rebalancing/test_rebalancer.py` | DRIVING | Test della logica di ribilanciamento: deriva dei pesi, turnover, costo di ribilanciamento, e le regole should_rebalance/should_rebalance_hybrid. | `optimizer/rebalancing/` |
| `tests/scheduler/__init__.py` | DRIVING | Docstring di pacchetto per i test strutturali degli script scheduler/*.sh. | — |
| `tests/scheduler/test_shell_wrappers.py` | DRIVING | Test strutturali del contratto fra gli script scheduler/fetch.sh e scheduler/refetch_all.sh e la CLI Typer del demone di ingestion (ingestion/app/cli.py), dopo la rimozione del layer HTTP. | `scheduler/fetch.sh`, `scheduler/refetch_all.sh`, `ingestion/app/cli.py` |
| `tests/scoring/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/scoring. | — |
| `tests/scoring/test_config.py` | DRIVING | Test di ScorerConfig: configurazione dello scorer usato in cross-validation e tuning (Sharpe, Sortino, Calmar, CVaR ratio, custom). | `optimizer/scoring/` |
| `tests/scoring/test_factory.py` | DRIVING | Test di build_scorer: dispatch verso lo scorer skfolio corretto per ciascuna misura di rapporto, inclusi scorer custom. | `optimizer/scoring/`, `optimizer/optimization/` |
| `tests/scoring/test_information_ratio.py` | DRIVING | Test dello scorer Information Ratio, che richiede rendimenti di benchmark espliciti (issue #21). | `optimizer/scoring/` |
| `tests/synthetic/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/synthetic. | — |
| `tests/synthetic/test_config.py` | DRIVING | Test degli enum e di VineCopulaConfig per la generazione di dati sintetici via vine copula. | `optimizer/synthetic/` |
| `tests/synthetic/test_factory.py` | DRIVING | Test di build_vine_copula e build_synthetic_data. | `optimizer/synthetic/` |
| `tests/test_check_branch_coverage.py` | DRIVING | Test del gate di copertura a livello di branch (scripts/check_branch_coverage.py) invocato in CI sul report coverage.xml. | — |
| `tests/test_properties.py` | DRIVING | Test basati su proprietà (Hypothesis) che codificano invarianti matematiche trasversali a rebalancing e factors: normalizzazione dei pesi, non-negatività, limitatezza del turnover, media/varianza dello z-score, semidefinita-positività della covarianza campionaria, limiti del winsorizing e del rank-percentile. | `optimizer/rebalancing/`, `optimizer/factors/` |
| `tests/tuning/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/tuning. | — |
| `tests/tuning/test_config.py` | DRIVING | Test di GridSearchConfig e RandomizedSearchConfig per la ricerca di iperparametri. | `optimizer/tuning/`, `optimizer/scoring/`, `optimizer/validation/` |
| `tests/tuning/test_factory.py` | DRIVING | Test di build_grid_search_cv e build_randomized_search_cv. | `optimizer/tuning/` |
| `tests/uncertainty_set/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/uncertainty_set. | — |
| `tests/uncertainty_set/test_factory.py` | DRIVING | Test di MuUncertaintySetConfig/CovarianceUncertaintySetConfig e delle relative factory (empirica/bootstrap). | `optimizer/uncertainty_set/` |
| `tests/uncertainty_set/test_robust_integration.py` | DRIVING | Test di integrazione end-to-end: le config di uncertainty_set alimentano correttamente RobustMeanRisk. | `optimizer/uncertainty_set/`, `optimizer/optimization/` |
| `tests/universe/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/universe. | — |
| `tests/universe/test_config.py` | DRIVING | Test di ExchangeRegion, HysteresisConfig e InvestabilityScreenConfig: soglie di investibilità (market cap, ADDV, frequenza di trading, prezzo minimo, storia di quotazione) con isteresi entrata/uscita. | `optimizer/universe/` |
| `tests/universe/test_factory.py` | DRIVING | Test di screen_universe: applicazione end-to-end degli screen di investibilità su fondamentali, prezzi e volumi sintetici. | `optimizer/universe/` |
| `tests/universe/test_screener.py` | DRIVING | Test delle funzioni di screening individuali: ADDV, età di quotazione, frequenza di trading, conteggio bilanci finanziari, soglie percentile per exchange, e apply_investability_screens/apply_screen. | `optimizer/universe/` |
| `tests/validation/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/validation. | — |
| `tests/validation/test_config.py` | DRIVING | Test di WalkForwardConfig, CPCVConfig e MultipleRandomizedCVConfig per la cross-validation su serie temporali finanziarie. | `optimizer/validation/` |
| `tests/validation/test_factory.py` | DRIVING | Test di build_walk_forward, build_cpcv, build_multiple_randomized_cv, compute_optimal_folds e run_cross_val. | `optimizer/validation/` |
| `tests/validation/test_forecast.py` | DRIVING | Test di run_covariance_forecast_evaluation (valutazione offline di più stimatori di covarianza in walk-forward). | `optimizer/validation/`, `optimizer/moments/` |
| `tests/validation/test_metadata_routing.py` | DRIVING | Test end-to-end (marcati slow) dell'instradamento di metadati ausiliari attraverso i fold di cross-validation skfolio: volatilità implicita, pesi di benchmark, pesi precedenti per vincoli di turnover (issue #20). | `optimizer/validation/`, `optimizer/moments/`, `optimizer/optimization/` |
| `tests/validation/test_online_forecast.py` | DRIVING | Test di run_online_covariance_forecast_evaluation: valutazione online (partial_fit) di più stimatori di covarianza. | `optimizer/validation/`, `optimizer/moments/` |
| `tests/validation/test_regime_validation.py` | DRIVING | Test di run_regime_validation: verifica che l'alpha out-of-sample provenga in prevalenza dal regime macro atteso (es. espansione) piuttosto che da rumore. | `optimizer/validation/`, `optimizer/factors/` |
| `tests/views/__init__.py` | DRIVING | Docstring di pacchetto per i test del sottomodulo optimizer/views. | — |
| `tests/views/test_config.py` | DRIVING | Test di ViewUncertaintyMethod, BlackLittermanConfig, EntropyPoolingConfig e OpinionPoolingConfig per l'integrazione di viste soggettive nella stima dei rendimenti attesi. | `optimizer/views/` |
| `tests/views/test_factory.py` | DRIVING | Test di build_black_litterman, build_entropy_pooling e build_opinion_pooling. | `optimizer/views/` |
| `tests/views/test_uncertainty.py` | DRIVING | Test di calibrate_omega_from_track_record: calibrazione empirica della matrice di incertezza delle viste (omega) dallo storico degli errori di previsione. | `optimizer/views/` |

<!-- FINE GENERATO: top-level -->

---

## 9. Decisioni tecnologiche

Una seconda passata di ricerca, adattata ai vincoli fissati sopra, ha sciolto le scelte
concrete di libreria. Il ribaltamento principale rispetto alla prima passata: **non adottiamo
LangGraph come orchestratore primario.** Il valore centrale di LangGraph è lo stato a
canali/reducer persistito come checkpoint con scope di thread — che *duplica il contratto
`FundState` che già possediamo* (§6.1). Per quattro agenti sopra un oggetto di stato che
controlliamo noi, un motore a grafo completo per lo più ci combatte contro.

### Orchestratore — scritto a mano sopra `FundState` *(confidenza media)*

Un normale orchestratore Python itera le quattro port `Agent.run(FundState) -> FundState` sopra
l'unico stato condiviso. Nessun DSL da framework a grafo. Massimizza la leggibilità ed evita che
un framework si scontri con il nostro oggetto di stato. Questo *è* il pattern blackboard di
§6.1: l'orchestratore è il mediatore.

### Substrato durevole — LangGraph SOLO per checkpoint e interrupt *(confidenza media)*

L'unica cosa che vale la pena prendere in prestito da LangGraph è il suo **checkpointing
PostgreSQL** collaudato (`PostgresSaver` / `AsyncPostgresSaver`,
`langgraph-checkpoint-postgres`) più **`interrupt()` / `Command(resume=...)`** per la pausa in
attesa di approvazione umana. Per un sistema che un giorno muoverà denaro vero questo è più
affidabile di un pause/resume artigianale, quindi lo adottiamo *soltanto* come substrato
durevole: **si monta `FundState` come unico canale del grafo**, così non c'è duplicazione di
stato — il controllo di flusso scritto a mano resta nostro, LangGraph solo persiste, mette in
pausa e riprende.

**Trappola:** al resume, un nodo interrotto si riesegue dal suo inizio, quindi la logica di nodo
**dev'essere idempotente**. Riguarda direttamente l'agente di esecuzione (§6.4) e il broker.

### Guardrail deterministico — validatori Pydantic v2 *(confidenza alta)*

Nessuna dipendenza da un motore a regole. `@field_validator` (campo singolo) e
`@model_validator(mode='after')` (fra campi: limiti di posizione, esposizione, concentrazione,
più il kill-switch) sono il confine pass/fail. Una violazione **solleva `ValueError` /
`PydanticCustomError`** (→ `ValidationError`); lo stesso confine pulito serve sia il
**pre-check deterministico prima dell'LLM** sia il **gate di interrupt**.

Si usa `ValueError`, **mai `AssertionError`**: quest'ultimo viene saltato sotto `python -O`, il
che disabiliterebbe un guardrail in silenzio. Le librerie di regole dichiarative in JSON
(`business-rules`, `json-logic-py`) sono entrambe manutenute al minimo — al più si usano come
*formato dati serializzabile* per le definizioni dei limiti, mai come dipendenza portante.

### Passaggi deterministico-prima-di-probabilistico *(confidenza alta)*

Le condizioni critiche si verificano **per prime, senza alcuna chiamata LLM**, così un limite
superato o il kill-switch hanno la precedenza deterministica su qualunque instradamento fatto da
un modello. Il guardrail Pydantic qui sopra è il meccanismo; lo schema a condizioni di contesto
di AG2 è il pattern di riferimento.

### LLM self-hosted via Ollama + BAML — solo BAML *(confidenza media)*

BAML resta l'unico confine di output strutturato dietro `ports/llm.py`. Il suo parsing allineato
allo schema produce output tipizzato da modelli locali (Llama 3.1, Mistral, Gemma 2) attraverso
il provider `openai-generic` — che è già il meccanismo in uso in `ingestion/baml_src/`.

**PydanticAI è stato valutato e respinto** come alternativa: è capace (Ollama nativo, JSON
vincolato dalla grammatica), ma adottarlo significherebbe far girare un *secondo* meccanismo di
output strutturato accanto a BAML senza guadagno. Un confine, un meccanismo.

**Avvertenza:** l'imposizione di schema di Ollama garantisce token grammaticalmente validi solo
*al momento della generazione* — un troncamento a metà generazione può comunque produrre JSON
invalido, quindi lo strato di validazione e retry alla port va tenuto.

### Broker — nessuno per default; la port è aperta, l'adapter lo porta l'utente *(confidenza alta)*

**Nessun broker viene distribuito configurato.** Senza broker il prodotto è un ottimizzatore di
portafoglio e un sistema di supporto alle decisioni: universo e segnali, obiettivo e vincoli,
frontiera, backtest, la deliberazione a quattro agenti, il monitoraggio del rischio,
l'attribuzione e lo stress testing girano tutti invariati. Manca l'ultimo metro — piazzare un
ordine — e nient'altro.

Non è una limitazione aggirata: è la postura di default. L'alternativa che abbiamo respinto è un
broker simulato che riporta eseguiti finti — aggiunge macchineria per fingere una capacità che
semplicemente non c'è, e un eseguito che non è mai avvenuto è peggio di nessun eseguito.

Le posizioni quindi NON si leggono da un broker. `domain/portfolio/` le possiede, popolate a
mano o per import CSV (`adapters/portfolio/`). Un broker configurato SINCRONIZZA quello stato
invece di esserne la sorgente — ed è questo che fa comportare ribilanciamento, monitoraggio del
rischio e attribuzione in modo identico con o senza.

Aggiungere un broker è un'azione dell'utente: implementare `ports/broker.py` e registrare
l'adapter. La port deve quindi assorbire semantiche che cambiano da sede a sede — ordini
denominati in valore, conti multivaluta, idempotenza. **L'idempotenza sta sopra la port**, nel
nucleo, così è risolta una volta sola invece di essere ridiscussa da ogni adapter (§12).

#### Implementazione di riferimento — REST di Trading 212

Tenuta come prova che la port è adeguata, non come broker del prodotto.

Trading 212 **espone** un'API REST pubblica ufficiale che piazza ordini veri —
`POST /api/v0/equity/orders/{market,limit,stop,stop_limit}` per azioni ed ETF (l'affermazione
"nessuna API, solo Selenium" è stata confutata). Il paper-first è uno **scambio di base-URL e di
chiave**: demo `https://demo.trading212.com/api/v0` contro live
`https://live.trading212.com/api/v0`. Si costruisce un client **`httpx`** sottile dietro la port
broker, con ambiente e chiave come configurazione; **non** si usano le librerie Selenium
(`pytrading212` è uno scraper di UI fragile).

**Il primato del consiglio è ormai strutturale**, non solo un gate di policy: senza broker
configurato non c'è niente con cui piazzare un ordine. Dove uno È configurato, il guardrail e il
gate umano tengono l'esecuzione spenta finché non la si sblocca deliberatamente.

**Avvertenze sostanziali per il denaro vero:** l'API ordini è in BETA e **non idempotente**
(rischio di ordine duplicato, §12); solo conti Invest e Stocks & Shares ISA (niente SIPP, niente
CFD); solo valuta del conto principale; nessun ordine per valore. Queste avvertenze sono
l'argomento A FAVORE del default qui sopra: i vincoli beta di una singola sede non devono
plasmare il prodotto che riceve ogni utente.

### Trasporto — SSE nativo di FastAPI *(confidenza alta)*

Gli eventi di deliberazione per nodo e l'avanzamento di `portopt` si trasmettono con l'**SSE
integrato** di FastAPI (`fastapi.sse.EventSourceResponse` + `ServerSentEvent`, aggiunto in
FastAPI 0.135.0) — zero dipendenze aggiuntive, con ping di keep-alive a 15s, `Cache-Control:
no-cache` e `X-Accel-Buffering: no` (che neutralizza il buffering di Nginx) già a posto. È
costruito su **`sse-starlette`**: si scende a quello direttamente solo se servono manopole oltre
lo strato nativo. I campi `data/event/id/retry` di `ServerSentEvent` portano eventi tipizzati per
nodo; `Last-Event-ID` dà la riconnessione.

**Nota:** oggi il repo *non ha* alcuno strato HTTP — il demone è APScheduler headless. Questo
backend FastAPI + SSE è tutto nuovo e vive in `fund/app/http/` (§6).

### Elenco concreto dei pacchetti

| Ambito | Si adotta | Note |
|---|---|---|
| Orchestratore | **Python scritto a mano** sopra `FundState` | nessun DSL a grafo; è il mediatore di §6.1 |
| Pausa/ripresa durevole | **LangGraph** `PostgresSaver` + `interrupt()` **soltanto** | `FundState` = canale unico del grafo; nodi idempotenti |
| Output strutturato LLM | **BAML** dietro `ports/llm.py` | PydanticAI respinto; tenere validazione e retry |
| Guardrail | validatori **Pydantic v2** | `ValueError`, non `AssertionError`; librerie di regole JSON solo come dati |
| Calcolo convesso | **CVXPY** + **CLARABEL**, via `skfolio` dentro `portopt` | LP/QP/QCQP/SOCP + cono esponenziale; transitivo attraverso il pin di `skfolio`; il solver è configurazione, non codice; stato non ottimale = fallimento registrato (§4) |
| Calcolo intero misto | **rimandato** — nessun solver MIP installato | serve solo per cardinalità, soglie, lotti; SCIP è la scelta open-source di `skfolio`; si aggiunge come extra opt-in quando quel percorso arriva |
| Broker | **nessuno per default** — `ports/broker.py` aperta, nessun adapter registrato | adapter di riferimento (non un default): `httpx` → REST ufficiale di Trading 212, demo/live = scambio di base-URL e chiave; l'idempotenza sta sopra la port |
| Trasporto | **`fastapi.sse`** (FastAPI ≥ 0.135.0) | `sse-starlette` come ripiego |

*Sensibilità al tempo: l'SSE nativo di FastAPI (0.135.0), `sse-starlette` v3.4.5 e il rilascio
degli ordini live di Trading 212 stanno tutti al cutoff di gennaio 2026 o dopo, e sono stati
stabiliti dalla passata di ricerca — verificare versioni e percorsi di import (`fastapi.sse`
contro `fastapi.responses`) al momento dell'implementazione.*

*Sensibilità al tempo dello stack di calcolo: il default CLARABEL, e i pin `cvxpy-base` /
`clarabel` che lo portano, sono proprietà della release di `skfolio` attualmente installata
contro il floor `skfolio>=0.20.1` di `portopt`. Un default di solver a monte è una scelta, non un
contratto — va ricontrollato a ogni bump di `skfolio`, insieme al fatto che la raccomandazione
mixed-integer sia ancora SCIP.*

---

## 10. Capability delle librerie di riferimento

Questa sezione mappa ciò che `skfolio` e `yfinance` **offrono davvero** contro ciò che
l'architettura assume che offrano. Serve a un fine solo: impedire che una sezione di disegno
prometta una grandezza che nessuna libreria a valle sa produrre.

È l'unica sezione generata a partire dalle Agent Skill invece che dalle specifiche di file, ed è
volutamente tenuta separata dalle decisioni di §9: là si sceglie *cosa adottare*, qui si registra
*cosa quella scelta permette*. Una capability assente non è un difetto da correggere — è un
vincolo che le sezioni di disegno devono rispettare.

<!-- GENERATO: capabilities -->

_Generato da `scriptorium arch --capabilities` il 2026-08-03, una passata per pagina, con le Agent Skill **skfolio** (0.20.1) e **yfinance** (1.3.0) come uniche fonti. Un simbolo è elencato solo se trovato nei file di riferimento della Skill; ciò che nessuna delle due copre è registrato come **gap**, non inventato. La prova di dove ogni simbolo è stato trovato sta in `data/architecture-spec/capability/`._

### 10.1 Sintesi

| Pagina | Sezione | skfolio | yfinance | Gap |
|---|---|---:|---:|---:|
| Agente allocation | Fondo | 7 | 1 | 7 |
| Agente execution & ordini | Fondo | 1 | 2 | 4 |
| Agente macro & regimi | Fondo | 5 | 2 | 5 |
| Agente risk | Fondo | 3 | 2 | 4 |
| Attribuzione del rischio | Rischio | 6 | 1 | 10 |
| Backtest & Validazione | Risultati | 5 | 0 | 8 |
| Dashboard | Dashboard | 6 | 3 | 8 |
| Dati alternativi & sentiment | Segnali avanzati | 4 | 7 | 8 |
| Deliberazione del fondo | Fondo | 0 | 0 | 5 |
| Factor timing & rotazione | Segnali avanzati | 6 | 3 | 6 |
| Gate di approvazione umana | Approvazioni | 0 | 0 | 5 |
| Guardrail & kill-switch | Approvazioni | 2 | 1 | 4 |
| Guscio e navigazione | Guscio | 0 | 0 | 3 |
| Monitoraggio del rischio | Rischio | 8 | 2 | 7 |
| Obiettivo & Vincoli | Costruisci | 10 | 0 | 7 |
| Regimi di mercato | Segnali avanzati | 7 | 3 | 8 |
| Report & Audit trail | Report | 0 | 0 | 5 |
| Revisione & Ribilanciamento | Risultati | 6 | 2 | 10 |
| Risultati & Frontiera | Risultati | 7 | 1 | 7 |
| Run & Diagnostica solver | Costruisci | 1 | 0 | 4 |
| Segnali & Fattori | Costruisci | 8 | 6 | 6 |
| Stress testing & scenari | Rischio | 3 | 2 | 7 |
| Turbulence & rischio sistemico | Rischio | 4 | 2 | 7 |
| Universo & Dati | Costruisci | 8 | 6 | 7 |
| Views Builder | Costruisci | 4 | 2 | 4 |

Pagine di sola interfaccia, senza calcolo né dati di mercato (4): Deliberazione del fondo, Gate di approvazione umana, Guscio e navigazione, Report & Audit trail

### 10.2 skfolio

111 usi concreti.

| Simbolo | Pagina | Usato per |
|---|---|---|
| `skfolio.optimization.MeanRisk(objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.VARIANCE, min_weights=0.0)` | Agente allocation | Calcolo del portafoglio globale di minima varianza (GMVP) — marcatore ◆ GMVP sul grafico frontiera; con min_weights=0.0 si impone il vincolo di non negatività (vendite allo scoperto non ammesse) descritto in §2.5 |
| `skfolio.optimization.MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.VARIANCE, min_weights=0.0)` | Agente allocation | Ricerca del portafoglio a massimo rapporto rendimento/rischio impiegato come proxy del portafoglio di tangenza (▲) e per il metodo media-varianza selezionabile nella pagina |
| `skfolio.optimization.RiskBudgeting(risk_measure=RiskMeasure.VARIANCE, risk_budget=None)` | Agente allocation | Metodo 'Risk budgeting / Risk parity' selezionabile nella pagina: risk_budget=None produce il portafoglio ERC (contributi di rischio uguali, §3.2); risk_budget custom implementa il risk parity fra classi di attività (§3.5, es. 23/77 azioni/obbligazioni) |
| `skfolio.prior.EmpiricalPrior(mu_estimator=EmpiricalMu(), covariance_estimator=EmpiricalCovariance() | LedoitWolf())` | Agente allocation | Stima di Σ (sempre richiesta) e μ (richiesta solo per il ramo media-varianza, assente/non usata per risk budgeting come mostrato nel pannello 'Parametri del metodo') a partire dai rendimenti storici del titolo |
| `skfolio.preprocessing.prices_to_returns` | Agente allocation | Conversione della serie prezzi (scaricata da yfinance per l'universo del fondo) in rendimenti lineari X, input obbligatorio di ogni prior/optimizer |
| `skfolio.portfolio.Portfolio (oggetto restituito da model.predict(X))` | Agente allocation | Popolare il riquadro 'Statistiche di portafoglio' (mean → E(Rp), standard_deviation → σp, sharpe_ratio → Sharpe) e i pesi/composition della tabella allocazione |
| `skfolio.population.Population.plot_frontier()` | Agente allocation | Base grafica per il pannello 'Posizione sulla frontiera' (scatter σ–r); da arricchire lato frontend con i marcatori GMVP/tangenza/allocazione proposta (vedi gap) |
| `skfolio.moments.EWVariance` | Agente execution & ordini | Stima della volatilità di prezzo σ per titolo (decadimento esponenziale, half_life configurabile) — input al termine di varianza V(x)=σ²Στx_k² di Almgren-Chriss e alla Σ0 del modello di Kyle. Uso marginale: è l'unico punto in cui skfolio tocca il dominio di questa pagina, come stimatore di un parametro di input, non come solver di scheduling. |
| `skfolio.prior.BlackLitterman` | Agente macro & regimi | Combinare l'equilibrio di mercato con la view macro condizionata al regime (rendimenti attesi e covarianze) mostrata nella card 'MacroView corrente' — tab Panoramica |
| `skfolio.moments.EquilibriumMu` | Agente macro & regimi | Prior di equilibrio (CAPM) su cui BlackLitterman applica il delta sintetico in p.p. mostrato nella tabella view rendimenti attesi |
| `skfolio.moments.RegimeAdjustedEWCovariance` | Agente macro & regimi | Badge 'covarianze riviste/invariate': riscala la covarianza EW con un moltiplicatore quando il rischio realizzato diverge da quello predetto — meccanismo skfolio più vicino a 'covarianza condizionata al regime', ma è un moltiplicatore scalare STVU, NON il filtro Markov a 4 stati con matrice di transizione dichiarato nel banner 'Modello di stato attivo' (vedi Gap) |
| `skfolio.prior.SyntheticData` | Agente macro & regimi | Generare gli scenari di stress condizionati (conditioning sui fattori) per popolare la tab 'Stress testing' con uno scenario chiave severo-ma-plausibile |
| `skfolio.distribution.VineCopula` | Agente macro & regimi | Modello di dipendenza multivariato (code pesanti, asimmetria) che alimenta SyntheticData per i campioni dello scenario di stress |
| `skfolio.moments.EmpiricalCovariance / skfolio.moments.LedoitWolf` | Agente risk | Stimare la matrice di covarianza Σ dei rendimenti dei componenti del portafoglio, materia prima per σ di portafoglio e per le formule custom di contributo di Eulero (σ_c(Xi\|X)=c·cov(Xi,X)/σ) e dell'indice di diversificazione DI(X)=σ_p/Σw_iσ_i |
| `skfolio.optimization.MeanRisk` | Agente risk | Valutare σ di portafoglio e ES/CVaR aggregati dell'allocazione ESATTA proposta dall'agente allocation, pinnando i pesi con min_weights=max_weights=w_proposti (nessuna ottimizzazione reale) e leggendo poi Portfolio.standard_deviation / Portfolio.cvar da .predict(X) |
| `skfolio.optimization.RiskBudgeting` | Agente risk | Costruire il portafoglio di riferimento che realizzerebbe esattamente i budget di rischio b_i assegnati, come termine di paragone nella sezione 'Budget di rischio vs realizzato' (il confronto con l'allocazione proposta resta comunque calcolo custom, vedi Gap) |
| `skfolio.optimization.RiskBudgeting (risk_budget=None)` | Attribuzione del rischio | Costruire il portafoglio ERC di confronto ('Confronta con: Portafoglio ERC', preset 'Applica ERC' nella tab Risk budgeting) — §11 dominio. |
| `skfolio.optimization.RiskBudgeting (risk_budget=<array custom>)` | Attribuzione del rischio | Costruire il portafoglio a budget di rischio generale (RB) per 'Confronta con: Portafoglio a budget (RB)' e per i target b_i editabili della tab Risk budgeting — §12 dominio. Limite: risk_measure deve essere una RiskMeasure convessa (STANDARD_DEVIATION/CVAR/CDAR...), quindi copre solo Misura=Volatilità o ES, non VaR/Cornish-Fisher (vedi Gap). |
| `skfolio.prior.EmpiricalPrior(mu_estimator=..., covariance_estimator=LedoitWolf()/EmpiricalCovariance())` | Attribuzione del rischio | Stimare μ_i ed Σ del book al livello aggregazione scelto (Metodo di stima = Analitica): Σ alimenta la formula di contributo di Eulero alla volatilità x_i(Σx)_i/σ(x) (§3), μ_i alimenta RORAC(Xi\|X)=μ_i/contributo. |
| `skfolio.prior.TimeSeriesFactorModel(loading_matrix_estimator=LoadingMatrixRegression())` | Attribuzione del rischio | Ottenere la matrice di loading β_i sui fattori sistematici S, input necessario per calcolare var(E[L\|S]) alla base dell'impatto di rischio RI_ρ(L\|S) nella tab 'Fattori di rischio' (§7) — la formula RI stessa resta custom. |
| `skfolio.prior.SyntheticData(distribution_estimator=VineCopula())` | Attribuzione del rischio | Generare lo scenario campionario simulato quando Metodo di stima = 'Campionaria simulata' nella barra di configurazione. |
| `skfolio.portfolio.Portfolio (.standard_deviation, .cvar, .weights, .composition)` | Attribuzione del rischio | Leggere ρ(X) di book per Misura=Volatilità/ES nel KPI 'Rischio totale', e i pesi correnti per la colonna 'Peso' della tabella di attribuzione. |
| `skfolio.model_selection.WalkForward + cross_val_predict` | Backtest & Validazione | Tab 6 (Validazione OOS): genera lo split holdout/k-fold IS/OOS su cui leggere Sharpe IS e Sharpe OOS per fold nella tabella 'Risultati per fold/split' |
| `skfolio.model_selection.CombinatorialPurgedCV + cross_val_predict` | Backtest & Validazione | Tab 5 (PBO & CSCV): fornisce l'infrastruttura di split combinatoriale train/test (Population di MultiPeriodPortfolio) su cui costruire — a valle, con codice custom — i logit λ_c e la statistica φ/PBO; NON calcola PBO stesso |
| `skfolio.portfolio.Portfolio.sharpe_ratio / skfolio.RatioMeasure.SHARPE_RATIO` | Backtest & Validazione | Sharpe grezzo annuo (Tab 1, Tab 2 input) e Sharpe IS/OOS per fold (Tab 6), letto direttamente dall'oggetto Portfolio/MultiPeriodPortfolio invece di ricalcolare μ̂/σ̂ a mano |
| `skfolio.ExtraRiskMeasure.SKEW / skfolio.ExtraRiskMeasure.KURTOSIS` | Backtest & Validazione | γ̂₃ (asimmetria) e γ̂₄ (curtosi) dei rendimenti della strategia selezionata, input obbligatori del Deflated Sharpe Ratio in Tab 4 |
| `skfolio.population.Population` | Backtest & Validazione | contenitore per le M prove/strategie tentate (Tab 1 'Log delle prove', Tab 3 tabella p-value): Population.summary() dà Sharpe/rendimento per ciascun trial da cui derivare i p-value grezzi p_(i) prima delle correzioni |
| `skfolio.preprocessing.prices_to_returns` | Dashboard | Convertire i prezzi storici degli asset in rendimenti — lineari per le stime media-varianza (Blocco C), log per l'additività richiesta dal cumulo di Performance su più periodi (Blocco B) |
| `skfolio.moments.EmpiricalMu` | Dashboard | Stima del rendimento atteso E(R) per asset da storico, input per E(R) del portafoglio osservato (Blocco C) |
| `skfolio.moments.EmpiricalCovariance` | Dashboard | Matrice di covarianza degli asset, input per Var(R)/σ del portafoglio osservato (Blocco C) |
| `skfolio.distance.PearsonDistance` | Dashboard | Matrice di dipendenza lineare (coefficiente di Pearson, via codependence_) tra i rendimenti degli asset in portafoglio — Blocco F |
| `skfolio.optimization.MeanRisk` | Dashboard | Generare, per livelli di rendimento atteso, i portafogli a varianza minima che tracciano la frontiera efficiente media-varianza rispetto a cui posizionare il portafoglio osservato — Blocco D |
| `skfolio.population.Population` | Dashboard | Raccogliere i portafogli generati lungo la frontiera e tracciarla (plot_frontier) come riferimento per il grafico piano (Var,E) del Blocco D |
| `skfolio.linear_model.CSLinearRegression` | Dati alternativi & sentiment | Pannello 4 (modello caratteristiche condizionate E[R]=a+a1*T+b1'x+b2'*T*x, interazione sentiment×caratteristica) e Pannello 7 (regressione Fama-MacBeth r_i,t = α + γ·1_news + β·Positive + δ·Negative per ciascun ritardo k=0..13) — è vectorized cross-sectional WLS per t, esattamente lo schema richiesto |
| `skfolio.linear_model.CSLinearRegressorWrapper` | Dati alternativi & sentiment | Pannello 8, riga baseline OLS della tabella comparativa metodi (e in generale per wirare un qualunque regressore sklearn — Ridge/Lasso — nel contratto cross-sectional per-periodo, se si vuole una variante regolarizzata elementare) |
| `skfolio.preprocessing.CSWinsorizer` | Dati alternativi & sentiment | Cap degli outlier cross-sezionali sui predittori (le 920 caratteristiche/interazioni del Pannello 8) prima della regressione, per riga t |
| `skfolio.preprocessing.CSStandardScaler` | Dati alternativi & sentiment | Z-score cross-sezionale dei predittori per periodo prima di CSLinearRegression, layer di centratura standard per il pannello ML |
| `skfolio.optimization.MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO)` | Factor timing & rotazione | Build optimal tilt weights ω_t ∝ Σ_Z,t^-1 E_t[Z] (Tab1 tilt combinato per fattore, Tab2 variant weights, Tab5 IS/OOS comparison) — max-Sharpe portfolio over the selected factor/PC set. |
| `skfolio.optimization.MeanRisk(min_weights=..., max_weights=..., groups=..., linear_constraints=...)` | Factor timing & rotazione | Enforce Tab6 breadth guardrail (peso max 150% / min 50% per fattore) directly as optimizer constraints on the factor-tilt weights. |
| `skfolio.prior.TimeSeriesFactorModel(factor_prior_estimator=...)` | Factor timing & rotazione | Regress MSCI World universe asset returns (X) on the 6 factor return series (y) to get factor loadings — base exposure model underlying the Panoramica tilt table and Diagnostica backtest. |
| `skfolio.moments.RegimeAdjustedEWCovariance` | Factor timing & rotazione | Covariance of the factor set (Σ_Z,t) reacting to short-term vol-regime shifts, feeding the tilt-weight formula — partial substitute only, NOT the 2-state/4-state Markov-switching regime model the page actually specifies (see gap). |
| `skfolio.model_selection.WalkForward + cross_val_predict, skfolio.portfolio.Portfolio.sharpe_ratio` | Factor timing & rotazione | Tab5 Sharpe IS/OOS comparison across the 5 timing variants and across rebalance frequencies (mensile/trimestrale/annuale). |
| `skfolio.optimization.EqualWeighted` | Factor timing & rotazione | Baseline 'Pesi statici (nessun timing esplicito)' radio option in Tab2 and the 'Factor investing' row in Tab5 comparison table. |
| `skfolio.moments.LedoitWolf / EmpiricalCovariance / EWCovariance` | Guardrail & kill-switch | Stimare la matrice di covarianza Σ dei componenti (titoli/fattori/sotto-fondi) del book, input necessario per calcolare RC_i = x_i(Σx)_i/√(xᵀΣx) e l'indice di diversificazione DI_ρ mostrati nel tab "Limiti & budget di rischio" |
| `skfolio.optimization.RiskBudgeting` | Guardrail & kill-switch | Riferimento offline per verificare che una specifica di budget b_i sia risolvibile in un portafoglio RB coerente (sistema RC_i(x)=b_i con Σ e x≥0, Σx_i=1) prima di attivarla come regola RG — non usato inline per la validazione real-time del guardrail, che opera su pesi x_i già decisi dall'agente allocation |
| `skfolio.RiskMeasure.CVAR` | Monitoraggio del rischio | CVaR corrente del fondo/agente al livello β selezionato (card B, serie storica C, riga Σ agenti/Fondo in G) — la quantità coerente e subadditiva della formulazione Rockafellar–Uryasev |
| `skfolio.ExtraRiskMeasure.VALUE_AT_RISK` | Monitoraggio del rischio | VaR corrente col metodo Storico (empirico, senza forma distributiva) selezionato nel toggle Metodo VaR (A), card B e serie C |
| `skfolio.ExtraRiskMeasure.DRAWDOWN_AT_RISK` | Monitoraggio del rischio | soglia ζα tracciata sulla curva underwater (blocco D) come drawdown-VaR al livello α CDaR |
| `skfolio.RiskMeasure.CDAR` | Monitoraggio del rischio | CDaR α (riga tabella E, colonna CDaR in G) — Conditional Drawdown at Risk |
| `skfolio.RiskMeasure.MAXIMUM_DRAWDOWN` | Monitoraggio del rischio | MaxDD corrente (riga tabella E, colonna MaxDD in G, denominatore MaxDDRatio in F) |
| `skfolio.RiskMeasure.AVERAGE_DRAWDOWN` | Monitoraggio del rischio | AvDD corrente (riga tabella E) |
| `skfolio.RatioMeasure.CALMAR_RATIO` | Monitoraggio del rischio | MaxDDRatio = R(x)/MaxDD in tabella F (Calmar è esattamente Return/Max drawdown) |
| `skfolio.portfolio.Portfolio` | Monitoraggio del rischio | oggetto risultante da predict(X) su pesi fissati: .returns/.cumulative_returns per la curva D(x,t) e la serie storica VaR/CVaR (C), .annualized_mean per R(x) in F, .cvar/.max_drawdown come proprietà dirette per le card KPI (B) |
| `skfolio.optimization.MeanRisk (objective_function=ObjectiveFunction.MAXIMIZE_UTILITY, risk_measure=RiskMeasure.VARIANCE, risk_aversion=A)` | Obiettivo & Vincoli | Sezione 1 'Utilità media-varianza': U = E(r) - A/2·σ², con A da slider |
| `skfolio.optimization.MeanRisk (objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.CVAR, left_inequality/right_inequality)` | Obiettivo & Vincoli | Sezione 1 'Minimizzazione CVaR (Rockafellar–Uryasev)' con β confidenza; vincolo di rendimento minimo R via disuguaglianza lineare |
| `skfolio.optimization.RiskBudgeting (risk_budget=None)` | Obiettivo & Vincoli | Sezione 1 'Risk parity / ERC' — contributi di rischio uguali di default |
| `skfolio.optimization.RiskBudgeting (risk_budget=array)` | Obiettivo & Vincoli | Sezione 1 'Risk budgeting' + tabella Sezione 3 'Risk budget per asset/classe' (somma 100%) |
| `skfolio.optimization.MaximumDiversification` | Obiettivo & Vincoli | Sezione 1 'Massima diversificazione (MDP)' |
| `skfolio.optimization.MeanRisk (objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.MAXIMUM_DRAWDOWN | AVERAGE_DRAWDOWN | CDAR)` | Obiettivo & Vincoli | Sezione 1 'Reward/risk su drawdown' (MaxDDRatio/AvDDRatio/CDaRRatio) |
| `skfolio.RiskMeasure (VARIANCE, SEMI_VARIANCE, MEAN_ABSOLUTE_DEVIATION, CVAR, CDAR, MAXIMUM_DRAWDOWN, AVERAGE_DRAWDOWN)` | Obiettivo & Vincoli | Sezione 2 dropdown 'Misura di rischio' — 7 delle 11 opzioni del wireframe (Varianza, MAD, CVaR/ES, MaxDD, AvDD, CDaR + Semi-varianza) |
| `skfolio.optimization.MeanRisk (min_weights, max_weights, budget=1.0, groups, linear_constraints, left_inequality/right_inequality)` | Obiettivo & Vincoli | Sezione 3 — budget Σw=1, long-only/bound per titolo o classe, rendimento minimo, rischio massimo a soglia singola |
| `skfolio.optimization.BenchmarkTracker (tracking_error_target=...)` | Obiettivo & Vincoli | Sezione 3 'Tracking error vs benchmark' (fit(X, y=benchmark_returns)); non copre il vincolo di cardinalità K titoli di replica |
| `skfolio.uncertainty_set.{EmpiricalMuUncertaintySet, EmpiricalCovarianceUncertaintySet, BootstrapMuUncertaintySet, BootstrapCovarianceUncertaintySet} via MeanRisk(mu_uncertainty_set_estimator=..., covariance_uncertainty_set_estimator=...)` | Obiettivo & Vincoli | Sezione 4 'Applica ottimizzazione robusta' — insieme separabile M×S (stimatori indipendenti su μ e Σ), limitato a geometria ellissoidale parametrizzata da confidence_level |
| `skfolio.prior.BlackLitterman` | Regimi di mercato | Tab 'Ponte Black-Litterman': traduce direzione/probabilità del regime stimato in view assolute/relative (es. rendimento atteso più basso, correlazioni più alte) passate come stringhe views=[...] al bottone 'Genera view Black-Litterman da questo regime', propagate a Views Builder. |
| `skfolio.prior.TimeSeriesFactorModel` | Regimi di mercato | Bridging opzionale fattoriale: incapsula BlackLitterman con views sui fattori momentum/value/size (proxy ETF tipo MTUM/VLUE) per la parte 'Momentum & Value' del ponte BL, coerente col pattern 'Black-Litterman Factor Model'. |
| `skfolio.moments.EmpiricalMu` | Regimi di mercato | Tab 'Momenti per regime': media dei rendimenti calcolata separatamente sul sottocampione di date assegnate a ciascuno stato stimato esternamente (regime bucketing fatto fuori da skfolio). |
| `skfolio.moments.EmpiricalCovariance` | Regimi di mercato | Tab 'Momenti per regime': matrice di covarianza/correlazione per asset condizionata al sottocampione di regime, alimenta l'heatmap correlazione e l'indicatore sign-switching. |
| `skfolio.moments.RegimeAdjustedEWCovariance` | Regimi di mercato | Overlay di rischio in tempo reale complementare al filtro Hamilton: rescala la covarianza EW quando il rischio realizzato diverge da quello atteso (STVU) — indicatore di mercato dello shift di regime, non sostituisce la stima Markov-switching ma la affianca (es. per il badge affidabilità/volatilità). |
| `skfolio.moments.RegimeAdjustedEWVariance` | Regimi di mercato | Stessa logica STVU applicata alla sola volatilità per asset, utile come input rapido alla card 'Affidabilità segnale' o alla tab Momentum & Value (scala di volatilità del WML). |
| `skfolio.distance.PearsonDistance` | Regimi di mercato | Tab 'Correlazione di eccedenza': calcola la matrice di correlazione empirica sul sottocampione filtrato per soglia θ (filtraggio fatto in pandas, non da skfolio) per confrontarla con quella del modello a regimi. |
| `skfolio.optimization.MeanRisk` | Revisione & Ribilanciamento | Tab2 Smith (1967): ricostruire frontiera efficiente Var(X)-E(X) in t e in t+Δt (variando objective_function/risk_measure o vincoli di rendimento) per collocare z*_t e individuare i punti A/B/C rispetto alla nuova frontiera |
| `skfolio.population.Population.plot_frontier` | Revisione & Ribilanciamento | Tab2 Smith: grafico scatter frontiera t (tratteggiata) vs frontiera t+Δt (piena) nel piano Var(X)-E(X) |
| `skfolio.model_selection.WalkForward` | Revisione & Ribilanciamento | Tab5: backtest rolling del portafoglio Markowitz corrente su periodo/universo selezionato, base per confronto Sharpe lordo/netto |
| `skfolio.model_selection.cross_val_predict` | Revisione & Ribilanciamento | Tab5: generare il MultiPeriodPortfolio dal walk-forward per calcolare sharpe_ratio lordo e netto |
| `skfolio.optimization.MeanRisk(transaction_costs=...)` | Revisione & Ribilanciamento | Tab5: applicare costi di transazione nel MeanRisk usato dentro il walk-forward per ottenere lo Sharpe 'netto costi' da confrontare con quello lordo (transaction_costs=0) |
| `skfolio.portfolio.Portfolio.sharpe_ratio` | Revisione & Ribilanciamento | Tab5: valore Sharpe lordo/netto per strategia (Markowitz corrente, statico, dinamico) nella tabella comparativa |
| `skfolio.optimization.MeanRisk(objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.VARIANCE, min_weights=0.0)` | Risultati & Frontiera | GMVP long-only: card 'Portafogli chiave' → GMVP, estremo inferiore dell'arco long-only, punto di reset dopo Applica in modalità long-only. |
| `skfolio.optimization.MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.STANDARD_DEVIATION, min_weights=0.0)` | Risultati & Frontiera | Portafoglio di tangenza (max Sharpe) long-only: card 'Tangenza', punto ★ sul grafico, estremo superiore per la CAL. |
| `skfolio.optimization.MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_UTILITY, risk_measure=RiskMeasure.VARIANCE, risk_aversion=λ)` | Risultati & Frontiera | Tracciamento numerico dei punti intermedi della frontiera long-only (sweep su una griglia di risk_aversion λ, un fit per punto) per disegnare l'arco tra GMVP e titolo di massimo rendimento atteso. |
| `skfolio.optimization.MaximumDiversification` | Risultati & Frontiera | Portafoglio più diversificato (MDP) mostrato nel tab 'Diversificazione', con relativi E(R)/σ. |
| `skfolio.moments.EmpiricalMu / skfolio.moments.EmpiricalCovariance (via EmpiricalPrior di default)` | Risultati & Frontiera | Stima campionaria di r (rendimenti attesi) e V (matrice varianze-covarianze) sui rendimenti storici, base per tutti i solve sopra e per il tab 'Dati sorgente' (metodo di stima 'campionaria'). |
| `skfolio.portfolio.Portfolio (proprietà mean, standard_deviation, variance, sharpe_ratio, weights, composition)` | Risultati & Frontiera | Popolare le card GMVP/Tangenza/Corrente (E(R), σ, Sharpe) e la tabella 'Pesi ottimi' (peso x_i per titolo). |
| `skfolio.population.Population (metodo plot_frontier)` | Risultati & Frontiera | Assemblare i portafogli generati dallo sweep MAXIMIZE_UTILITY in una collezione per il rendering della curva sul piano rischio-rendimento. |
| `skfolio.optimization.MeanRisk` | Run & Diagnostica solver | Motore che risolve il problema di selezione Markowitz media-varianza (QP) e le sue estensioni SOCP (es. vincolo su deviazione standard/rischio di perdita) descritte in §3 — è il solver dietro il ramo 'caso convesso' della pagina (badge classe problema, p*, ecc.), non un componente UI ma il calcolo a monte del run |
| `skfolio.prior.TimeSeriesFactorModel` | Segnali & Fattori | Modello fattoriale economico (Mkt/SMB/HML/PR1YR) scheda 1: fit(X=rendimenti titoli, y=rendimenti fattori) produce B (esposizioni) e Σ = BΣ_fB'+Σ_e |
| `skfolio.prior.LoadingMatrixRegression` | Segnali & Fattori | Stima OLS delle esposizioni β per titolo (loading_matrix_estimator passato a TimeSeriesFactorModel), scheda 1 tabella β Mkt/SMB/HML |
| `skfolio.moments.EmpiricalCovariance / LedoitWolf` | Segnali & Fattori | Covarianza fattoriale Σ_f (pannello 'Matrice Σ_f') tramite covariance_estimator del factor_prior_estimator interno a TimeSeriesFactorModel |
| `skfolio.linear_model.CSLinearRegression` | Segnali & Fattori | Neutralizzazione cross-sezionale dell'alpha per periodo (benchmark, industry, fattori non previsti): regressione alpha ~ β_benchmark + dummy industry, residuo = alpha neutralizzato — scheda 4 |
| `skfolio.preprocessing.CSWinsorizer` | Segnali & Fattori | Trimming/winsorizzazione dell'alpha oltre soglia 3× scala (scheda 4 'Trimming') |
| `skfolio.preprocessing.CSStandardScaler / CSGaussianRankScaler` | Segnali & Fattori | Scaling dell'alpha alla scala naturale Std{α} derivata da IC×volatilità residua (scheda 4, riquadro 'Alpha Analysis') |
| `skfolio.preprocessing.CSPercentileRankScaler` | Segnali & Fattori | Percentile intra-settore per soglie relative in screening fondamentale (scheda 2, switch 'soglie relative al settore') e ranking composito per liste buy/hold/sell (scheda 5) |
| `skfolio.optimization.EqualWeighted` | Segnali & Fattori | Pesatura equipesata della buy list nello screener (scheda 5, opzione 'Equipesata') |
| `skfolio.prior.EmpiricalPrior` | Stress testing & scenari | Stima di μ e Cov dei fattori sistematici dallo storico (Fondamenti §5-6), base per calcolare Maha(r) e la forma dell'ellissoide Ell_k. |
| `skfolio.moments.LedoitWolf` | Stress testing & scenari | Covarianza shrinkage dei fattori, per tenere la matrice ben condizionata/invertibile — copre lo stato di errore "matrice di covarianza dei fattori non disponibile/singolare" della pagina. |
| `skfolio.prior.SyntheticData (con skfolio.distribution.VineCopula, sample_args=dict(conditioning={...}))` | Stress testing & scenari | Motore per il trattamento (iv) "distribuzione condizionata ai fattori fissati" nel pannello A (r_D) — genera scenari sistematici condizionati ai fattori fissati, esattamente ciò che l'interfaccia descrive come "disponibile solo se il motore di calcolo la supporta". |
| `skfolio.preprocessing.prices_to_returns` | Turbulence & rischio sistemico | Converte pannello prezzi (da yfinance) in rendimenti lineari X, input comune a tutti i pannelli B-H |
| `skfolio.moments.EmpiricalMu` | Turbulence & rischio sistemico | Media multivariata μ dei rendimenti, termine (y_t-μ) nella distanza di Mahalanobis d_t |
| `skfolio.moments.EmpiricalCovariance` | Turbulence & rischio sistemico | Matrice di covarianza Σ (finestra 500gg, controllo A) per d_t, absorption ratio, rango effettivo |
| `skfolio.moments.EWCovariance` | Turbulence & rischio sistemico | Varianze con pesatura esponenziale (parametro half_life = emivita, controllo A) usate nel calcolo di AR |
| `skfolio.moments.EmpiricalCovariance` | Universo & Dati | Scheda 4, radio 'Campionaria (statica)' — matrice S=XX'/n usata come baseline e come input al badge di condizionamento diagnostico |
| `skfolio.moments.EWCovariance` | Universo & Dati | Scheda 4, radio 'EWMA (RiskMetrics)' — covarianza condizionale pesata esponenzialmente; param half_life ricavato dal λ unico impostato in scheda 2 (K=ln(Υ_L)/ln(λ) va convertito a monte, half_life=ln(0.5)/ln(λ)) |
| `skfolio.moments.EWMu` | Universo & Dati | Scheda 2, radio 'Pesi esponenziali (EWMA)' applicato alla stima del rendimento medio atteso mostrato in diagnostica |
| `skfolio.moments.EmpiricalMu` | Universo & Dati | Riquadro diagnostico 'Rendimento atteso medio universo' quando ponderazione = SMA (pesi uguali); base per SE=SD/√n e IC 95% |
| `skfolio.moments.LedoitWolf` | Universo & Dati | Scheda 4, blocco 'Shrinkage della matrice di covarianza' → radio 'Lineare (identità)', intensità ρ mostrata come stimata/sola lettura |
| `skfolio.moments.DenoiseCovariance` | Universo & Dati | Scheda 4, radio 'Non lineare (autovalori)' — regolarizzazione individualizzata per autovalore (RMT), alimenta il grafico 'Confronta autovalori: campionari vs contratti' |
| `skfolio.pre_selection.SelectComplete` | Universo & Dati | Scheda 1, azione 'Escludi titoli con < N osservazioni' / badge ⚠ storia corta — filtra titoli senza storico completo nella finestra scelta (match parziale: è binario completo/incompleto, non una soglia N configurabile, vedi Gap) |
| `skfolio.model_selection.online_covariance_forecast_evaluation / CovarianceForecastComparison` | Universo & Dati | Pulsante 'Stima λ ottimo su griglia' — valuta più EWCovariance(half_life=...) su una griglia e confronta la qualità di previsione fuori campione per proporre il λ migliore (match parziale: metrica nativa è qlike/calibration, non l'esatta RMSE_v del documento, vedi Gap) |
| `skfolio.prior.BlackLitterman` | Views Builder | Core engine of the page: combines equilibrium prior with investor views (absolute 'TICKER == value' / relative 'TICKER1 - TICKER2 == value') via tau, producing the posterior ReturnDistribution (r_BL, V_BL) shown in the anteprima tables and used by 'Valida view'/'Salva'. |
| `skfolio.moments.EquilibriumMu` | Views Builder | Reverse-optimization of the market-equilibrium expected return Π* = aVx_M; `risk_aversion` maps directly to the page's `a` parameter (precompiled 2.50). |
| `skfolio.prior.EmpiricalPrior` | Views Builder | Wraps EquilibriumMu + a covariance estimator (e.g. LedoitWolf) as the `prior_estimator` fed into BlackLitterman — supplies the V matrix used both in Π* and in the a-priori σ = √diag(τV) column. |
| `skfolio.prior.ReturnDistribution` | Views Builder | Output object of the fitted prior — its `mu` and `covariance` fields are r_BL and V_BL, exactly the two quantities the 'Rendimenti attesi' and 'Incertezza' tables display and that 'Esporta V_BL (.csv)' downloads. |

### 10.3 yfinance

48 usi concreti.

| Simbolo | Pagina | Usato per |
|---|---|---|
| `yf.download(tickers, period=..., interval="1d", auto_adjust=True)` | Agente allocation | Scaricare lo storico prezzi (OHLCV) dei titoli dell'universo del fondo, da convertire in rendimenti per stimare Σ e μ effettivamente usati dal run dell'agente allocation |
| `yf.download / Ticker.history (colonna Volume)` | Agente execution & ordini | Volume di riferimento V nella legge a radice quadrata di Bouchaud-Farmer-Lillo (Δp∝√(Q/V)) e nel dimensionamento del turnover rispetto alla liquidità media del titolo. |
| `Ticker.fast_info (tenDayAverageVolume, threeMonthAverageVolume, lastPrice)` | Agente execution & ordini | Volume medio giornaliero (proxy di liquidità/profondità di mercato) e prezzo corrente S0 di riferimento per calcolare E atteso e la % del turnover rispetto al volume tipico. |
| `yf.download` | Agente macro & regimi | Serie storiche di prezzo per i tre asset del VAR a regime (large cap, small cap, obbligazioni a lungo termine) usate come input del filtro a 4 stati e per calcolare il rendimento cumulato di mercato R^m_{t-1-J,t-1} della card Momentum |
| `yf.Calendars.economic_events` | Agente macro & regimi | Calendario eventi macro (CPI, Fed) potenzialmente utile come lista pubblicazioni per la tabella 'News recenti' del tab Nowcasting — ma fornisce solo date/eventi in calendario, non la serie storica dei ~200 valori pubblicati necessaria al nowcast (vedi Gap) |
| `yf.download` | Agente risk | Scaricare in blocco le serie storiche di prezzo (adjusted close) di tutti i componenti del portafoglio (es. Azioni USA, Bond EM, Oro) e dei proxy fattoriali (tassi USA, equity globale, credito EM) per costruire la matrice di rendimenti X su cui girano covarianza, VaR, ES e diagnostica code |
| `Ticker.history` | Agente risk | Recuperare/verificare la serie storica di un singolo componente quando serve controllare la lunghezza della finestra dati per il caso limite 'metodo storico con serie corta' (disabilitazione opzione VaR storico se serie insufficiente) |
| `yf.download(tickers, period="2y", interval="1d")` | Attribuzione del rischio | Serie storiche daily dei componenti (titoli) per la finestra 'Storico 2y' / N=504 usata a stimare Σ, μ e le distribuzioni campionarie per VaR/ES/Cornish-Fisher. |
| `yf.download` | Dashboard | Storico prezzi (OHLCV, adjusted close) degli asset del portafoglio osservato — base per rendimenti storici usati in media/varianza/correlazione e per la serie della sparkline di Performance (Blocco B) |
| `Ticker.history` | Dashboard | Alternativa per-ticker allo stesso storico prezzi, con Dividends inclusi in colonna — quando serve granularità/errori per singolo asset invece di download bulk |
| `Ticker.dividends` | Dashboard | Dividendi storici per asset, componente richiesta dalla definizione di rendimento (percentuale/log) usata per il calcolo di Performance totale |
| `yf.Search / yf.Lookup` | Dati alternativi & sentiment | Autocomplete titolo/universo in testata (selettore ticker) |
| `yf.download / Ticker.history` | Dati alternativi & sentiment | Serie OHLCV (prezzo, volume) per overlay su indice/titolo nei Pannelli 1,3,4,5,7 — rendimento cumulato per firma temporale, Vlm_t nel VAR del Pannello 3, serie eventi Pannello 1 |
| `Ticker.shares_full / get_shares_full` | Dati alternativi & sentiment | Storico azioni in circolazione per approssimare la componente TURN (rotazione degli scambi) usata nella scomposizione Baker-Wurgler del Pannello 4 |
| `Ticker.sec_filings` | Dati alternativi & sentiment | Pannello 3, sotto-tab titolo: individuare data di deposito e URL del 10-K dato CIK/ticker (solo metadati, non il testo) |
| `Ticker.news` | Dati alternativi & sentiment | Pannello 7: lista notizie/headline/timestamp per costruire il flag 'esistenza della notizia' (1_news) — non fornisce il punteggio di tono |
| `yf.Calendars(...).ipos` | Dati alternativi & sentiment | Pannello 5, sotto-sezione IPO: calendario storico/prossimo IPO come base per la tabella rendimento 1° giorno per bucket attenzione |
| `Ticker.earnings_history / yf.Calendars(...).earnings` | Dati alternativi & sentiment | Marcatore 'annuncio utili' sulla timeline del Pannello 7 (PEAD) e riferimento agli annunci di utili nel Pannello 4 |
| `yf.download(tickers, period=..., interval='1mo')` | Factor timing & rotazione | Price history for MSCI World universe constituents/factor ETF proxies — base for the 36m/12m/24m momentum state (§2.1 UP/DOWN) and for return series feeding TimeSeriesFactorModel/WalkForward. |
| `Ticker.valuation_measures` | Factor timing & rotazione | Per-company Price/Book, Trailing/Forward P/E — cross-sectional inputs to build the Value factor value-spread metric in Tab4 (P/B selector). |
| `Ticker.dividends` | Factor timing & rotazione | Historical dividend series combined with price to derive trailing dividend yield, a Valuation-category predictor (§9 taxonomy: div yield). |
| `yf.download` | Guardrail & kill-switch | Serie storiche prezzi dei componenti del book, da convertire in rendimenti (skfolio.preprocessing.prices_to_returns) per stimare Σ a monte del calcolo RC_i/DI_ρ |
| `yf.download` | Monitoraggio del rischio | storico OHLCV bulk dei titoli sottostanti (per fondo e per ciascuno dei 4 agenti/sleeve) da cui derivare i rendimenti per periodo grafico 6m/1a/YTD e per il ricalcolo di VaR/CVaR/drawdown |
| `yf.Ticker.history` | Monitoraggio del rischio | recupero per-ticker con start/end espliciti per il periodo '3a' (non presente nell'enum period) e per il range 'Custom' del selettore periodo grafici |
| `yf.download / Ticker.history` | Regimi di mercato | Serie prezzi per gli asset/panieri usati come input al modello di regime (indice di mercato per Hamilton 2-stati; large cap/small cap/obbligazioni proxy ETF per il VAR Guidolin-Timmermann a 4 stati) e per calcolare rendimenti WML, momenti per regime, correlazione di eccedenza. |
| `Ticker.balance_sheet / Ticker.quarterly_balance_sheet` | Regimi di mercato | Approssimazione del book value corrente per un indicatore value/book-to-market a livello di singolo titolo nella tab 'Momentum & Value' (solo snapshot recente, non panel storico point-in-time). |
| `yf.Calendars.economic_events` | Regimi di mercato | Tab 'Nowcasting macro': date del calendario economico prospettico (CPI, Fed, ecc.) per popolare la colonna data della tabella 'news per pubblicazione' e il badge 'pannello sbilanciato'. |
| `yf.download` | Revisione & Ribilanciamento | Tab1: recuperare P(t,1), P(t,2), P(t+Δt,1), P(t+Δt,2) dei due titoli selezionati per popolare il calcolatore verdetto |
| `Ticker.history` | Revisione & Ribilanciamento | Tab1/Tab2: prezzi puntuali per titolo quando serve granularità/parametri extra (pre/post market) oltre al bulk download |
| `yf.download(tickers, start=..., end=..., auto_adjust=True)` | Risultati & Frontiera | Storico prezzi dell'universo (38 titoli) che alimenta prices_to_returns → stima campionaria mu_/covariance_ usata dal solver quando l'utente preme 'Applica' (ricalcolo frontiera); il periodo/T mostrato nel tab 'Dati sorgente' deriva da questa finestra. |
| `yf.download / Ticker.history` | Segnali & Fattori | Serie storiche di prezzo mensili (finestra 36-60 mesi) per rendimenti titoli e fattori, base dell'OLS scheda 1 |
| `Ticker.income_stmt / balance_sheet / cashflow (annual/quarterly)` | Segnali & Fattori | Voci contabili grezze (Net Income, Sales, Total Assets, Total Debt, EBIT, D&A, Interest Expense, Current Assets/Liabilities) da cui derivare ROE/ROA/ROIC/D-E/current-quick-cash ratio/DuPont — scheda 2 |
| `Ticker.valuation_measures` | Segnali & Fattori | Serie storiche Trailing/Forward P/E, Price/Book, EV/EBITDA per i multipli di valutazione — scheda 2 |
| `yf.screen / EquityQuery` | Segnali & Fattori | Costruzione/filtraggio dell'universo iniziale (812→...) per marketcap, sector, peratio, pbratio, evtoebitda, margini — funnel scheda 2 |
| `Ticker.growth_estimates / earnings_estimate / revenue_estimate / analyst_price_targets` | Segnali & Fattori | Input di crescita attesa g_n (consensus) per la costruzione alpha da DDM — scheda 4 |
| `Ticker.fast_info (marketCap)` | Segnali & Fattori | Capitalizzazione di mercato per pesatura 'Per capitalizzazione' dello screener (scheda 5) e per l'equity multiplier nel DuPont |
| `yf.download` | Stress testing & scenari | Storico prezzi dei fattori sistematici tradabili sui mercati (es. tasso di cambio E via ticker valuta tipo "EURUSD=X") da cui derivare i rendimenti/variazioni annuali usati per stimare μ e Cov del fattore. |
| `yf.download` | Stress testing & scenari | Storico prezzi/rendimenti delle posizioni di portafoglio, da abbinare ai rendimenti dei fattori per stimare un modello di esposizione fattoriale (loading) che traduce uno scenario di fattori nell'impatto sul valore del portafoglio. |
| `yf.download` | Turbulence & rischio sistemico | Prezzi OHLCV storici dell'universo attivo (2018-2026), frequenza giornaliera/settimanale/mensile via interval=1d/1wk/1mo |
| `yf.download (ticker benchmark/indice)` | Turbulence & rischio sistemico | Serie rendimento indice di riferimento per overlay in C e drawdown mensile overlay in E |
| `yf.download / Ticker.history` | Universo & Dati | Scarico storico prezzi per l'universo selezionato, per frequenza giornaliera/mensile (interval='1d'/'1mo') e intervallo date scelto in scheda 2 — base per calcolo rendimenti, 'n oss.' per titolo e prima osservazione in scheda 1 |
| `yf.Search / yf.Lookup` | Universo & Dati | Barra 'Cerca titoli' con autocomplete su ticker/nome (Lookup filtra anche per tipo asset per il filtro rapido 'Asset class') |
| `Ticker.isin` | Universo & Dati | Ricerca/visualizzazione ISIN nella barra di ricerca e nella tabella universo (solo lookup ticker→ISIN, non l'inverso, vedi Gap per import CSV via ISIN) |
| `Ticker.info` | Universo & Dati | Colonna 'Settore' in tabella universo scheda 1 (fast_info non espone il settore) |
| `Ticker.fast_info.currency` | Universo & Dati | Filtro rapido 'Valuta' in scheda 1 |
| `EquityQuery + yf.screen` | Universo & Dati | Filtro rapido 'Settore' come screening bulk alternativo/complementare alla ricerca singola (campo 'sector' della classificazione) |
| `Ticker.fast_info (marketCap, shares)` | Views Builder | Per-asset market capitalization to build the benchmark/market weight vector x_M feeding Π* = aVx_M, shown in the 'Peso mkt → peso BL' column. |
| `yf.download` | Views Builder | Historical OHLCV/price series for the investable universe, converted to returns to estimate covariance V (feeds both Π* reverse-optimization and the σ a priori/a posteriori columns). |

### 10.4 Gap — da costruire o procurare (156)

Ciò che nessuna delle due librerie fornisce. Un gap onesto vale più di un simbolo inventato: dichiarare una classe che non esiste manda l'implementazione in un vicolo cieco.

| Pagina | Serve | Perché manca | Cosa ci serve |
|---|---|---|---|
| Agente allocation | Tasso privo di rischio r_f come parametro esplicito dell'ottimizzazione (Sharpe ratio al netto di r_f, capital allocation line, vero portafoglio di tangenza come max (r-rf)/σ) e sua eventuale acquisizione da mercato | Verificato in skfolio: nessun parametro risk_free_rate in MeanRisk, RiskBudgeting, Portfolio o RatioMeasure (grep su tutti i file di references/skfolio — nessun match); RatioMeasure.SHARPE_RATIO è documentato come 'Return / StdDev' senza sottrazione di r_f. Verificato in yfinance: nessun campo o classe dedicata a un tasso privo di rischio in nessuno dei reference file (ticker/price_history/financials/analysis/screener/market_search) | Codice custom: (a) sottrarre r_f dai rendimenti prima di stimare μ (nessun mu_estimator rf-adjusted esiste) o ricalcolare Sharpe/CAL/tangenza a valle con numpy usando l'output di MeanRisk; (b) memorizzare r_f come parametro configurato manualmente nel FundState (coerente col wireframe 'r_f = 2,10%'), eventualmente con proxy di mercato da un ticker Treasury generico (es. yf.download('^IRX')) come fonte dati alternativa, non una feature dedicata |
| Agente allocation | Contributo marginale (∂σ/∂xi) e contributo totale al rischio (xi·∂σ/∂xi) per titolo, colonne della tabella allocazione, con verifica Σ contributi = σp | Verificato in skfolio: references/portfolio.md elenca le proprietà di Portfolio (returns, mean, variance, standard_deviation, sharpe_ratio, sortino_ratio, calmar_ratio, cvar, max_drawdown, weights, composition) — nessuna proprietà di contributo di rischio per singolo asset è documentata, né in RiskBudgeting (references/optimization.md) | Codice custom: calcolare Σx/√(x'Σx) (contributo marginale) e xi·(quel valore) (contributo totale) direttamente da Σ (già disponibile dalla prior) e dai pesi restituiti da Portfolio.weights |
| Agente allocation | Beta del titolo rispetto al portafoglio (βi = SD(Ri)·Corr(Ri,Rp)/SD(Rp)), colonna della tabella allocazione | Verificato in skfolio: nessuna proprietà 'beta' per asset in Portfolio (references/portfolio.md) né altrove nei file di riferimento consultati | Codice custom: calcolare Cov(Ri,Rp)/Var(Rp) per ciascun titolo a partire dai rendimenti (X) e dai rendimenti di portafoglio (Portfolio.returns) |
| Agente allocation | Coefficienti espliciti della soluzione in forma chiusa media-varianza (α=r'V⁻¹r, β=r'V⁻¹e, γ=e'V⁻¹e, x*) se la pagina deve esporli come parte della sostanza di dominio, oltre ai soli pesi ottimi | skfolio risolve MeanRisk numericamente via un solver convesso generico (nessuna formula chiusa α/β/γ/x* documentata in references/optimization.md o priors.md); la libreria produce pesi ottimi, non i coefficienti algebrici intermedi | Codice custom in numpy/scipy che replichi l'algebra lineare (V⁻¹, r'V⁻¹r, ecc.) se questi coefficienti devono comparire nell'interfaccia o nei log di audit |
| Agente allocation | Grafico frontiera con marcatori etichettati GMVP/tangenza/allocazione proposta, tooltip numerici, e fallback 'non disponibile per questo run' quando la tangenza non è calcolabile | skfolio fornisce solo Population.plot_frontier() come scatter generico (references/portfolio.md) senza API per annotare/etichettare marcatori specifici o gestire casi di assenza | Componente frontend custom (es. libreria di charting Angular) che compone i risultati di 3 run MeanRisk separati (GMVP, tangenza, allocazione corrente) con logica di visualizzazione/fallback propria |
| Agente allocation | Messaggi di errore di dominio nello stato 'Errore' ('Σ non definita positiva: soluzione in forma chiusa non disponibile'; 'risoluzione ERC non converge entro le iterazioni consentite') e relativa diagnostica | Nessun parametro di tolleranza/iterazioni o codice di stato del solver documentato per MeanRisk/RiskBudgeting in references/optimization.md; le eccezioni del solver convesso sottostante non sono documentate come messaggi di dominio | Validazione custom pre-solve (controllo autovalori di Σ per la definita-positività) e livello di traduzione delle eccezioni del solver in messaggi utente specifici, con log di esecuzione collegato al pulsante 'Vedi log esecuzione' |
| Agente allocation | Persistenza e audit del FundState: storico run, timestamp di scrittura, confronto run side-by-side, esportazione JSON/CSV grezza, badge di incoerenza su vincoli violati | Estraneo sia a skfolio (libreria di solo calcolo/ottimizzazione, nessuna persistenza di stato applicativo) sia a yfinance (solo dati di mercato) — nessun riferimento in nessuna delle due Skill | Backend custom (DB/tabella run + serializzazione del dataclass FundState) secondo il pattern architetturale blackboard già definito in ARCHITECTURE.md, esterno alle due librerie |
| Agente execution & ordini | Motore di trade-scheduling/market-impact: frontiera E–V di Almgren-Chriss (soluzione iperbolica, half-life 1/κ), lambda/beta di Kyle, Bellman di Bertsimas-Lo (incl. cross-impact multi-asset), strategia a blocchi-e-flusso di Obizhaeva-Wang (resilienza ρ), scheduling non lineare a legge di potenza (integrale di Beltrami, 2F1 ipergeometrica). | skfolio verificato (optimization.md, portfolio.md, model_selection.md, PATTERNS.md, grep esaustivo su impact/execution/kyle/almgren/bertsimas/obizhaeva/bouchaud/VWAP/TWAP/resilien*): zero risultati. skfolio ottimizza pesi di portafoglio (MeanRisk, RiskBudgeting, HRP/HERC/NCO, Stacking...) con al più un costo di transazione lineare scalare (transaction_costs in MeanRisk) — non un problema di scheduling multi-periodo del turnover con impatto permanente/temporaneo/transiente. Il dominio di questa pagina (esecuzione ottima) non è coperto dalla libreria. | Modulo di calcolo custom (numpy/scipy — scipy.special.hyp2f1 per il caso a legge di potenza, programmazione dinamica per Bertsimas-Lo) che implementi le formule chiuse citate nei Fondamenti di dominio; nessuna libreria terza sostituisce questo motore, va scritto ad-hoc nel backend dell'agente execution. |
| Agente execution & ordini | Parametri di impatto calibrati per titolo: coefficiente di impatto permanente γ/λ, temporaneo η, spread bid-ask ε/s, tasso di resilienza ρ dell'order book, profondità di mercato 1/λ, matrice di cross-impact Γ/A fuori diagonale. | yfinance verificato (references/ticker.md, price_history.md, websocket.md, screener.md, market_search.md — grep su bid/ask/spread/order book/depth/tick): nessun campo di bid-ask spread, nessun order-book/Level-2, nessun dato di segno-trade tick-by-tick. yfinance espone solo OHLCV giornaliero/intraday e volumi medi aggregati — dati insufficienti a stimare γ, η, ρ, λ o il cross-impact. | Fonte dati di microstruttura dedicata (es. LOBSTER, Polygon.io trades&quotes, Refinitiv/LSEG Tick History, o feed TCA del broker/prime broker del fondo) più codice di calibrazione custom (regressione della legge a radice quadrata BFL, stima di λ alla Kyle su book reale). |
| Agente execution & ordini | Matrice di cross-impact titolo×titolo (pannello D) — termini fuori diagonale di A/Γ nel modello multi-asset di Bertsimas-Lo/Almgren-Chriss. | Stesso gap dei parametri di impatto: richiede dati di co-movimento tick-level fra order flow di titoli correlati, che né skfolio (nessun modulo di market-impact) né yfinance (nessun dato tick/order-flow) forniscono. | Stessa fonte di microstruttura di cui sopra, con stima della matrice di cross-impact via regressione multivariata su serie di order-flow reali (non derivabile dai soli rendimenti storici usati da skfolio per la covarianza di portafoglio). |
| Agente execution & ordini | Stato condiviso FundState fra i quattro agenti (macro/allocation/risk/execution) e confine architetturale read-only/compute-only sui tool dell'LLM (nessun tool di invio ordini richiamabile), più macchina a stati del workflow di approvazione (Bozza→In verifica guardrail→In attesa→Approvata/Rifiutata/Scaduta). | Né skfolio (libreria di ottimizzazione portafoglio) né yfinance (libreria di dati di mercato) offrono orchestrazione multi-agente, stato condiviso o permissioning dei tool: sono fuori perimetro per entrambe per costruzione. | Framework di orchestrazione agenti custom (es. LangGraph o equivalente per lo shared-state a grafo) più servizio di guardrail deterministico e gate di approvazione, già previsti come pagine separate ('Guardrail & kill-switch', 'Gate di approvazione umana') nell'architettura del fondo. |
| Agente macro & regimi | Filtro a cambio di regime Markov a 4 stati (Guidolin-Timmermann: VAR multi-asset, probabilità filtrata/smoothed, matrice di transizione 4×4, π stazionario, log-verosimiglianza) alla base dell'intera tab 'Regimi & filtro' e del badge regime in Panoramica | Verificato skfolio: nessun modello Markov-switching/HMM discreto. RegimeAdjustedEWCovariance esiste ma è un moltiplicatore scalare STVU (realizzato vs predetto), non un filtro Hamilton/Kim a stati discreti con matrice di transizione stimata e probabilità filtrate/smoothed. Nessuna evidenza in optimization.md, priors.md, moments.md, distance_clustering.md, model_selection.md. yfinance non è pertinente (è solo fonte dati prezzi, non modelli statistici) | Libreria esterna: statsmodels.tsa.regime_switching.markov_autoregression / markov_regression (MarkovAutoregression, MarkovRegression), oppure hmmlearn per HMM gaussiano multivariato; implementazione custom del filtro ricorsivo a 5 passi e dello smoother di Kim per il caso a 4 stati specifico di Guidolin-Timmermann |
| Agente macro & regimi | Nowcasting macro in tempo reale: modello a fattori dinamici su pannello sbilanciato di ~200 serie mensili, filtro di Kalman, calcolo NEWS[i,v_j] per pubblicazione, diffusion index di Stock-Watson (6/12 fattori via componenti principali) — intera tab 'Nowcasting' | Verificato skfolio: TimeSeriesFactorModel è un modello fattoriale su rendimenti asset (fit(X,y) con y=rendimenti fattore già noti), non un dynamic factor model macro con filtro di Kalman su pannello sbilanciato; nessun Kalman filter, nessuna PCA a fattori dinamici, nessun concetto di 'news'/tempestività in moments.md, priors.md, linear_model.md. Verificato yfinance: yf.Calendars.economic_events dà solo un calendario di eventi (CPI, Fed) non la serie storica vintage delle ~200 pubblicazioni mensili; nessun ticker/campo in ticker.md, financials.md, analysis.md porta dati macro (PIL, produzione industriale, occupazione, spread, disoccupazione) | Fonte dati macro: FRED (via fredapi o pandas-datareader) per il pannello di serie mensili (vintage/ALFRED per gestire la pubblicazione asincrona); libreria di modellazione: statsmodels.tsa.statespace.DynamicFactorMQ (o dfm-diffusion custom) per il filtro di Kalman e l'estrazione dei fattori comuni; PCA (scikit-learn) per il diffusion index di Stock-Watson |
| Agente macro & regimi | Premi cross-sectional regime-dipendenti: stato momentum UP/DOWN da R^m_{t-1-J,t-1}, regressione di momentum-crash con beta condizionato allo stato bear (β_B, β_{B,U}), CAPM condizionato con beta tempo-variante su istrumenti macro DIV/DEF/TERM/TB per il value, size premium per i 4 regimi | skfolio.linear_model.CSLinearRegression può stimare regressioni cross-sectional per periodo (utile per la parte di stima beta) ma non fornisce gli istrumenti macro necessari (dividend yield DIV, default spread DEF, term spread TERM, T-bill TB) né la logica di stato/soglia; verificato yfinance: nessun campo per spread di credito, term spread o tasso T-bill aggregato in ticker.md/financials.md/analysis.md/market_search.md — solo prezzi/OHLCV e fondamentali single-name | Fonte dati: FRED per DIV (dividend yield aggregato, es. S&P), DEF (spread BAA-AAA), TERM (10Y-3M), TB (T-bill 3M); codice custom (statsmodels/CSLinearRegression di skfolio) per la definizione di stato UP/DOWN, la regressione momentum-crash con beta condizionato e il CAPM condizionato con beta tempo-variante |
| Agente macro & regimi | Stress test sistematico di Breuer-Jandačka-Rheinberger-Summer: dominio ellissoidale Ell_k basato su distanza di Mahalanobis, minimizzazione del Conditional Expected Profit (CEP) nel dominio per il Maximum Loss, scomposizione Loss Contribution/Maximum Loss Contribution per fattore — tabella 'Scenario chiave' e drill-down LC/MLC nel tab Stress testing | Verificato skfolio: SyntheticData/VineCopula generano scenari campionati/condizionati ma non risolvono il problema di ottimizzazione min CEP(r) su Ell_k né calcolano LC/MLC; RegimeAdjustmentTarget.MAHALANOBIS in RegimeAdjustedEWCovariance usa Mahalanobis solo come target di calibrazione del moltiplicatore di covarianza, non come vincolo di dominio per una ricerca di caso peggiore. Nessuna evidenza in distance_clustering.md (Mahalanobis non è tra i distance estimator elencati) né in optimization.md | Codice custom: ottimizzazione vincolata (scipy.optimize.minimize con vincolo quadratico sull'ellissoide, o cvxpy) che implementa il metodo Breuer et al. sopra covarianza/mu stimati (eventualmente con RegimeAdjustedEWCovariance di skfolio come input); calcolo LC(i,r)/MLC(i) come funzione derivata, non disponibile in nessuna libreria |
| Agente macro & regimi | Contratto architetturale FundState/MacroView: dataclass frozen condivisa fra i 4 agenti, pattern Agent.run(state)->state, checkpoint/audit trail versionato con diff fra versioni | Non è un problema di dati di mercato né di ottimizzazione di portafoglio — nessuna delle due librerie tratta orchestrazione multi-agente o persistenza di stato applicativo; correttamente fuori perimetro di skfolio e yfinance | Codice custom lato backend: dataclass Python (@dataclass(frozen=True)) per FundState/MacroView, meccanismo di checkpoint/versioning (es. tabella DB append-only) e diff strutturato per la tab 'Cronologia & audit' |
| Agente risk | VaR al 95%/99% con tre metodi selezionabili dall'utente: parametrico RiskMetrics (z_ε·σ+μ), storico (empirico), Cornish-Fisher (corretto per asimmetria e curtosi) | Verificato in skfolio (portfolio.md, optimization.md): Portfolio espone solo `cvar` fra le misure di coda, nessuna proprietà `value_at_risk` configurabile per metodo/confidenza; `ExtraRiskMeasure.VALUE_AT_RISK` esiste solo come misura non-convessa per scoring/cross-validation, non come funzione che restituisce il VaR di un portafoglio a pesi fissi con scelta del metodo. Nessuna espansione di Cornish-Fisher documentata. yfinance non fa calcolo di rischio, solo dati di mercato (verificato in tutti i file di references/). | Codice custom: quantili gaussiani (scipy.stats.norm.ppf) per RiskMetrics, quantile empirico ordinato per il metodo storico, espansione di Cornish-Fisher con skew/kurtosis campionari (scipy.stats.skew/kurtosis) per il terzo metodo. |
| Agente risk | Contributi di Eulero per componente (su σ, VaR, ES) di un'allocazione a pesi ESTERNI già fissata, incl. contributo marginale with-without per confronto e stima kernel Nadaraya-Watson (bandwidth di Silverman) del contributo al VaR | Verificato in skfolio: nessuna proprietà per-asset di risk contribution documentata su `skfolio.portfolio.Portfolio` (portfolio.md elenca solo misure aggregate: variance, standard_deviation, cvar, ecc.); `RiskBudgeting` (optimization.md) COSTRUISCE un portafoglio a contributi assegnati risolvendo per i pesi, ma non ATTRIBUISCE il rischio di un'allocazione esterna arbitraria già data — sono operazioni inverse, come nota esplicitamente la pagina stessa. | Implementazione custom delle formule di Eulero (RC_i=w_i·cov(r_i,r_p)/σ_p per la deviazione standard; attesa condizionata di coda per l'ES; stimatore kernel Nadaraya-Watson per il VaR) usando la matrice Σ prodotta da skfolio.moments come input. |
| Agente risk | Diagnostica code: stima dei parametri α (esponente caratteristico), β (asimmetria), σ (scala) di una distribuzione Lévy-Pareto stabile sui rendimenti di portafoglio, con flag α<2 | Verificato in skfolio.distribution (distributions.md): copule disponibili (Gaussian/StudentT/Clayton/Gumbel/Joe/Independent) e marginali univariate Gaussian/StudentT/JohnsonSU/NormalInverseGaussian — nessuna distribuzione α-stabile/Lévy-Pareto nell'elenco. | scipy.stats.levy_stable.fit() (SciPy) oppure libreria dedicata (es. pylevy) per stimare i quattro parametri della stabile. |
| Agente risk | Indice di diversificazione di portafoglio DI_ρ(X) e marginale DI_ρ(Xi\|X), e impatto di rischio per fattore RI_ρ(L\|S) (generalizzazione di R² = var(E[L\|S])/var(L)) | Verificato in skfolio: `MaximumDiversification` (optimization.md) massimizza un 'diversification ratio' concettualmente collegato ma non lo espone come metrica calcolabile a posteriori su un'allocazione a pesi fissi arbitraria (nessuna proprietà `diversification_ratio` in portfolio.md); nessuna funzione di impatto di rischio fattoriale RI/R² in skfolio.distance, skfolio.linear_model o altrove nei file di riferimento. | Calcolo custom diretto dalle formule: DI = σ_portfolio / Σ(w_i·σ_i) dai dati Σ già stimati; RI tramite regressione (es. statsmodels OLS o skfolio.linear_model.CSLinearRegression adattata) di L sui fattori S, prendendo var(E[L\|S])/var(L). |
| Attribuzione del rischio | Decomposizione di Eulero per-componente (Contrib. Eulero, % contributo, DI_ρ(Xi\|X), RORAC(Xi\|X)) su un portafoglio già allocato con pesi fissi. | skfolio verificato: Portfolio espone solo misure aggregate di book (variance, standard_deviation, cvar), nessuna proprietà di contributo per-asset; RiskBudgeting risolve solo il problema inverso di costruzione (dato un budget, trova i pesi), non espone i contributi di un vettore pesi esterno arbitrario. | Codice custom: RC_i = x_i·(Σx)_i/σ(x) per la volatilità (Σ presa dallo stimatore skfolio), formule analoghe per VaR/ES da implementare a mano sopra le serie/covarianza stimate da skfolio. |
| Attribuzione del rischio | Stima kernel del contributo al VaR (Nadaraya-Watson, bandwidth di Silverman b=0.9·min(σ,R/1.34)·N^-1/5). | skfolio verificato: nessuna occorrenza di 'kernel'/'Silverman'/regressione non parametrica in nessun file di riferimento (optimization.md, priors.md, moments.md, distributions.md); VaR esiste solo come ExtraRiskMeasure aggregato 'non-convex — scoring only'. | Implementazione custom (numpy/scipy o statsmodels) dello stimatore kernel e della regola di Silverman. |
| Attribuzione del rischio | VaR di Cornish-Fisher e relativo contributo per differenziazione dei co-momenti (asimmetria/curtosi). | skfolio verificato: nessuna occorrenza di 'Cornish-Fisher' in tutta la Skill (grep su references/*.md e PATTERNS.md vuoto). | Implementazione custom: tensori di co-skewness/co-kurtosis, espansione di Cornish-Fisher e suo gradiente rispetto ai pesi. |
| Attribuzione del rischio | Contributo marginale with-without ρ(X)−ρ(X−Xi) e riconciliazione 'Contributi non allocati' / icona '≠Σ'. | skfolio verificato: nessuna funzione calcola direttamente ρ sul book privato di una componente per un vettore di pesi dato dall'esterno. | Orchestrazione custom che rivaluta ρ(X−Xi) riusando lo stimatore di rischio di skfolio come primitiva, poi sottrae/riconcilia lato applicativo. |
| Attribuzione del rischio | Indici di diversificazione DI_ρ(X)=ρ(X)/Σρ(Xi) e DI_ρ(Xi\|X)=contributo/ρ(Xi) stand-alone. | skfolio verificato: MaximumDiversification ottimizza un 'diversification ratio' diverso (obiettivo di costruzione, non readout per un book dato); nessun simbolo calcola questo DI specifico. | Codice custom che combina i ρ(Xi) stand-alone (da Portfolio single-asset) con i contributi di Eulero del Gap 1. |
| Attribuzione del rischio | Impatto di rischio dei fattori RI_ρ(L\|S) come generalizzazione di R² (contributo di Eulero di E[L\|S] a ρ(L)). | skfolio verificato: TimeSeriesFactorModel fornisce loading matrix e covarianza fattoriale ma nessuna metrica RI_ρ pronta all'uso. | Calcolo custom di var(Xβ)/var(L) (e generalizzazione non lineare) sopra l'output di TimeSeriesFactorModel. |
| Attribuzione del rischio | Contributo atteso alla perdita di Qian, c_i = p_i + D_i/L. | skfolio e yfinance verificati: nessun simbolo implementa questa formula finanziaria specifica (percentage-contribution-to-risk più termine D_i condizionato a una soglia di perdita L). | Formula custom sopra p_i (beta, già derivabile da Σ) e μ_i/w_i stimati da skfolio. |
| Attribuzione del rischio | Verifica assiomatica di Denault (no undercut su 2^n coalizioni, simmetria, riskless allocation) e principio di Aumann-Shapley. | skfolio verificato: nessun simbolo di teoria dei giochi cooperativi/valore di Shapley in optimization.md, priors.md o altrove nella Skill. | Codice combinatorio custom (verosimilmente limitato/campionato alle coalizioni/sotto-portafogli effettivamente definiti nel book, come richiesto dalla spec). |
| Attribuzione del rischio | Portafogli ERC/RB di confronto quando Misura = VaR o VaR Cornish-Fisher. | skfolio verificato: RiskBudgeting accetta solo RiskMeasure convesse (STANDARD_DEVIATION, VARIANCE, CVAR, CDAR, ...); VaR e Cornish-Fisher sono ExtraRiskMeasure 'non-convex — scoring only', non utilizzabili come risk_measure di RiskBudgeting. | Solver numerico custom (es. SLSQP) che risolve RC_i(x)=b_i con le formule di contributo VaR/Cornish-Fisher del Gap 1-3, per i due soli casi non coperti da RiskBudgeting. |
| Attribuzione del rischio | Serie storiche dei fattori di rischio sistematici S (value, momentum, size, macro) da passare come y a TimeSeriesFactorModel. | yfinance verificato: nessun dataset di fattori di rischio accademici/standard (Fama-French o simili) in ticker.md, market_search.md, financials.md, analysis.md — solo prezzi grezzi e rollup di settore/industria. Nota: la spec di pagina esplicita che i fattori sono 'già disponibili nel sistema' (curati altrove, es. pagina Segnali & Fattori), quindi non è compito di questa pagina procurarli. | Pipeline dati fattoriale a monte (libreria Ken French/pandas-datareader, vendor tipo Barra/MSCI, o fattori proprietari costruiti dal team Segnali & Fattori) — dipendenza da un'altra pagina, non un gap di libreria per questa. |
| Backtest & Validazione | Haircut Sharpe Ratio (Harvey & Liu): t-statistic, p^S (test singolo), p^M (multi-test su N prove), ĤSR e sconto proporzionale hc — Tab 2 | skfolio verificato (optimization.md, model_selection.md, portfolio.md, SKILL.md): nessun simbolo haircut/HSR/p-value multi-test; skfolio espone solo Sharpe grezzo, non la correzione di Harvey-Liu | codice custom (scipy.stats.t / scipy.stats.norm) che implementa esattamente le formule del dominio: t=μ̂/(σ̂/√T), p^M=1-(1-p^S)^N, inversione per ĤSR |
| Backtest & Validazione | Correzioni p-value per test multipli — Bonferroni, Holm, Benjamini-Hochberg-Yekutieli (BHY) — Tab 3 | skfolio verificato: nessuna procedura di aggiustamento p-value nel package (solo make_scorer/GridSearchCV per tuning, non per correzione FWER/FDR) | libreria esterna statsmodels (statsmodels.stats.multitest.multipletests, metodi 'bonferroni'/'holm'/'fdr_by') oppure implementazione custom delle tre formule già fornite nel dominio |
| Backtest & Validazione | Deflated Sharpe Ratio (DSR) e Probabilistic Sharpe Ratio (PSR), incl. SR₀ atteso (formula con costante di Eulero-Mascheroni e approssimazione di Gumbel) — Tab 4 | skfolio verificato: nessun DSR/PSR/expected-max-Sharpe tra le misure o i metrics; skfolio fornisce solo gli input grezzi (Sharpe, skew, kurtosis via ExtraRiskMeasure), non la formula di deflazione | codice custom (scipy.stats.norm per Z e Z⁻¹) che implementa la formula DSR = Z[(ŜR-ŜR₀)√(T-1) / √(1-γ̂₃ŜR+((γ̂₄-1)/4)ŜR²)] usando gli input skfolio come sorgente dati |
| Backtest & Validazione | Probability of Backtest Overfitting (PBO) e procedura CSCV di Bailey et al.: partizione in S sottomatrici, combinazioni S/2, rango OOS, logit λ_c, φ=∫f(λ)dλ, regressione degradazione β, Prob[R̄<0], test di dominanza stocastica 1°/2° ordine — Tab 5 | skfolio verificato: CombinatorialPurgedCV genera split combinatoriali con purge/embargo (concetto affine ma non identico al CSCV originale, che è una partizione pura senza purging) e ritorna solo Population di portafogli — non calcola rango OOS→logit, φ, regressione IS/OOS né dominanza stocastica | codice custom sopra l'output di CombinatorialPurgedCV: calcolo rango, trasformazione logit, istogramma f(λ), integrale φ, regressione OLS (statsmodels/scipy) per β, e test di dominanza stocastica (confronto CDF campionarie, es. con scipy.stats.ecdf o implementazione custom) |
| Backtest & Validazione | Numero di prove indipendenti implicite N̄ = ρ̂+(1-ρ̂)M e correlazione media fuori diagonale ρ̂ tra le serie di rendimento delle M prove tentate — Tab 1, Tab 3 | skfolio verificato: nessun estimatore dedicato a 'correlazione media tra strategie candidate' come statistica di processo (skfolio calcola matrici di covarianza/correlazione tra asset per l'ottimizzazione, non tra run di backtest storici) | codice custom con pandas/numpy (DataFrame.corr() sulle serie di rendimento delle prove loggate + media fuori diagonale) — dato di processo, non di ottimizzazione |
| Backtest & Validazione | Persistenza del 'Log delle prove tentate' (M strategie, import CSV, aggiunta manuale, colonne nome/p-value/Sharpe+T) — Tab 1, Tab 3 | né skfolio né yfinance offrono storage/CRUD; sono librerie di calcolo e di dati di mercato, non di persistenza applicativa | backend custom (tabella DB run_trials + endpoint import CSV/validazione) — infrastruttura applicativa, non libreria dati/quant |
| Backtest & Validazione | Dichiarazione di motivazione economica (Tab 7): testo tesi, riferimenti, date, stato salvato/bloccato, badge Criterio 2; checklist Criterio 1 (Tab 6) con persistenza per run e permessi per ruolo | sono asserzioni di processo inserite manualmente dall'analista, non calcolabili né da skfolio né da yfinance per definizione del dominio stesso ('non dedotta a posteriori dal backtest') | backend custom (tabella run_validation_criteria, RBAC per ruolo Analista) — nessuna libreria quantitativa è pertinente qui |
| Backtest & Validazione | Esportazione 'report PDF' con tutte le 7 tab e i grafici in stato corrente, con stati asincroni (in corso/errore/timeout/successo) | fuori perimetro sia di skfolio (libreria di ottimizzazione) sia di yfinance (libreria di dati mercato) | libreria di rendering PDF lato backend (es. WeasyPrint, o rendering headless via Playwright già usato altrove nello stack) + job asincrono con polling di stato |
| Dashboard | Registro del portafoglio osservato: pesi x_i correnti, nome ("Portafoglio Alpha"), lista portafogli selezionabili, timestamp di ultimo aggiornamento | skfolio calcola statistiche a partire da rendimenti e pesi ma non persiste/serve quali pesi appartengano a quale portafoglio nominato; yfinance fornisce solo dati di mercato, non lo stato del fondo. Verificato in entrambe le skill: nessun modello di storage/registro portafogli in nessuna delle due. | API/DB propria del fondo (già prevista in ARCHITECTURE.md come 'optimizer API') che espone il portafoglio corrente e i suoi pesi al frontend |
| Dashboard | Istanziare un oggetto Portfolio di skfolio (con le sue proprietà mean/variance/standard_deviation/cvar/sortino/...) a partire da pesi ESTERNI già fissati (il portafoglio osservato), non derivati da un fit di ottimizzazione | portfolio.md documenta Portfolio solo come output di model.predict(X) dopo il fit di un optimizer; nessun costruttore diretto da pesi arbitrari è documentato nei file di riferimento skfolio consultati | Calcolo custom w'μ e w'Σw a partire dagli output mu_/covariance_ di EmpiricalMu/EmpiricalCovariance, oppure verifica del costruttore Portfolio(X=, weights=) contro la doc ufficiale skfolio.org (non presente nella skill) |
| Dashboard | Semi-varianza e MAD come valori leggibili per il portafoglio osservato (card Blocco C) | Verificato in skfolio: portfolio.md elenca solo mean/variance/standard_deviation/cvar/sortino_ratio/calmar_ratio/max_drawdown come proprietà di Portfolio; RiskMeasure.SEMI_VARIANCE e MEAN_ABSOLUTE_DEVIATION (optimization.md) sono documentati solo come risk_measure target per l'ottimizzatore MeanRisk, non come scorer read-only su un vettore di pesi fisso; model_selection.md mostra make_scorer solo per RatioMeasure o funzioni custom | Implementazione custom delle formule già fornite nella pagina (semi-Var e MAD in forma discreta) applicate ai rendimenti storici pesati, oppure conferma di un'API skfolio più ampia (skfolio.measures) non coperta da questa skill |
| Dashboard | Conteggio di dominanza media-varianza rispetto ad altri portafogli ("Dominato da: 2", "Domina: 3", "Non confrontabili: 5") — Blocco D | Verificato in skfolio: l'unico strumento Pareto-correlato è SelectNonDominated in skfolio.pre_selection (distance_clustering.md), che filtra ASSET in fase di pre-selezione, non PORTAFOGLI; Population (portfolio.md) espone solo summary/plot_cumulative_returns/plot_composition/plot_frontier/filter/sort, nessun metodo di classificazione a coppie per dominanza parziale | Codice custom che applica la definizione di dominanza media-varianza data nella pagina stessa (E(X1)≥E(X2) e Var(X1)≤Var(X2), disuguaglianza stretta su almeno una) a coppie di (E,Var) ottenute per ciascun portafoglio via EmpiricalMu/EmpiricalCovariance |
| Dashboard | Rendimento REALE (corretto per l'inflazione) da mostrare in Performance (Blocco B) | Verificato in entrambe le skill: yfinance copre solo dati di mercato (prezzi, dividendi, fondamentali), nessun indice CPI/inflazione; skfolio non stima/importa serie macro | Fonte dati macro esterna (es. API FRED per CPI) per deflazionare i rendimenti nominali prima del calcolo di Performance |
| Dashboard | Rendimento TOTALE netto di costi di transazione e tasse per la posizione già realizzata (non in fase di ottimizzazione) — definizione di Performance | skfolio espone transaction_costs/management_fees solo come parametri di INPUT di MeanRisk usati durante la ricerca dei pesi ottimi (optimization.md), non come motore di contabilità post-hoc su una posizione storica fissata; yfinance non fornisce dati fiscali/di conto | Modulo di accounting custom (o dati dal ledger/contabilità del fondo) per nettare tasse e costi di transazione dal rendimento realizzato |
| Dashboard | Stato semaforico (ok/attenzione/critico/n·d.) delle sei condizioni critiche: stabilità delle relazioni tra asset (break strutturali), sensibilità ai valori iniziali, ribilanciamento, errori e frodi — Blocco E | Verificato in skfolio: nessun test di break strutturale, nessuna diagnostica di sensitività ai valori iniziali della composizione ottima, nessun tracciamento di frequenza di ribilanciamento e nessun modulo antifrode nei file di riferimento consultati (RegimeAdjustedEWCovariance in moments.md produce un regime_multiplier_ che è al più un proxy parziale per instabilità di correlazione, non un semaforo pronto); yfinance non copre nulla di questo | Logica diagnostica custom: test di break strutturale (es. CUSUM/Chow su rolling covariance), analisi di sensitività via re-run perturbati dei pesi ottimi, tracking del calendario/eventi di ribilanciamento, sistema di monitoraggio/anomaly-detection per errori e frodi — nessuno di questi è coperto da skfolio o yfinance |
| Dashboard | Classificazione qualificante del portafoglio (badge: quantitativo/qualitativo/misto; conservativo/moderato/aggressivo; strategico/tattico/misto) — header | Sono criteri di business/dominio definiti a monte (in "Obiettivo & Vincoli" / deliberazione), non grandezze calcolabili da dati di mercato o da un ottimizzatore; né skfolio né yfinance offrono tassonomie di questo tipo | Campo di classificazione manuale/regola di business persistito lato backend del fondo, la Dashboard lo legge in sola lettura |
| Dati alternativi & sentiment | Dizionari testuali Harvard IV-4 (General Inquirer, 77 categorie) e Loughran-McDonald (Fin-Neg 2.337 vocaboli + liste ausiliarie) più motore di ponderazione proporzionale/tf.idf sui documenti (Pannello 2) | Verificato skfolio: nessun modulo di dizionari/NLP testuale (solo linear_model, preprocessing cross-sezionale numerico, ottimizzazione portafoglio). Verificato yfinance: nessun accesso a testo di articoli o filing, solo metadati/link. | Dataset dizionari LM Master Dictionary + Harvard IV-4 (fonti esterne, es. sito Loughran-McDonald) e libreria NLP (es. scikit-learn TfidfVectorizer o nltk) per calcolare tf, idf, w_i,j. |
| Dati alternativi & sentiment | Testo integrale di articoli media (rubrica WSJ) e dei 10-K per l'analisi di pessimismo/tono (Pannelli 2, 3) | yfinance Ticker.sec_filings fornisce solo form type/data/URL del filing, non il contenuto; nessun archivio media storico. | Fetch testo via SEC EDGAR full-text API (usando l'URL da sec_filings) e/o archivio media a pagamento (Factiva/WSJ) per la rubrica giornaliera. |
| Dati alternativi & sentiment | Modello VAR a 5 ritardi (Dow~Pess+Vlm+Exog) e test di causalità di Granger su Calm/Dow (Pannelli 3, 6) | Verificato skfolio: CSLinearRegression è regressione cross-sezionale per-periodo (un β per t), non un VAR multivariato con propri ritardi né un test di Granger; nessun symbol equivalente nei file optimization.md/priors.md/model_selection.md. | statsmodels.tsa.api.VAR e statsmodels.tsa.stattools.grangercausalitytests (o implementazione custom). |
| Dati alternativi & sentiment | Costruzione dell'indice SENT/SENT⊥ di Baker-Wurgler (PCA su CEFD, TURN, NIPO, RIPO, S, PDND con ritardi, poi ortogonalizzazione a produzione industriale/consumi/occupazione/recessioni) | Nessuna utility di PCA-su-proxy-di-sentiment o ortogonalizzazione macro in skfolio (solo PCA implicita dentro alcuni prior/covariance, non per indici di mercato esogeni). yfinance non espone sconto fondi chiusi (CEFD), quota azionaria nelle nuove emissioni (S) né premio sui dividendi (PDND); TURN e NIPO/RIPO solo parzialmente ricostruibili da shares_full e Calendars.ipos. | sklearn.decomposition.PCA per l'estrazione della prima componente + serie macro esterne (FRED per produzione industriale/consumi/occupazione/NBER recessioni) + fonte dati CEFD/PDND (es. CRSP, dataset pubblico Baker-Wurgler su rvz's site). |
| Dati alternativi & sentiment | Search Volume Index / ASVI da Google Trends (Pannello 5) | Verificato yfinance: nessun connettore Google Trends nei file ticker.md/market_search.md/screener.md; è dato esterno a Yahoo Finance. | pytrends (API non ufficiale Google Trends) o Google Trends API ufficiale. |
| Dati alternativi & sentiment | Umori collettivi social (OpinionFinder bipolare + GPOMS 6 dimensioni) da flussi tweet/social storici, standardizzazione locale Z, e rete SOFNN per accuratezza direzionale (Pannello 6) | Verificato yfinance: nessun dato social/testuale (solo news headline via Ticker.news, senza classificazione emotiva). Verificato skfolio: nessuna rete neurale fuzzy auto-organizzante né classificatore di mood testuale. | Vendor dati social (X/Twitter API storica, StockTwits, o dataset Bollen et al. se disponibile) + classificatore GPOMS/OpinionFinder + implementazione custom SOFNN (nessuna libreria standard la fornisce). |
| Dati alternativi & sentiment | Motore di sentiment su larga scala per news (rete a 3 strati addestrata su corpus triplo-annotato, output probabilità positiva/negativa/neutra per articolo, su archivio storico multi-milione di articoli tipo Reuters) | yfinance Ticker.news restituisce solo headline/link/publisher recenti (nessun punteggio di tono, nessun archivio storico su larga scala); skfolio non ha moduli NLP/text-classification. | Vendor news analytics (RavenPack, Refinitiv News Analytics, Bloomberg NEWS) oppure pipeline NLP custom (es. fine-tuning di un classificatore su corpus annotato). |
| Dati alternativi & sentiment | Suite ML comparativa completa del Pannello 8: Elastic Net con perdita di Huber e validazione iper-parametri, PCR, PLS, alberi/foreste/boosting, reti NN1-NN5, metrica R²_oos riferita a zero, Sharpe* e spread long-short a decili | Verificato skfolio: CSLinearRegression/CSLinearRegressorWrapper offrono solo WLS ordinario per-periodo senza regolarizzazione, perdita di Huber, selezione di iperparametri su validation set né score R²_oos-vs-zero; nessun modulo equivalente a PCR/PLS/foreste/NN dedicato al premio al rischio nei file optimization.md, priors.md, moments.md, model_selection.md, online_learning.md. | scikit-learn (ElasticNet, PCA+LinearRegression per PCR, PLSRegression, RandomForestRegressor, GradientBoostingRegressor, MLPRegressor) più implementazione custom della perdita di Huber e della metrica R²_oos a zero; eventualmente wired nel contratto cross-sezionale di skfolio via CSLinearRegressorWrapper solo per le varianti lineari. |
| Deliberazione del fondo | FundState come oggetto tipizzato, immutabile (frozen dataclass), checkpointato ad ogni scrittura di agente (macro_view, allocation, risk_verdict, proposed_orders, approvals), interrogabile come sequenza storica per il pannello 'FundState grezzo' e 'Diff vs precedente'. | Verificato skfolio: offre solo Portfolio/MultiPeriodPortfolio/Population (risultati di ottimizzazione), nessun costrutto di stato condiviso multi-agente né store di checkpoint applicativi. Verificato yfinance: nessuna pertinenza, è libreria dati mercato. | Codice custom lato backend: dataclass frozen FundState (Python) + tabella/append-log di checkpoint (es. Postgres con colonna JSON versionata, o event store dedicato) scritto da ogni agente via replace(state, campo=valore). |
| Deliberazione del fondo | Trasporto SSE per-nodo con campi data/event/id/retry ed endpoint di riconnessione che onora l'header Last-Event-ID. | Nessuna delle due skill copre streaming HTTP/SSE applicativo — skfolio è libreria di calcolo, yfinance ha solo yf.WebSocket per quote realtime Yahoo, non per eventi applicativi custom. | sse-starlette (EventSourceResponse) o StreamingResponse nativa di FastAPI, con gestione manuale di Last-Event-ID e buffer di replay lato server. |
| Deliberazione del fondo | Orchestrazione dei quattro agenti (macro/allocation/risk/execution) con interfaccia uniforme Agent.run(state: FundState) -> FundState, ciascuno 'scatola sigillata' senza chiamate dirette ad altri agenti. | skfolio non ha concetto di agente né di grafo di esecuzione multi-step con stato condiviso; è un framework sklearn-style fit/predict per portafogli, non un orchestratore di agenti. | Framework di orchestrazione custom (o libreria terza tipo LangGraph) che implementi il pattern a stato condiviso descritto in ARCHITECTURE.md; skfolio va invocato solo internamente al singolo nodo (es. dentro l'agente allocation) come dettaglio implementativo, non come orchestratore. |
| Deliberazione del fondo | Log eventi SSE ordinabile per id con distinzione fra evento 'deliberation.node' e 'portopt.progress', e correlazione riga-log ↔ nodo-timeline. | Nessuna libreria delle due fornisce modellazione di eventi applicativi/audit log; portopt.progress è un evento di progresso dell'ottimizzatore ma la sua emissione/trasporto va costruita a mano attorno alla chiamata skfolio, non è esposta da skfolio stesso. | Tabella/coda eventi lato backend (id progressivo, categoria, payload) + wiring che, durante l'esecuzione dell'agente allocation, emette eventi di progresso incapsulando le iterazioni/step del solver skfolio. |
| Deliberazione del fondo | Esportazione della timeline dei checkpoint FundState nel formato di checkpointing (audit export). | Non è funzione di libreria di ottimizzazione o dati mercato; è serializzazione di stato applicativo. | Endpoint custom di export (JSON/JSONL) che serializza la sequenza di checkpoint FundState così come persistita — serializzazione standard (pydantic/dataclasses.asdict), nessuna libreria terza necessaria. |
| Factor timing & rotazione | 2-state (value, §2.2) and 4-state (multi-asset, §2.4) Markov-switching regime models with transition probabilities, driving the 'Regime valutazione value' and 'Regime multi-asset' pills in Tab1 and the SDF regime markers in Tab5. | Checked skfolio moments.md/optimization.md/distance_clustering.md — only RegimeAdjustedEWCovariance exists, an EW-covariance multiplier reacting to a short-term-vs-long-term vol signal, not a labeled-state HMM/Markov-switching estimator with π_t stationary probabilities and transition matrix p_ij. | Custom HMM/Markov-switching regression (e.g. statsmodels MarkovRegression/MarkovAutoregression, or hmmlearn) fit outside skfolio; feed resulting state probabilities into skfolio only as exogenous conditioning. |
| Factor timing & rotazione | Macro/financial-conditions/sentiment predictor series: credit spread, TED spread, money-supply growth, GDP, capacity utilization, consumer confidence, ISM PMI, CAPE — the 5-category predictor taxonomy of Tab3 (§9). | Checked yfinance ticker.md, financials.md, market_search.md, screener.md — yfinance covers per-company/ETF prices, valuation ratios, and financial statements only; no macro/economic-indicator series. | External macro data vendor — FRED via fredapi/pandas-datareader (credit spreads, T-bill, GDP, capacity utilization, consumer confidence), CBOE for VIX history if not proxied via a price ticker. |
| Factor timing & rotazione | Academic long-short factor return series (Value, Momentum, Size, Quality, LowVol, BAB) and the 50-anomaly CRSP-COMPUSTAT panel behind §5's PC1-PC5 decomposition and book-to-market predictor. | yfinance (checked ticker.md/financials.md) only serves individual security/ETF prices and per-company valuation ratios — it does not provide constructed academic long-short factor portfolios or characteristic-sorted anomaly deciles. | AQR data library / Ken French data library (or WRDS) for factor and anomaly return series; alternatively construct proxy long-short returns in-house from stock-level fundamentals pulled via yfinance, at lower fidelity. |
| Factor timing & rotazione | SDF / near-arbitrage diagnostic: squared-Sharpe-ratio decomposition, Assumption-2 upper bound, R²_total invariant-to-rotation statistic (Tab5 near-arbitrage gauge and banner). | Checked skfolio optimization.md/model_selection.md/portfolio.md — no Hansen-Jagannathan bound / SDF-variance estimator or near-arbitrage constraint class; skfolio reports realized Sharpe/return moments only, not this HKS-specific diagnostic. | Custom numpy/statsmodels implementation of the HaddadKozakSantosh(2020) decomposition, run on top of skfolio's factor/PC return outputs. |
| Factor timing & rotazione | Multi-signal panel regression with interactions (financial conditions × macro × sentiment × valuation × trend, pooled over time) forecasting each factor's alpha — Tab2 'modello multi-segnale' and Tab3 alpha-model comparison (Models 1-6). | Checked skfolio linear_model.md — CSLinearRegression/CSLinearRegressorWrapper solve one cross-sectional regression PER TIME PERIOD across assets (Fama-MacBeth style, shape (T,N,K)), not a pooled time-series panel with interaction terms forecasting future factor returns; forcing it would misrepresent the estimator's contract. | Custom panel/time-series regression (statsmodels or linearmodels, with shrinkage for the 'Model 4 shrunk' variant) built outside skfolio, feeding forecasted mu into the optimizer. |
| Factor timing & rotazione | Injection of externally forecasted expected returns (the valuation-dependent alpha models, Tab3) as the mu input to a skfolio optimizer. | Checked skfolio moments.md — shipped mu estimators are only EmpiricalMu, EWMu, ShrunkMu, EquilibriumMu; none accepts a precomputed/externally-modeled mu series. | Thin custom skfolio-compatible BasePrior/mu-estimator subclass (sklearn contract) wrapping the panel-regression alpha forecasts — code we write, not a shipped skfolio symbol. |
| Gate di approvazione umana | Macchina a stati/orchestratore con interrupt() e checkpoint delle 4 fasi (pre-check → regole → gate umano → broker), audit trail replayable | Verificato skfolio (skill skfolio, overview: solo optimization/prior/moments/model_selection/portfolio — nessun concetto di workflow, interrupt o stato approvazione) e yfinance (skill yfinance, overview: solo recupero dati mercato/fondamentali/screener) — nessuno dei due copre orchestrazione o gestione stato applicativo | Orchestratore esterno con checkpointing (es. LangGraph già citato in ARCHITECTURE.md) + tabella DB per stato/transizioni checkpointed |
| Gate di approvazione umana | Adapter broker Trading 212 (POST /api/v0/equity/orders/{market,limit,stop,stop_limit}, switch base-URL demo/live) per far uscire l'ordine reale in fase 4 | Non è dominio né skfolio (nessun modulo di order routing/broker) né yfinance (solo lettura dati, nessuna scrittura/ordine verso broker) — verificato su entrambe le skill | Client REST custom per Trading 212 API (auth, gestione non-idempotenza, retry/dedup ordini) |
| Gate di approvazione umana | Stato kill-switch deterministico, aggiornamenti realtime della coda (cronometro, concorrenza multi-operatore, aria-live) | Nessuna delle due librerie gestisce stato applicativo realtime o concorrenza; skfolio è compute-only, yfinance offre solo WebSocket per quote di mercato (non per stato applicativo interno) | Canale push custom (WebSocket/SSE applicativo) + lock ottimistico/versioning in DB per rilevare decisioni concorrenti |
| Gate di approvazione umana | Criterio di calcolo della priorità (Alta/Media/Bassa) mostrato in tooltip, calcolato 'a monte' dal backend | Non è un output di alcun optimizer/risk measure skfolio né un campo dati yfinance — è business logic di code/triage specifica del sistema, non trovata in nessuna reference delle due skill | Servizio di scoring custom lato backend (regole o modello proprio), fuori perimetro di entrambe le librerie |
| Gate di approvazione umana | Audit log/traccia completa esportabile e persistenza storica delle decisioni (fuori scope pagina ma richiesta come link) | Né skfolio né yfinance offrono persistenza/audit trail — sono librerie di calcolo e di dati mercato rispettivamente, non di storage applicativo | Store di audit dedicato (tabella append-only o event store) con export |
| Guardrail & kill-switch | Attribuzione forward di Eulero (RC_i per deviazione standard, VaR, ES) e indice DI_ρ(X)/DI_ρ(X_i\|X) calcolati su un book con pesi x_i già dati (non da ottimizzare) — è il cuore quantitativo del tab "Limiti & budget di rischio" | Verificato skfolio (references/portfolio.md): `Portfolio` espone solo misure scalari di portafoglio (`variance`, `standard_deviation`, `cvar`, ...), nessuna proprietà di contributo per-asset; `RiskBudgeting` (references/optimization.md) risolve il problema inverso budget→pesi, non l'attribuzione forward su pesi esistenti. Nessuna funzione standalone di decomposizione di Eulero documentata in nessun reference file skfolio. | Codice custom (numpy/pandas): RC_i = x_i(Σx)_i/√(xᵀΣx) per σ; stima empirica delle attese condizionate per VaR/ES contribution (§3.3); DI_ρ = ρ(X)/Σρ(X_i). Σ può però venire legittimamente da uno stimatore skfolio (vedi sopra). |
| Guardrail & kill-switch | Motore guardrail deterministico stesso: `@field_validator`/`@model_validator(mode='after')` Pydantic v2, sollevamento ValueError/PydanticCustomError, orchestratore con `interrupt()` per il gate umano, checkpointing di ogni transizione di stadio | Non è materia né di skfolio (libreria di ottimizzazione portafoglio) né di yfinance (dati di mercato) — verificati entrambi, nessun riferimento a validation framework o orchestrazione agentica in nessuno dei due. | Pydantic v2 nativo per i validator; un framework di orchestrazione agentica con checkpointing (es. LangGraph, secondo lo schema AG2 citato in ARCHITECTURE.md) per l'`interrupt()` e la persistenza di stato. |
| Guardrail & kill-switch | Audit trail persistente (log eventi per stadio, esito, attore, timestamp, dettaglio errore, export CSV/JSON) e stato kill-switch (ambito, attore, motivazione, storico) | Nessuna delle due librerie fornisce persistenza/log — skfolio produce oggetti Portfolio in-memory, yfinance fornisce solo dati di mercato in lettura. | Store applicativo dedicato (tabella DB / event store) con schema evento (trade_id, stadio, esito, attore, timestamp, dettaglio) e query di export; stato kill-switch come record versionato per ambito. |
| Guardrail & kill-switch | Modello dati trade/ordine in pipeline (id, strumento, size, stadio corrente, esito per stadio) mostrato nelle card del tab "Pipeline & stato" | yfinance non modella ordini/trade proprietari (solo dati mercato pubblici); skfolio non ha nozione di trade/ordine, solo pesi di portafoglio risultato di un fit. | Modello dati custom (OMS interno o adapter broker) per rappresentare il trade lungo le 4 fasi della pipeline. |
| Guscio e navigazione | Elenco portafogli persistiti (id, nome, N, vettore pesi x_i, flag Σ=100%) servito via HTTP al selettore di portafoglio e al pannello Dettaglio | skfolio verificato (references/portfolio.md, optimization.md, model_selection.md): fornisce solo oggetti Portfolio/Population in-process, prodotti da predict()/cross_val_predict() dentro uno script Python — nessun layer di persistenza né endpoint REST. yfinance verificato (references/ticker.md, price_history.md, screener.md): copre solo dati di mercato (prezzi, fondamentali, screener), non memorizza né serve portafogli utente. | Endpoint REST proprio del core del fondo (FastAPI, già nello stack optimizer/) che espone GET /portfolios e GET /portfolios/{id}, serializzando l'oggetto Portfolio (definizione N-vettore, Σx_i=1) verso JSON; il frontend resta client HTTP puro, coerente col vincolo 'frontend cieco al core, core cieco al frontend'. |
| Guscio e navigazione | Ricerca/filtro live (debounce) sull'elenco portafogli per nome/id | Non è una funzione di ottimizzazione né di dati di mercato: skfolio e yfinance non gestiscono ricerca su oggetti applicativi persistiti (verificato negli stessi file sopra — nessuna delle due librerie espone query/filter su entità 'portafoglio salvato'). | Filtro client-side (Angular signal/computed) sulla lista già caricata, oppure parametro query ?q= sull'endpoint GET /portfolios lato backend custom — codice applicativo, non di dominio. |
| Guscio e navigazione | Contenuto testuale del popover 'perché 3?' (rimando al vincolo architetturale delle 3 destinazioni di navigazione) | Testo statico legato a un principio architetturale (frontend/src/features chiuso a 3 ambiti), non un dato o calcolo: fuori perimetro sia di skfolio che di yfinance per costruzione. | Contenuto statico di configurazione frontend (stringa/i18n) che referenzia ARCHITECTURE.md, non riformulandolo — nessuna libreria esterna necessaria, solo codice/contenuto custom. |
| Monitoraggio del rischio | Selettore β (90/95/99%) e holding period (1g/10g/1m) applicati dinamicamente a VaR/CVaR/CDaR ricalcolati in blocco A | skfolio: RiskMeasure.CVAR/CDAR, ExtraRiskMeasure.VALUE_AT_RISK e le proprietà Portfolio.cvar/.max_drawdown documentate in optimization.md e portfolio.md non espongono un parametro beta/holding-period configurabile nei file di riferimento della skill | layer custom che applichi la formula β-quantile (Rockafellar–Uryasev, scenari ordinati) direttamente su Portfolio.returns con β e holding-period scelti dall'utente, verificando contro l'installazione reale skfolio 0.20.1 se esiste un argomento non documentato qui |
| Monitoraggio del rischio | VaR parametrico 'RiskMetrics' (VaR_ε = z_ε·σ + μ) selezionabile in alternativa al metodo Storico | skfolio: nessuna classe dedicata trovata in optimization.md/portfolio.md; solo i building block (Portfolio.mean, Portfolio.standard_deviation) sono documentati, non la formula RiskMetrics stessa | codice custom: z_ε (scipy.stats.norm.ppf) combinato con Portfolio.mean/standard_deviation di skfolio |
| Monitoraggio del rischio | VaR-superiore α_β⁺, peso di combinazione λβ e decomposizione CVaR = λβ·VaR + (1-λβ)·CVaR⁺ (blocco H, diagnostica di stabilità) | skfolio: nessun simbolo per 'upper VaR' o λβ trovato in optimization.md/portfolio.md — solo VaR e CVaR di base sono esposti | implementazione custom delle formule discrete Rockafellar–Uryasev su scenari ordinati (α_β⁺, λβ) a partire dai rendimenti storici del portafoglio |
| Monitoraggio del rischio | AvDDRatio e CDaRRatio (tabella F) = R(x)/AvD e R(x)/CDaR | skfolio: RatioMeasure documentato in optimization.md elenca solo SHARPE_RATIO, SORTINO_RATIO, CALMAR_RATIO, CVAR_RATIO — nessun AVERAGE_DRAWDOWN_RATIO o CDAR_RATIO | divisione custom di Portfolio.annualized_mean per i valori AVERAGE_DRAWDOWN/CDAR già calcolati da skfolio (RiskMeasure), nessuna nuova libreria ma codice di composizione |
| Monitoraggio del rischio | Scomposizione gerarchica per agente (4 sleeve) con confronto fondo-vs-Σagenti per subadditività VaR/CVaR (blocco G) | skfolio: opera su un singolo Portfolio/insieme di pesi per fit/predict; nessun concetto nativo di 'fondo multi-libro/multi-agente' trovato in optimization.md o portfolio.md | orchestrazione applicativa: calcolare Portfolio separati per ciascun agente e per il fondo aggregato con skfolio, poi logica custom di confronto/badge (🟢/🔴) sulle somme |
| Monitoraggio del rischio | Serie storica dei pesi effettivamente realizzati per fondo e per ciascun agente (il 'cammino effettivamente realizzato' che alimenta la curva di drawdown D(x,t) e la serie VaR/CVaR nel tempo) | yfinance fornisce solo prezzi/OHLCV dei singoli strumenti, non pesi di portafoglio nel tempo; skfolio calcola misure dato un vettore di pesi ma non li genera né li persiste storicamente | fonte dati interna (registro ordini/esecuzioni dell'agente execution, o snapshot di portafoglio persistiti dall'app) da cui ricostruire pesi storici, poi combinarli con i prezzi yfinance per ottenere rendimenti di portafoglio via skfolio.preprocessing.prices_to_returns |
| Monitoraggio del rischio | Parametri operativi utente ν1/ν2/ν3 (soglie capitale drawdown), soglie di stato 80%/100%, soglia di instabilità H — persistenza e gestione | sono parametri di configurazione applicativa scelti dall'utente; né skfolio né yfinance gestiscono impostazioni/soglie utente | storage applicativo (DB/config utente) per ν1..ν3 e soglie badge, con validazione [0,1] lato backend/frontend — nessuna libreria di dominio coinvolta |
| Obiettivo & Vincoli | Vincolo di cardinalità K (#{i:wᵢ>0}≤/=K), lotto minimo di transazione, acquisto minimo lᵢ per titolo selezionato — variabili binarie δᵢ, MIQP/MILP | Verificato skfolio (references/optimization.md, references/distance_clustering.md, PATTERNS.md, evals.json — grep su cardinality/threshold/integer/lot): tutti gli ottimizzatori (MeanRisk, RiskBudgeting, HRP/HERC/NCO/Schur, Stacking) sono risolutori convessi via cvxpy; nessun parametro di cardinalità o variabile intera è documentato. | Layer custom di programmazione mista-intera (cvxpy + backend MIP tipo HiGHS/CBC/Gurobi) che implementi δᵢ∈{0,1}, lᵢδᵢ≤xᵢ≤uᵢδᵢ, Σδᵢ=K sopra/accanto al modello skfolio; oppure euristica two-stage (selezione titoli esterna + re-ottimizzazione pesi in skfolio sul sottoinsieme fisso). |
| Obiettivo & Vincoli | VaR come misura ottimizzabile in obiettivo o vincolo, coi due metodi Parametrico RiskMetrics / Storico | Verificato skfolio references/optimization.md: VALUE_AT_RISK è classificato in ExtraRiskMeasure, esplicitamente 'non-convex — scoring only' — non passabile a MeanRisk.risk_measure per ottimizzare/vincolare, solo per scoring post-hoc su un Portfolio già calcolato. | Codice custom: RiskMetrics parametrico z_ε·σ+μ da mu_/covariance_ del prior fittato con skfolio; metodo storico via quantile empirico (numpy/pandas); per uso come vincolo/obiettivo di ottimizzazione serve riformulazione custom (surrogato CVaR o scenario-MIP), assente in skfolio. |
| Obiettivo & Vincoli | Tail Conditional Expectation (TCE) e Worst Conditional Expectation (WCE) come misure di rischio | Verificato skfolio references/optimization.md — tabelle complete RiskMeasure ed ExtraRiskMeasure non contengono TCE né WCE. | TCE: implementazione custom = media condizionata dei rendimenti sotto -VaR (pandas su Portfolio.returns di skfolio). WCE: nessuna formulazione convessa nota — trattarla come indicatore solo teorico/informativo in UI, non ottimizzabile con gli strumenti disponibili. |
| Obiettivo & Vincoli | Misura two-sided ρ_{a,p}(X) = a‖(X-E[X])⁺‖₁+(1-a)‖(X-E[X])⁻‖ₚ-E[X] con parametri a, p | Verificato skfolio references/optimization.md — non presente né in RiskMeasure né in ExtraRiskMeasure. | Obiettivo cvxpy custom che implementi direttamente la formula, fuori dall'API pubblica di MeanRisk (o sottoclasse del suo optimizer di base — da verificare a livello di sorgente, oltre quanto documentato dalla skill). |
| Obiettivo & Vincoli | Vincolo di rischio massimo multi-livello: più righe (β_i, ω_i) simultanee sulla stessa misura nello stesso modello | Verificato skfolio references/optimization.md — MeanRisk espone un solo risk_measure per istanza; nessun parametro documentato per vincoli multi-soglia sulla stessa misura di rischio. | Comporre a mano più vincoli CVaR (formulazione LP di Rockafellar–Uryasev per ciascun β_i) come vincoli cvxpy aggiuntivi iniettati nel modello, tramite wrapper custom attorno a MeanRisk. |
| Obiettivo & Vincoli | Geometria libera dell'insieme di incertezza (Column-wise/Soyster, Box, Poliedrica) con parametro θ ad ancore 0/1/√n; obiettivo robusto esplicito Sharpe/VaR/utilità quadratica del caso peggiore | Verificato skfolio references/distance_clustering.md — uncertainty_set offre solo stimatori Empirical/Bootstrap per μ e Σ, parametrizzati da confidence_level; nessuna scelta di geometria (box/column-wise/poliedrica) né parametro θ; nessun obiettivo 'worst-case Sharpe/VaR/quadratic utility' documentato per MeanRisk (solo minimizzazione del rischio caso-peggiore tramite gli uncertainty set forniti). | Classi di uncertainty set custom con le geometrie box/column-wise/poliedrica parametrizzate da θ (verificare se le basi skfolio sono sottoclassabili); obiettivo SOCP custom via cvxpy diretto (fuori da MeanRisk) per Sharpe/VaR/utilità quadratica del caso peggiore secondo le formule del documento di dominio. |
| Obiettivo & Vincoli | Motore di validazione/coerenza (avvisi ⚠/errori ⛔, conflitti fra sezioni, test di consistenza 'utilità quadratica o rendimenti ellittici', duplicati β_i, K vs righe bound) e persistenza bozza/configurazione | Né skfolio (libreria di ottimizzazione, nessuna logica di form-validation/persistenza) né yfinance (dati di mercato) coprono regole di business o storage — verificati entrambi gli indici delle skill. | Servizio backend custom (stack di progetto: FastAPI + repository pattern) che codifichi le regole descritte in 'Fondamenti di dominio' (consistenza obiettivo↔misura, robustezza↔obiettivo caso-peggiore, K vs bound, duplicati β_i) più tabella DB per bozze/configurazioni salvate. |
| Regimi di mercato | Stima del filtro di Hamilton (2 stati) e dello smoother — probabilità filtrate/smoothed, matrice di transizione p/q, durata attesa, MLE della verosimiglianza marginale — motore dell'intera pagina (barra modello, card di sintesi, tab Probabilità nel tempo/Transizioni & durate). | Verificato skfolio 0.20.1 (optimization.md, priors.md, moments.md, distance_clustering.md, online_learning.md, linear_model.md, preprocessing.md): nessuna classe implementa un modello di Markov-switching/HMM univariato o AR-switching; RegimeAdjustedEW* produce solo un moltiplicatore scalare di rischio, non uno stato discreto stimato per verosimiglianza. yfinance non fa stima statistica, solo dati. | statsmodels.tsa.regime_switching.markov_regression.MarkovRegression / markov_autoregression.MarkovAutoregression (o libreria HMM dedicata) per stimare (α0,α1,p,q,σ,φ) e ottenere filtered_marginal_probabilities / smoothed_marginal_probabilities. |
| Regimi di mercato | VAR a cambio di regime a 4 stati multivariato (Guidolin-Timmermann) con stima EM su large cap/small cap/obbligazioni, per card 'Regime corrente' e tab Momenti/Correlazione quando è selezionato il modello a 4 stati. | Nessun estimatore VAR-switching multivariato in skfolio 0.20.1 (verificato optimization.md, priors.md, moments.md); TimeSeriesFactorModel è un modello fattoriale statico, non uno switching-VAR con matrice di transizione stimata via EM. | Implementazione custom (EM su VAR-switching) oppure libreria specializzata esterna (es. porting da pacchetti R come MSwM/MSGARCH, o statsmodels esteso a mano) — non coperta né da skfolio né da yfinance. |
| Regimi di mercato | Nowcasting macro (modello a fattori dinamici + filtro di Kalman su pannello sbilanciato ~200 serie mensili) e diffusion index (PCA + previsione multi-step) per la tab 'Nowcasting macro'. | skfolio non ha modelli a fattori dinamici con filtro di Kalman su pannelli sbilanciati (TimeSeriesFactorModel richiede fattori già osservati y, non li stima da un pannello mixed-frequency); yfinance non espone alcuna serie macro/PIL/occupazione — solo dati equity/fund/ETF e calendario eventi prospettico. | statsmodels.tsa.statespace.DynamicFactorMQ (o pacchetto dedicato tipo nowcasting.de / FRED-MD) più fonte dati macro con vintages (FRED/ALFRED) — nessuna delle due librerie la fornisce. |
| Regimi di mercato | Serie macro storiche lunghe: PIL trimestrale USA, tasso T-bill, default spread (Baa-Aaa), term spread, dividend yield di mercato pluridecennale, crescita base monetaria, spread di funding-liquidity (TED e simili) — usate nei modelli value/beta condizionati (sez. 13-14) e nel ciclo macro (sez. 2). | yfinance (verificato ticker.md, price_history.md, financials.md, analysis.md, funds.md, screener.md, market_search.md) copre solo strumenti quotati (equity/ETF/indici) e calendario eventi prospettico, non serie macroeconomiche storiche o spread creditizi da agenzie di rating. | Vendor macro dedicato: FRED via fredapi/pandas-datareader (GDP, DGS3MO, BAA/AAA, T10Y2Y, TEDRATE) — libreria esterna non presente nello stack attuale. |
| Regimi di mercato | Pannello storico point-in-time di book-to-market, decili momentum/value/size, rendimenti aggiustati per delisting su decenni (1926/1932 in poi) per costruire WML, HML, SMB coerenti col dominio (sez. 10,11,13,15). | Verificato yfinance financials.md/analysis.md: balance_sheet/income_stmt coprono solo pochi anni recenti (annuale/trimestrale/TTM), nessun dato point-in-time storico né aggiustamento per delisting; nessuna copertura precedente all'IPO/quotazione del ticker (esclude episodi come luglio-agosto 1932). | Kenneth French Data Library (fattori HML/SMB/WML pronti) o CRSP+Compustat con point-in-time fundamentals — fuori dallo stack skfolio/yfinance. |
| Regimi di mercato | Regressioni con errori standard HAC/Newey-West per CAR_t su stato UP/DOWN (sez. 10) e per beta condizionati bear/up-market (sez. 11) e CAPM condizionato (sez. 14). | skfolio.linear_model.CSLinearRegression (verificato linear_model.md) è WLS cross-sezionale per-periodo (T,N,K)→(T,N), senza opzione di covarianza HAC e pensato per panel cross-sectional, non per serie storiche con autocorrelazione; yfinance non fa regressioni. | statsmodels.regression.linear_model.OLS(...).fit(cov_type='HAC', cov_kwds={'maxlags':...}) — libreria esterna. |
| Regimi di mercato | Modello GARCH asimmetrico come benchmark di confronto nella tab 'Correlazione di eccedenza' (regime-switching vs normale vs GARCH asimmetrico). | Nessun modello GARCH in skfolio 0.20.1 (verificato moments.md, distributions.md, optimization.md) — gli estimatori di covarianza sono EW/shrinkage/RMT, non condizionali GARCH. | Pacchetto `arch` (Python) per GJR-GARCH/EGARCH — libreria esterna. |
| Regimi di mercato | Soluzione del problema dinamico CRRA multi-periodo con domande di copertura intertemporali regime-dipendenti (pesi α*_t(s_t) via quadratura, sez. 5-6) da mostrare come sostanza dietro 'pesi ottimi differiscono per regime'. | skfolio.optimization (verificato optimization.md) risolve problemi convessi single-period (MeanRisk, RiskBudgeting, HRP/HERC/NCO/Schur, Stacking) — nessun solver per controllo stocastico dinamico multi-periodo con quadratura su probabilità di transizione. | Codice custom (value function iteration / quadratura numerica su scipy) — non ottenibile componendo skfolio; al più skfolio.optimization.MeanRisk può dare un'approssimazione statica per-regime (un fit separato per bucket), non la soluzione dinamica con hedging demand. |
| Report & Audit trail | Immutable per-decision audit log (model, params, full output) per agent per run, queryable/filterable/exportable | Checked skfolio skill overview + architecture (optimization/prior/moments/model_selection/etc. — no logging/audit module) and yfinance skill overview (market-data retrieval only, no logging). Neither library models LLM agent decisions or audit trails. | Custom backend: append-only audit_log table/store (run_id, agent, decision_type, model, params JSON, output text/JSON, timestamp, logging_status) written by each agent node; REST endpoint feeding regions ③/③b. |
| Report & Audit trail | Reproducibility state toggle (ATTUALE vs TARGET) tied to which Ollama model is selected, incl. prompt hash when TARGET | Not in skfolio (no LLM/prompt concept at all) nor yfinance (market data only). This is Ollama/LLM-serving concern, outside both libraries' scope entirely. | Custom config record (chosen model + seed/temperature support flag) + prompt-hash computation (e.g. sha256 of rendered prompt) at the LLM call site; surfaced as a flag on each logged decision. |
| Report & Audit trail | Broker order idempotency key generation, dedup check, and post-send status polling/reconciliation against Trading 212 | Checked yfinance skill (Yahoo Finance market-data only — no order placement, no broker execution API) and skfolio skill (portfolio construction/backtesting only — no live execution/broker adapter). Trading 212's own order API is outside both. | Custom broker adapter: client-generated idempotency key per order intent, dedup-check store, polling client against Trading 212 REST API (`POST /api/v0/equity/orders/...`, demo vs live base-URL) to reconcile registered vs actual order state; feeds region ④ (currently DEFERRED). |
| Report & Audit trail | Run timeline/event store per agent step (start/end timestamps, interrupted/restarted-from-scratch badges) | Neither skfolio (no run/orchestration-event tracking beyond CV fold results) nor yfinance (no execution telemetry) expose agent-orchestration timelines. | Custom pipeline-run event log (per-node start/end timestamp, restart flag) written by the multi-agent orchestrator; feeds region ⑤. |
| Report & Audit trail | Execution-posture/gate display: current environment (demo/live base-URL), gate-deliberate-off status, account constraints (Invest/ISA only, main-currency only, no order-by-value, BETA non-idempotent API) | This is static policy/account-constraint metadata about the Trading 212 integration, not a computed or fetched dataset — outside both skfolio's and yfinance's scope (neither touches broker config or account posture). | Custom config record read from the execution-gate service (env=demo\|live, gate_enabled bool, static list of account constraints); rendered read-only in banner ①. |
| Revisione & Ribilanciamento | Rendimenti attesi r(t+Δt,i) da inserire nel calcolatore verdetto (Tab1) e nelle formule di Smith/Stone&Hill | yfinance fornisce solo prezzi/rendimenti storici realizzati (OHLCV), non stime di rendimento atteso; skfolio stima mu storico (EmpiricalPrior) ma non è documentato nella Skill come input diretto a un calcolatore a 2 attività — il verdetto della pagina richiede un r atteso specifico per titolo, non un vettore mu di portafoglio ottimizzato | Modulo custom che deriva r(t+Δt,i) dai segnali del fondo (pagina 'Segnali & Fattori') o da input utente manuale; nessuna delle due librerie lo produce direttamente |
| Revisione & Ribilanciamento | Formula del verdetto di convenienza a due attività con imposta capital gain (r_R vs r_NR, soglia = r1 + incidenza imposta) — Tab 1 | Non è un problema di ottimizzazione di portafoglio (skfolio: verificato in optimization.md, portfolio.md, model_selection.md) né un dato di mercato (yfinance): è algebra a due variabili specifica del dominio (nota di dominio §'Il verdetto di convenienza') | Codice custom (funzione pura) che implementa le due espressioni r_NR/r_R e il confronto, con tassazione capital gain condizionata |
| Revisione & Ribilanciamento | Scomposizione costi di revisione (dati fisso, transazione variabile, imposta capital gain variabile condizionata) — Tab 1 | skfolio ha solo transaction_costs/management_fees come termini scalari nell'obiettivo di MeanRisk (verificato in optimization.md), non una scomposizione esplicita per tipologia con condizionalità su capital gain; yfinance non fornisce costi di intermediazione né aliquote fiscali | Modulo di calcolo custom + tabella aliquote fiscali/costi broker come dati di configurazione utente |
| Revisione & Ribilanciamento | Modello di Stone & Hill (1979): problema di revisione lineare risolto come knapsack lineare, ranking acquisto/vendita per c_i-bc_i / c_j+sc_j, regola di scambio — Tab 2 sotto-scheda | skfolio è framework di ottimizzazione convessa/gerarchica su rendimento-rischio (MeanRisk, RiskBudgeting, HRP/HERC/NCO/SchurComplementary — verificato in optimization.md); non espone solver LP/knapsack generico né vincoli di uguaglianza sugli importi zb/zs con limiti f_i per titolo in questa forma | Solver LP custom (es. scipy.optimize.linprog o PuLP/HiGHS) che implementa esattamente il problema di revisione lineare di Stone & Hill con i vincoli descritti |
| Revisione & Ribilanciamento | Information coefficient (convenzionale, ritardato/lagged, di orizzonte) e relazione IC^h ≈ avg(IC)·√(h+1) — Tab 3 | skfolio non espone un misuratore di IC fra segnale e rendimento cross-sezionale in nessuno dei file di riferimento controllati (optimization.md, priors.md via SKILL.md index, moments.md, model_selection.md, linear_model.md, preprocessing.md); ha CSLinearRegression per regressione cross-sezionale ma non un calcolo IC/decadimento dedicato | Codice custom: correlazione cross-sezionale rolling fra fattore laggato F_{t-l} e rendimento R_t (numpy/scipy/pandas .corr per periodo), calcolata sui rendimenti scaricati via yfinance e sui segnali prodotti dalla pagina 'Segnali & Fattori' |
| Revisione & Ribilanciamento | Turnover indotto dal segnale T = √(N/π)·σ_model·√(1-ρ_f)·E(1/σ) e autocorrelazione del fattore ρ_f — Tab 3 | skfolio non ha una proprietà 'turnover' né un misuratore di autocorrelazione di segnale in Portfolio/MultiPeriodPortfolio/Population (verificato in portfolio.md, unico riferimento su misure di portafoglio) — solo transaction_costs come termine di costo nell'obiettivo, non come conseguenza analitica della persistenza del segnale | Implementazione custom della formula (Grinold & Kahn) con autocorrelazione ρ_f calcolata su serie storiche di fattore (pandas .autocorr / corr fra F^t e F^{t+1}) |
| Revisione & Ribilanciamento | Media mobile del fattore MA(2), pesi v0/v1, e ottimizzazione information ratio vincolata all'autocorrelazione obiettivo ρ_target (IR = v'IC̄ / √(v'S_IC v) con vincolo ρ_f=ρ_target) — Tab 3 | Non trovato in skfolio: CSLinearRegression (linear_model.md, verificato via indice SKILL.md) fa regressione cross-sezionale su fattori ma non ottimizza pesi di combinazione fattoriale sotto vincolo di autocorrelazione; nessun estimator per 'information ratio vincolato' nei file controllati | Ottimizzatore custom (es. scipy.optimize con vincolo di uguaglianza non lineare su ρ_f, o programmazione quadratica con vincolo quadratico) implementato ad hoc |
| Revisione & Ribilanciamento | Modello di trading dinamico con rendimenti prevedibili e costi quadratici (Gârleanu–Pedersen): matrici B (factor loading), Φ (mean-reversion dei fattori), Λ (costi quadratici), risoluzione della Riccati/A_xx per l'aim portfolio, tasso di trading a/λ — Tab 4 | skfolio non implementa programmazione dinamica intertemporale con costi di transazione quadratici né stima Φ (velocità di ritorno alla media dei fattori); TimeSeriesFactorModel (verificato: cercato in optimization.md/moments.md/model_selection.md via SKILL.md, non compare in questi file) stima loading statici B ma non risolve l'equazione di Riccati per l'aim portfolio, né skfolio ha un parametro Λ matriciale (solo transaction_costs scalare in MeanRisk) | Implementazione custom della soluzione closed-form di Gârleanu & Pedersen (calcolo di A_xx via iterazione di Riccati algebrica, es. con numpy/scipy.linalg.solve_discrete_are) — libreria esterna specializzata in controllo ottimo (es. package Python `control`) o codice ad hoc |
| Revisione & Ribilanciamento | Aim portfolio ponderato per persistenza del fattore aim_t = (γΣ)^{-1}B(I + (a/γ)Φ)^{-1}f_t e traiettoria di convergenza x_t come media esponenziale degli aim passati — Tab 4 | Diretta conseguenza del modello Gârleanu-Pedersen sopra: nessun simbolo trovato in skfolio per aim portfolio, ridimensionamento fattori per Φ, o tasso di trading a/λ | Stesso modulo custom del punto precedente; Σ può riusare skfolio.moments (es. LedoitWolf) per la covarianza, ma la combinazione con B, Φ, a resta custom |
| Revisione & Ribilanciamento | Confronto Sharpe lordo/netto per tre strategie (Markowitz corrente, statico γ/λ ritoccati, dinamico ottimo) su universo di 15 futures materie prime 1996-2009, e risposta a shock del predittore — Tab 5 | skfolio può fornire Sharpe lordo/netto per un backtest walk-forward standard (vedi uso dichiarato sopra), ma non implementa la policy 'statica' (γ,λ ritoccati con aim=Markowitz corrente) né la policy 'dinamica ottima' di Gârleanu-Pedersen: entrambe richiedono il modulo custom del Tab 4; yfinance inoltre non copre dati storici di futures su materie prime 1996-2009 in questo dettaglio (verificato: price_history.md copre equity/ETF/index tickers standard, non garantisce copertura storica di contratti futures commodity di quel periodo) | Motore di simulazione custom che applica le tre policy (Markowitz via skfolio MeanRisk, statico e dinamico via modulo Gârleanu-Pedersen custom) sullo stesso path di rendimenti; per i dati futures 1996-2009, fonte dati storica alternativa (es. dataset accademico originale Gârleanu-Pedersen, Bloomberg/Refinitiv, o CSV precaricato) se non disponibile su Yahoo Finance per quei ticker/periodo |
| Risultati & Frontiera | Frontiera in forma chiusa (modalità 'non vincolata'): iperbole completa nel piano σ-E(R) via formula analitica x* = (γV⁻¹r − βV⁻¹e)π + (αV⁻¹e − βV⁻¹r) / (αγ−β²), con α,β,γ da V⁻¹. | Verificato in skfolio (optimization.md, moments.md, PATTERNS.md): tutti gli optimizer (MeanRisk incluso) risolvono via QP/solver numerico convesso, punto per punto; nessuna funzione espone la soluzione analitica chiusa con calcolo esplicito di α/β/γ e V⁻¹. | Codice custom (numpy: np.linalg.inv su covariance_ prodotta da EmpiricalCovariance, poi calcolo diretto di α, β, γ, x* secondo le formule del dominio) — skfolio resta utile solo per fornire mu_/covariance_ in input. |
| Risultati & Frontiera | Punto esatto sulla frontiera per un rendimento obiettivo π arbitrario digitato dall'utente (cursore π, 'Vai al punto'), sia in modalità long-only sia non vincolata. | Verificato in skfolio (optimization.md): nessun parametro documentato di tipo min_return/target_return/mu_target su MeanRisk per vincolare x'r=π esattamente; l'unico sweep documentato (risk_aversion in MAXIMIZE_UTILITY) produce punti approssimati, non un π scelto esattamente. | Per la modalità long-only: QP custom (es. cvxpy o scipy.optimize) con vincolo esplicito Σw_i·r_i=π sopra covariance_ di skfolio, oppure bisezione su risk_aversion fino a convergenza su π. Per la modalità non vincolata: uso diretto della formula chiusa custom del gap precedente (π è già un parametro libero della formula). |
| Risultati & Frontiera | Tab 'Contributo al rischio': per ciascun titolo x_i, SD(R_i), Corr(i,P), β_i^P, contributo a σ_P (= x_i·SD(R_i)·Corr(i,P) = x_i·β_i^P·SD(R_P)). | Verificato in skfolio (portfolio.md): Portfolio espone solo metriche a livello di portafoglio (variance, standard_deviation, sharpe_ratio, ...), nessuna proprietà di risk contribution/decomposizione per singolo asset. | Codice custom: da weights (skfolio) e covariance_ (skfolio EmpiricalCovariance), calcolare marginal = V·x, poi contributo_i = x_i·marginal_i/σ_P e β_i^P = marginal_i/σ_P²; nessuna libreria terza necessaria oltre numpy, ma la formula non è fornita da skfolio. |
| Risultati & Frontiera | Tab 'Diversificazione': diversification ratio D(w), correlazione media pesata ρ(w), concentration ratio CR(w), D² come 'fattori effettivi', per un portafoglio arbitrario selezionato (Corrente/GMVP/Tangenza/π), non solo per l'MDP. | Verificato in skfolio (optimization.md, portfolio.md): MaximumDiversification massimizza D(w) internamente ma non lo restituisce come metrica leggibile su Portfolio/RatioMeasure/PerfMeasure; nessuna delle tabelle di misure elenca diversification_ratio. | Codice custom: D(w)=w'σ/√(w'Vw), ρ(w) e CR(w) dalle formule del dominio, calcolati direttamente da weights e covariance_/correlation_ già prodotti da skfolio. |
| Risultati & Frontiera | Correlazione tra portafoglio corrente/selezionato e MDP, ρ_{P,MDP}, mostrata nel tab 'Diversificazione'. | Verificato in skfolio (portfolio.md, distance_clustering.md): nessuna funzione di correlazione cross-portfolio tra due vettori di pesi diversi è documentata; Population/Portfolio non espongono un tale metodo. | Codice custom: ρ_{P,MDP} = w_P'·V·w_MDP / (σ_P·σ_MDP), usando covariance_ e i due vettori di pesi già ottenuti dai fit skfolio. |
| Risultati & Frontiera | Tasso privo di rischio r_f editabile che alimenta CAL, Sharpe e selezione del portafoglio di tangenza. | Verificato in skfolio (optimization.md, portfolio.md): né MeanRisk (obiettivo MAXIMIZE_RATIO) né la proprietà sharpe_ratio di Portfolio documentano un parametro risk_free_rate configurabile in questa Skill. | Wrapper custom attorno all'obiettivo MAXIMIZE_RATIO: ricalcolare Sharpe/CAL a valle con r_f dell'utente usando mean/standard_deviation già esposti da Portfolio (SR=(mean−r_f)/std); verificare comunque il codice sorgente reale di skfolio 0.20.1 prima di escludere un parametro nativo non coperto da questa Skill. |
| Risultati & Frontiera | Flag titolo 'individualmente dominato' (⚑) nella tabella Pesi ottimi: E(R_i)≥E(R_j) e σ_i≤σ_j per qualche coppia. | Verificato in skfolio (optimization.md, portfolio.md, distance_clustering.md): SelectNonDominated filtra asset Pareto-non-dominati come pre-selezione per l'ottimizzazione, ma non espone un flag leggibile per-titolo da mostrare in UI dopo il fit. | Codice custom: confronto pairwise su mu_ e vettore delle σ_i (già prodotti da EmpiricalMu/EmpiricalCovariance) per marcare dominanza a livello di titolo singolo. |
| Run & Diagnostica solver | Stato formale del problema convesso a run terminato: p* con le tre casistiche (ammissibile/illimitato/finito), certificato di convessità, moltiplicatori duali ottimi λ*/ν*, tabella KKT con f_i(x*)/h_i(x*) e stato attivo/non attivo per complementarità, gap di dualità per iterazione, condizione di Slater, numero di iterazioni del metodo a punto interno | Verificato in skfolio (references/optimization.md, references/portfolio.md, references/model_selection.md, references/online_learning.md) e con grep sull'intera skill per 'dual', 'KKT', 'status', 'iteration', 'solver': MeanRisk espone solo fit/predict→Portfolio (pesi, misure di rischio/rendimento); nessun attributo documentato per stato del problema, valori duali, iterazioni del solver o gap di dualità per-iterazione. skfolio incapsula il solve (via cvxpy sotto il cofano, non documentato/esposto nella skill) senza esporre queste grandezze nella sua API pubblica. | Codice custom nel backend che, attorno al fit di MeanRisk, accede direttamente all'oggetto cvxpy.Problem interno (status, value, solver_stats.num_iters, constraints[i].dual_value) — richiede introspezione della struttura interna di skfolio o, più robusto, reimplementare il problema QP/SOCP direttamente in cvxpy (bypassando MeanRisk) per avere controllo esplicito su duali/iterazioni/gap da emettere come eventi SSE portopt. |
| Run & Diagnostica solver | Solver metaeuristico PSO per problema misto-intero (cardinalità K, lotti minimi, quota minima/massima) con penalizzazione esatta, stato sciame (M particelle, topologia fully-connected/von Neumann/ring, peso inerzia w o w_min/w_max, φ1/φ2, ε, k_max), curva di convergenza gbest per iterazione, scomposizione fitness (varianza vs penalità residua), diagnostiche di ammissibilità approssimata (% soddisfacimento vincoli, scostamento z_i da {0,1}) | Verificato in skfolio (references/optimization.md, elenco completo import 'Naive/Convex/Clustering/Ensemble'): nessuna classe PSO o metaeuristica di alcun tipo; nessun supporto a vincoli di cardinalità/lotti/variabili binarie nei parametri documentati di MeanRisk (solo linear_constraints, groups, min/max_weights, l1/l2_coef, transaction_costs — niente interi/binari). skfolio è library di ottimizzazione convessa/gerarchica, non metaeuristica. | Implementazione custom della PSO (es. libreria pyswarms, o PSO scritta a mano per controllo fine su topologia/inerzia/penalizzazione come da §8-9 del fondamento di dominio) che calcoli fitness = termine media-varianza + Σ termini di penalità (budget, cardinalità, quota min/max, binarietà z_i(1-z_i))/ε, ed emetta a ogni iterazione gbest, stato sciame e scostamenti z_i via evento SSE portopt. |
| Run & Diagnostica solver | Classificazione automatica della classe di problema (LP/QP/QCQP/SOCP vs misto-intero→PSO) da esporre nel badge 'Solver rilevato' e nel banner NP-hard, in base ai vincoli configurati a monte (pagina Obiettivo & Vincoli) | skfolio non espone (verificato in references/optimization.md) alcun meccanismo di introspezione/classificazione del tipo di problema costruito da MeanRisk; è l'utilizzatore a sapere quale classe sta istanziando, non la libreria a dichiararla. | Piccola funzione custom nel backend che ispeziona la configurazione del problema (presenza di vincoli quadratici, vincoli di norma/SOCP, presenza di variabili intere/cardinalità) e ne deriva l'etichetta di classe e il routing verso il ramo convesso o PSO, prima di avviare il run. |
| Run & Diagnostica solver | Infrastruttura di esecuzione asincrona e streaming: job/run persistito, endpoint SSE nativo FastAPI con eventi tipizzati (status, portopt, ping), campo id per Last-Event-ID, keep-alive ogni 15s, riconnessione senza perdita eventi | Non pertinente a skfolio né a yfinance: nessuna delle due librerie fornisce esecuzione asincrona di run, code di job o streaming SSE — sono librerie di calcolo/dati, non application server. | Implementazione custom lato FastAPI (EventSourceResponse o sse-starlette) con task runner per l'esecuzione del run in background (asyncio task o coda tipo Celery/RQ), storage dello stato run e replay degli eventi da Last-Event-ID; già indicato come requisito architetturale in architecture/ARCHITECTURE.md::451-457 citato nella pagina stessa. |
| Segnali & Fattori | Diagnostica OLS per titolo (t(α), R², σ_e idiosincratico, flag collinearità/VIF) e correzione di Blume (β_corretto=0,67β+0,33) — scheda 1 | Verificato skfolio (priors.md, linear_model.md, moments.md): LoadingMatrixRegression/TimeSeriesFactorModel espongono solo la loading matrix B, nessun attributo documentato per t-stat/R²/residual vol; 'Blume' non compare in nessun file di riferimento skfolio | Layer OLS custom (es. statsmodels.OLS o numpy.linalg.lstsq) che calcoli coefficienti+diagnostica per titolo, più shrinkage di Blume applicato manualmente ai β grezzi |
| Segnali & Fattori | Modello fattoriale statistico via PCA (K componenti, % varianza spiegata/cumulata) — scheda 1, opzione 'Statistica' | Verificato skfolio (priors.md, moments.md, distance_clustering.md): nessun estrattore di fattori statistici/PCA; TimeSeriesFactorModel richiede rendimenti fattoriali y già forniti, non li estrae dai rendimenti titolo | sklearn.decomposition.PCA (libreria terza, non skfolio) o eigendecomposizione custom di Σ campionaria, i cui score vengono poi passati come y a TimeSeriesFactorModel |
| Segnali & Fattori | Rapporti fondamentali pronti (ROE, ROA, ROIC, current/quick/cash ratio, D/E, debt-to-capital, equity multiplier, interest coverage, DuPont) e validazione di coerenza equity/enterprise sui multipli custom — scheda 2 | Verificato yfinance (financials.md, screener.md): income_stmt/balance_sheet/cashflow espongono solo voci contabili grezze; i campi EquityQuery coprono revenue/ebitda/netincome/totaldebt/totalcash/margini ma non ROE/D-E/current-ratio/DuPont | Layer di calcolo rapporti custom che mappi le voci grezze Yahoo (nomi/disponibilità variabili tra titoli) nelle famiglie di indici standard, con normalizzazione percentile settoriale e controllo bloccante numeratore/denominatore sui multipli custom |
| Segnali & Fattori | Motore IC/decadimento/legge fondamentale: IC convenzionale/ritardato/orizzonte, autocorrelazione ρ_f, turnover T, media mobile del segnale, curve turnover/IR vs ρ_target, ω*/VA*, aim portfolio con decadimento Φ e costi di transazione quadratici — schede 3 e 4 | Verificato skfolio (optimization.md, model_selection.md, moments.md, portfolio.md): nessuna metrica di information coefficient, breadth, turnover-da-segnale o formule di Grinold-Kahn; sono quantità di ricerca sui segnali, non di ottimizzazione di portafoglio | Libreria quant custom (numpy/pandas/scipy, eventualmente statsmodels per le regressioni) che implementi direttamente le formule del testo di dominio — nessun vendor le fornisce |
| Segnali & Fattori | Motore di costruzione alpha da valutazione DDM: g_n* implicito, conversione IRR/NPV, ribasamento settoriale, shrinkage verso implicito, valutazione comparativa via GLS, combinazione multi-fonte (IC combinato, correlazione fra fonti) — scheda 4 | Verificato skfolio (priors.md, optimization.md): nessun estimatore di valutazione/DDM o di combinazione di alpha in stile Grinold-Kahn; yfinance (analysis.md) fornisce solo gli input grezzi di crescita/stime, non il modello di conversione valutazione→alpha | Implementazione custom delle formule DDM/Golden-Rule e di una regressione GLS (es. statsmodels.GLS) per la valutazione comparativa; yfinance resta solo fonte di g_n grezzo |
| Segnali & Fattori | Pesatura 'per capitalizzazione' nello screener — scheda 5 | Verificato skfolio (optimization.md, sezione 'Naive Models'): i modelli naive sono limitati a EqualWeighted, InverseVolatility, Random — nessun allocatore cap-weighted | Normalizzazione custom w_i = marketCap_i / Σ marketCap, con marketCap preso da yfinance fast_info/screener |
| Stress testing & scenari | Calcolo della distanza di Mahalanobis Maha(r) = sqrt((r-μ)ᵀ Cov⁻¹ (r-μ)) per uno scenario arbitrario r — è la metrica di plausibilità mostrata ovunque in pagina (slider k, card Maha(r), tabella confronto scenari). | skfolio verificato (moments.md, distance_clustering.md): fornisce stimatori di μ/Cov ma nessuna funzione che calcoli la distanza di Mahalanobis di un punto arbitrario rispetto a quei momenti; il modulo skfolio.distance produce distanze tra asset (Pearson/Kendall/Spearman/...) per il clustering, non la distanza di un vettore-scenario da μ. | Codice custom (es. scipy.spatial.distance.mahalanobis o numpy) che consuma covariance_/mu_ prodotti da skfolio.prior.EmpiricalPrior. |
| Stress testing & scenari | Valore atteso condizionato r_C ai fattori fissati (trattamento iii, formula di Gauss condizionale su media/covarianza partizionate) — opzione di default consigliata nel pannello A. | skfolio verificato (priors.md, moments.md, distance_clustering.md): nessun estimatore espone il condizionamento analitico di una gaussiana multivariata su un sottoinsieme di componenti fissate; SyntheticData/VineCopula copre solo il campionamento condizionato (r_D), non la scorciatoia analitica r_C. | Codice custom con l'algebra lineare della gaussiana condizionata (partizionamento di Cov in blocchi fissati/non fissati), a partire dalla Cov stimata da skfolio. |
| Stress testing & scenari | Ricerca sistematica min CEP(r) soggetta a r ∈ Ell_k (ottimizzazione vincolata nello spazio degli scenari-fattore r, non nei pesi di portafoglio) — è il cuore della modalità B "Ricerca sistematica caso peggiore". | skfolio verificato (optimization.md): tutti gli ottimizzatori (MeanRisk, RiskBudgeting, DistributionallyRobustCVaR, HRP/HERC/NCO/SchurComplementary, StackingOptimization) ottimizzano pesi di portafoglio su una distribuzione di rendimento data; DistributionallyRobustCVaR è il più vicino concettualmente (worst-case) ma minimizza CVaR su una palla di Wasserstein di scenari di rendimento, non CEP su un'ellissoide di Mahalanobis nello spazio dei fattori a pesi fissi — variabile di ottimizzazione e vincolo diversi, non è una corrispondenza reale. | Solver custom (es. scipy.optimize / cvxpy) che implementa min CEP(r) s.t. (r-μ)ᵀCov⁻¹(r-μ) ≤ k², riusando la funzione di valutazione CEP(r) (vedi gap successivo). |
| Stress testing & scenari | Decomposizione Loss Contribution LC(i,r) e Maximum Loss Contribution MLC(i) — attribuzione della perdita ai singoli fattori (tabella "Fattori di rischio — contributo alla perdita"). | skfolio verificato (optimization.md, portfolio.md): espone risk_contribution per asset rispetto a una risk measure della distribuzione di rendimento (es. CVaR contribution), non la decomposizione LC(i,r) = [CEP(μ) − CEP con solo fattore i stressato] / [CEP(μ) − CEP(r)] definita nella pagina, che è specifica per scenario e per fattore macro. | Codice custom che implementa la formula del §9 direttamente: per ciascun fattore, valuta CEP con solo quel fattore al valore di r e gli altri a μ, e forma il rapporto. |
| Stress testing & scenari | Reverse stress test: dato un esito target (violazione capitale regolamentare, illiquidità, insolvenza, o soglia CEP/perdita custom) e una soglia k, cercare in modo esaustivo scenari in Ell_k che lo producono. | skfolio verificato (optimization.md): nessun ottimizzatore risolve un problema di ricerca inversa/feasibility su un dominio di scenari; è un problema diverso sia dall'allocazione di pesi sia dalla minimizzazione diretta di CEP. | Ricerca/ottimizzazione custom (es. bisezione su k combinata con solver di feasibility) che riusa lo stesso motore CEP(r) e lo stesso vincolo Ell_k del gap precedente. |
| Stress testing & scenari | Funzione di valutazione CEP(r): mappa uno scenario di fattori macro (PIL, tassi, cambio) nel valore/P&L del portafoglio — è il building block da cui dipendono Maha, Maximum Loss, LC/MLC e reverse ST; nell'esempio del dominio (§10) è una simulazione Monte Carlo a 4 passi su 100.000 traiettorie per prestiti in valuta estera. | Verificati entrambi: skfolio lavora su matrici di rendimento storiche/simulate di strumenti già prezzati (nessun motore di pricing/revaluation guidato da driver macro); yfinance (price_history.md, market_search.md, financials.md) fornisce solo serie di prezzo/dati fondamentali di mercato, nessun modello di valutazione o simulatore Monte Carlo. | Modello di pricing/valutazione custom per classe di strumento (o motore Monte Carlo come nell'esempio del dominio) che traduce uno scenario di fattori in valore di portafoglio; è codice di dominio specifico del fondo, non una libreria generica. |
| Stress testing & scenari | Serie storiche di PIL (GDP) e tassi di interesse di policy domestico/estero, necessarie a stimare μ e Cov dei fattori macro non tradabili sui mercati. | yfinance verificato (price_history.md, market_search.md): nessuna serie storica di PIL o tassi di policy; yf.Calendars.economic_events è solo un calendario di date di eventi (CPI, Fed, ...), non valori storici della serie. | Fonte dati macro esterna: FRED (via fredapi/pandas-datareader), ECB Statistical Data Warehouse, Eurostat/ISTAT, o API della banca centrale rilevante. |
| Turbulence & rischio sistemico | Turbulence finanziaria d_t = (y_t-μ)Σ⁻¹(y_t-μ)' e classificazione outlier vs soglia χ²(n) (card B, grafico C, tabella H) | Verificato skfolio: nessun simbolo Mahalanobis/turbulence in optimization.md, moments.md, portfolio.md, distance_clustering.md. yfinance non fa calcolo, solo dati prezzi. | Codice custom: numpy.linalg.inv/solve su Σ da EmpiricalCovariance + scipy.stats.chi2.ppf per soglia |
| Turbulence & rischio sistemico | Decomposizione magnitude surprise (Σ solo diagonale) e correlation surprise = turbulence/magnitude (pannello D) | Nessuna classe skfolio produce questa decomposizione (verificato moments.md, optimization.md); è una variante custom della stessa forma quadratica di Mahalanobis, non presente in libreria | Codice custom: azzerare fuori-diagonale di Σ (np.diag(np.diag(Σ))), ricalcolare d_t, rapporto |
| Turbulence & rischio sistemico | Absorption ratio AR = Σ_i var(autovettore_i) / Σ_j var(attività_j) e ΔAR standardizzato (pannello E) | skfolio non espone una metrica 'absorption ratio'; DenoiseCovariance/DetoneCovariance (moments.md) fanno pulizia RMT della covarianza ma non calcolano/ritornano frazione di varianza spiegata da n autovettori | Codice custom: numpy.linalg.eigh su covariance_ di EmpiricalCovariance/EWCovariance, somma primi n autovalori / traccia |
| Turbulence & rischio sistemico | Rango effettivo erank(A) = exp(entropia di Shannon dei valori singolari normalizzati) (pannello F) | Nessun simbolo 'effective rank' in skfolio (verificato moments.md, distance_clustering.md, optimization.md, portfolio.md) | Codice custom: numpy.linalg.svd/eigh + entropia p_k log p_k, scipy per convenzione 0log0=0 |
| Turbulence & rischio sistemico | PCA rolling per ΔPC1(t) (quota varianza spiegata da PC1, finestra n/passo m) e participation ratio 1/I_k, I_k=Σω_ki⁴ (pannello G1, G3) | skfolio non ha un estimatore PCA/rolling-eigenvector generico esposto in API pubblica (verificato optimization.md, distance_clustering.md, moments.md); i pre-selection transformers non fanno PCA | Codice custom: sklearn.decomposition.PCA o numpy.linalg.eigh su finestre mobili di Σ, ricalcolo autovettori/autovalori per ogni finestra |
| Turbulence & rischio sistemico | Spettro autovalori vs bordo di Wishart/Marchenko-Pastur γ± = σ²(1+1/Q±2√(1/Q)) (pannello G2) | Nessuna funzione RMT-bound in skfolio; DenoiseCovariance/DetoneCovariance applicano RMT internamente per pulire la covarianza ma non espongono la curva di densità teorica ρ(γ) per confronto diagnostico | Codice custom: formula chiusa (numpy/scipy), nessuna libreria di terze parti necessaria oltre numpy |
| Turbulence & rischio sistemico | Motore di ricalcolo rolling che produce l'intera serie storica di d_t, AR, erank, PC1 su finestre scorrevoli (tutta la sezione C-H, non un singolo fit) | Gli estimatori skfolio (moments.md) fanno un solo fit su tutto X; WalkForward (model_selection) è pensato per backtest di ottimizzatori/CV, non per produrre serie temporali di statistiche diagnostiche ricalcolate finestra-per-finestra | Loop custom in Python che slitta la finestra (500gg) su X, richiama gli estimatori skfolio ad ogni passo e accumula d_t/AR/erank/PC1 in serie temporali |
| Universo & Dati | GARCH multivariato — varianti BEKK diagonale/scalare e DCC, stima a due passi (GARCH univariati poi correlazioni dinamiche), variance targeting, verifica delle condizioni di definitezza positiva di H_t (ω_i>0, α/β≥0, Σα+Σβ<1, autovalore minimo di R̄>0) | Verificato skfolio: la lista completa degli stimatori di covarianza in references/moments.md (EmpiricalCovariance, EWCovariance, LedoitWolf, OAS, ShrunkCovariance, DenoiseCovariance, DetoneCovariance, GerberCovariance, GraphicalLassoCV, ImpliedCovariance, RegimeAdjustedEWCovariance) non contiene nessun modello GARCH/BEKK/DCC; grep sull'intera skill per 'garch\|bekk\|dcc' non produce risultati. yfinance è solo un fornitore di dati, non fa stima statistica. | Costruire un motore DCC/BEKK custom (es. su base del pacchetto Python `arch` per il GARCH univariato per-serie, con secondo passo di correlazione dinamica implementato ad hoc: log-verosimiglianza a due stadi, standard error robusti di Bollerslev–Wooldridge, controllo definitezza positiva) oppure integrare librerie R (`rmgarch`/`ccgarch` via rpy2) o un fornitore commerciale di risk model multivariato. |
| Universo & Dati | Winsorizzazione a serie storica per singolo titolo con censura simmetrica a conteggio fisso i osservazioni per estremo e stimatore m_W a pesi doppi sugli estremi interni, con override per titolo e motivo in audit trail | Verificato skfolio: l'unico winsorizer in references/preprocessing.md è `CSWinsorizer(low, high)`, esplicitamente cross-sezionale (axis=1, tra titoli nella stessa osservazione) e basato su percentili, non su conteggio i di osservazioni censurate lungo il tempo per singolo titolo — contratto opposto a quanto richiesto dalla scheda 3. | Implementare in-house lo stimatore Winsor a campione censurato simmetrico (formula m_W del documento) come funzione custom su ciascuna serie storica per titolo, con tabella di override i per ticker e campo motivo persistito lato applicativo. |
| Universo & Dati | Shrinkage bayesiano del beta verso un prior di cross-sezione (mercato) o di settore — formula b''=(b'/s_b'²+b/s_b²)/(1/s_b'²+1/s_b²), varianti semplificate Merrill Lynch (k costante) e Bloomberg (2/3β+1/3), con 'precisione campionaria' per titolo | Verificato skfolio: nessuno stimatore di beta-shrinkage in moments.md, linear_model.md o altrove (grep 'beta shrink\|shrinkage_beta' nell'intera skill senza risultati). skfolio ha regressione cross-sezionale generica (CSLinearRegression) ma nessuna combinazione bayesiana beta grezzo/prior integrata. | Calcolare il beta grezzo per titolo (OLS semplice o skfolio.linear_model.CSLinearRegression) e implementare a mano la combinazione bayesiana/Merrill-Lynch/Bloomberg con la varianza cross-sezionale di settore o mercato come prior — codice custom lato backend. |
| Universo & Dati | Metrica esatta RMSE_v (errore quadratico medio di previsione della varianza, ε=r²−σ̂²) per la selezione del λ ottimo, come definita nel documento | Verificato skfolio: covariance_forecast_evaluation/CovarianceForecastComparison (references/online_learning.md) espongono qlike_loss e calibration ratios come diagnostiche native, non l'esatta formula RMSE_v a errore quadratico del documento — funzionalmente vicine ma non identiche. | Se serve fedeltà esatta alla formula del documento, calcolare RMSE_v con poche righe custom (loop su griglia half_life/λ, confronto r²_{t+1} vs σ̂²_{t+1\|t} da EWCovariance/EWVariance); altrimenti la metrica qlike di skfolio è un sostituto accettabile da documentare come scelta di prodotto. |
| Universo & Dati | '+ Aggiungi da indice' — importazione in blocco dei costituenti di un indice (es. FTSE MIB, S&P 500, STOXX Europe 600) | Verificato yfinance: le classi di screening in references/screener.md (EquityQuery/FundQuery/ETFQuery) espongono solo i campi Valuation/Price/Dividends/Financials/Growth/Classification(sector,industry,exchange,region)/Analyst — nessun campo di appartenenza a indice; nessun endpoint dedicato ai costituenti di un indice altrove nella skill. | Fonte dati esterna per le liste di costituenti (file mantenuto dal provider dell'indice, scraping controllato, o vendor a pagamento tipo Bloomberg/Refinitiv/ISS) da tenere aggiornata fuori da yfinance. |
| Universo & Dati | Import CSV con risoluzione ISIN→ticker per titoli non ancora presenti in universo (anteprima diff aggiunti/duplicati/non trovati) | Verificato yfinance: Ticker.isin (references/ticker.md) restituisce l'ISIN per un ticker già noto (lookup in avanti); yf.Search/yf.Lookup sono ricerca testuale fuzzy, non documentati come resolver ISIN→ticker affidabile. | Validare empiricamente se Search/Lookup accettano stringhe ISIN come query con esito accettabile; in caso contrario integrare un resolver dedicato (es. OpenFIGI API) per la fase di import CSV. |
| Universo & Dati | 'Escludi titoli con < N osservazioni' con soglia numerica N configurabile dall'utente | Verificato skfolio: SelectComplete (references/distance_clustering.md) è binario (storia completa sì/no nella finestra), senza parametro di soglia numerica N come richiesto dal controllo UI. | Piccola funzione custom lato applicativo (conteggio osservazioni non nulle per titolo, filtro contro N inserito dall'utente) — non serve libreria esterna, solo qualche riga di codice applicativo. |
| Views Builder | Per-view confidence input calibrated as an Ω diagonal element (three qualitative presets Bassa/Media/Alta plus numeric override), and Ω itself as an explicit editable K×K object. | Checked skfolio skill references (priors.md, PATTERNS.md, moments.md): `BlackLitterman(views=[...], tau=..., prior_estimator=...)` documents no `omega`/`view_confidences`/per-view-uncertainty parameter — only `views` (strings) and `tau`. | Custom code: build P/Q/Ω ourselves (parse view strings or construct P/Q directly, map slider level → variance or accept custom Ω), then subclass/wrap BlackLitterman to accept a precomputed Ω, or reimplement the Black-Litterman posterior formulas (r_BL, V_BL) directly using the domain doc's formulas rather than relying on skfolio's Ω-less documented surface. |
| Views Builder | Explicit numeric input of market-cap benchmark weights x_M into the equilibrium reverse-optimization (Π* = aVx_M). | Checked skfolio skill references (moments.md, PATTERNS.md): `EquilibriumMu` documents only a `risk_aversion` parameter; no documented way to pass a custom weight/benchmark vector at fit time. | Custom code: compute x_M from yfinance market caps and apply Π* = aVx_M directly (bypassing EquilibriumMu's undocumented internals), or confirm against the official (non-skill) skfolio API docs whether `EquilibriumMu.fit` accepts a weights argument before depending on it. |
| Views Builder | K ≤ N validation and live 'Riga P generata' preview as the user edits type/Asset A/Asset B in the editor. | Not a computation either library performs — checked skfolio (stateless estimator API, no view-list bookkeeping) and yfinance (market-data client only); this is UI/business-logic state, not a modeling function. | Plain backend/frontend logic: build the P row from selected assets, count K vs N — no external library needed. |
| Views Builder | Persistence of the (P, Q, Ω) view set tied to a portfolio, retrievable later by 'Obiettivo & Vincoli' and the solver run. | Checked both libraries: skfolio is a stateless scikit-learn-style estimator library (no storage layer) and yfinance is a market-data client (no storage layer) — neither persists application state. | Custom backend: a database table/model for portfolio views (P, Q, Ω, tau, a) with CRUD endpoints — ordinary application persistence, not a modeling gap. |

<!-- FINE GENERATO: capabilities -->

---

## 11. Idee confutate — NON costruire queste

Sono state indagate e scartate durante la ricerca. Restano registrate perché non vengano
resuscitate più avanti.

| Idea confutata | Verdetto | Cosa fare invece |
|---|---|---|
| Gate di verifica formale con theorem prover Lean su ogni operazione | confutata 0-3 | Usare uno strato di validazione **classico, basato su regole** |
| Determinismo e accuratezza sul compito sono statisticamente scorrelati | confutata 0-3 | Non trattarli come indipendenti |
| "Solo LangGraph e CrewAI si collegano a Ollama senza aggiramenti" | confutata 0-3 | Verificare da noi la connettività framework↔Ollama |
| Soglie di tool-use legate al numero di parametri (p.es. 7B≈71%, 32B≈82%) | confutata 0-3 | Non ancorare la scelta di taglia del modello a questi numeri |
| Smolagents è il migliore per modelli locali (tool call in codice contro JSON) | confutata 0-3 | Non sostanziato; non è una base per scegliere il framework |
| Esagonale = onion = clean = screaming, pienamente intercambiabili | confutata 1-2 | Stessa *famiglia*, non identici — non trattarli come uno solo |
| Trading 212 non ha un'API ordini ufficiale, solo automazione UI con Selenium | confutata 0-2 | L'API REST ordini ufficiale esiste; si costruisce su quella (§9), non su Selenium |
| Cron Paxos multi-replica / job store condiviso di APScheduler 4.x è la risposta di scala allo stato dell'arte | confutata 0-3 | Tenere un demone per database; elezione del leader con advisory lock solo se mai servisse (§5.1-§5.2) |
| asyncio ~3.5× più veloce dei thread per fetch I/O-bound | confutata 0-3 | Restare sincroni con thread; il guadagno è la concorrenza, già catturata (§5.6) |
| asyncpg ~3.4× di latenza in meno e ~1.9× di throughput contro SQLAlchemy | confutata 0-3 | Tenere SQLAlchemy 2.0 sincrono (§5.6) |

---

## 12. Decisioni prese e domande aperte

### Risolte (erano aperte alla prima passata)

- **Topologia degli agenti** — nodi blackboard in-process, non servizi, non A2A (§6.1, §6.2).
- **Comunicazione fra agenti** — un solo `FundState` tipizzato, orchestratore come mediatore,
  nessun import incrociato (§6.1).
- **Orchestratore** — scritto a mano sopra `FundState`; LangGraph adottato *solo* come substrato
  durevole di checkpoint e interrupt, con `FundState` come canale unico (§9).
- **Guardrail** — validatori Pydantic v2, basati su `ValueError`; le librerie di regole JSON sono
  solo dati (§9).
- **Confine LLM** — solo BAML; PydanticAI respinto (§9).
- **Broker** — nessuno per default; REST ufficiale di Trading 212 via `httpx` come riferimento,
  demo/live = scambio di base-URL e chiave (§9).
- **Trasporto** — comandi REST più SSE nativo di FastAPI per lo streaming (§9).
- **Postura di conformità** — consultiva e paper-first; l'esecuzione vera è tenuta spenta dietro
  il guardrail e il gate di approvazione umana (imposta come policy, non come limite del broker).
- **Solver convesso** — CVXPY + CLARABEL attraverso `skfolio` dentro `portopt`; lo stato terminale
  del solver è il verdetto; nessuna metaeuristica (§4).
- **Scheduler di raccolta** — tenere APScheduler in-process, un demone per database; l'elezione
  del leader con advisory lock è il percorso di scala futuro documentato, non costruito
  (§5.1-§5.2).
- **Reaper di raccolta** — passare a lease, e cambiare la gestione degli orfani da FAIL a RECLAIM
  con rimessa in coda — lo switch di reclaim è **gated** sull'audit degli upsert idempotenti
  (§5.3-§5.4).
- **I/O di raccolta** — restare su SQLAlchemy 2.0 sincrono con worker a thread; nessuna
  riscrittura async (§5.6).

### Ancora aperte — servono decisioni esplicite

**0. Audit degli upsert idempotenti (BLOCCA il reclaim di raccolta).** Prima che il reaper possa
fare RECLAIM degli orfani (§5.3), va verificata la scrittura di ogni job di fetch e vanno
convertiti gli insert semplici in `INSERT ... ON CONFLICT DO UPDATE` su chiavi naturali (§5.4).
Finché non è fatto, il reaper resta in modalità FAIL. L'angolo di ricerca 5 (watermarking, drift
di schema, incrementale contro backfill per yfinance e FRED) è rimasto coperto male e merita una
passata propria.

**1. Requisiti regolamentari concreti** (regole SEC per gli RIA, obblighi MiFID II sul trading
algoritmico, mandati di conservazione dei registri). La ricerca ha fatto emergere pattern ma
nessuna specificità verificata. Denaro vero ⇒ serve consulenza legale, non solo architettura.
Questo blocca lo *sblocco* dell'esecuzione, non la costruzione consultiva.

**2. Idempotenza e riconciliazione degli ordini Trading 212 (RIMANDATA al momento dello
sblocco).** L'API ordini è in BETA e **non idempotente**: un POST ritentato può piazzare un
ordine vero duplicato. Requisito documentato per l'adapter broker prima che l'esecuzione venga
sbloccata: chiave di idempotenza generata dal client per ogni intento d'ordine, controllo di
deduplicazione prima del POST, e polling dello stato ordine dopo l'invio per riconciliare.
Durante la fase consultiva non si scrive codice per ordini veri, quindi il disegno concreto è
rimandato — ma il requisito è registrato adesso perché interagisce con la trappola di idempotenza
al resume di §9 (un nodo di esecuzione interrotto riparte dal suo inizio) e con la
moltiplicazione in ordini figli di §6.4.

**3. Riproducibilità e determinismo per l'audit (RIMANDATA alla scelta del modello).** Se
impostare `temperature=0` più un `seed` fisso e registrare modello, parametri e hash del prompt
per ogni decisione dipende dal supporto a seed e temperatura del modello Ollama scelto, che varia
da modello a modello. Rimandata al momento della selezione. Fino ad allora: registrare modello,
parametri e output completo per ogni decisione, così l'audit funziona per ispezione anche senza
riproducibilità bit a bit, e tenere lo strato di validazione e retry alla port LLM (Ollama
garantisce token grammaticalmente validi solo al momento della generazione).

**4. Rivalidazione al momento della costruzione.** Diversi fatti di §9 stanno al cutoff di
gennaio 2026 o dopo (SSE nativo di FastAPI, versione di `sse-starlette`, rilascio degli ordini
live di Trading 212). Verificare versioni e percorsi di import quando l'implementazione comincia.

---

## 13. Qualità dell'evidenza e avvertenze

- I risultati sull'architettura poggiano principalmente su una fonte secondaria autorevole
  (*Architecture Patterns with Python*, Percival e Gregory — "Cosmic Python") più conferme
  canoniche (*Clean Architecture* di Robert C. Martin, l'esagonale di Cockburn). Forti per
  principi di disegno stabili e non sensibili al tempo.
- La **decisione sull'orchestratore** (scriverlo a mano sopra `FundState`; adottare LangGraph
  solo per checkpoint e interrupt) è un giudizio di sintesi costruito su fatti verificati sulle
  singole librerie più i vincoli fissati — non un'affermazione votata direttamente. La tensione
  residua vera è se il checkpointing Postgres collaudato di LangGraph valga l'accoppiamento
  rispetto a una tabella di audit e checkpoint append-only scritta a mano; abbiamo scelto
  LangGraph-come-substrato per l'affidabilità, in un sistema che un giorno muoverà denaro vero.
- I fatti sulle librerie (PydanticAI/Ollama, validatori Pydantic, ordini REST di Trading 212, SSE
  di FastAPI) sono ciascuno verificato individualmente 3-0 contro documentazione primaria o
  ufficiale.
- L'evidenza su determinismo e `pass^k` viene da un unico preprint 2026 non sottoposto a peer
  review, con benchmark piccoli e un risultato di correlazione internamente instabile. La
  sotto-affermazione su `pass^k` era spaccata 2-1 (il lavoro sulla riproducibilità è comunque
  rimandato, §12).
- Diversi fatti sui pacchetti stanno al cutoff di gennaio 2026 o dopo e sono stati stabiliti dalla
  passata di ricerca, non da conoscenza pregressa. **Rivalidare versioni, percorsi di import e il
  panorama in rapido movimento dei framework per agenti prima di impegnarsi** (§12, punto 4).
- Le **decisioni sul demone di raccolta (§5)** poggiano su fonti primarie o autorevoli (il libro
  SRE di Google, ACM Queue, la documentazione ufficiale di APScheduler 3.x, PostgreSQL e
  Kubernetes) — confidenza alta. Il dettaglio del reaper a lease poggia su un singolo changelog di
  progetto (haiku.rag) che conferma un pattern già standard. La raccomandazione sincrono contro
  asincrono è a confidenza media (fonti più deboli, blog di scraping) ma è rinforzata dal fatto
  che le affermazioni che avrebbero giustificato la migrazione sono state confutate 0-3.
- L'angolo di ricerca 5 sulla raccolta (upsert idempotenti, watermarking, drift di schema, outbox)
  è stato coperto male dall'evidenza verificata — quei principi vanno trattati come solidi ma non
  verificati; merita una passata propria (§12, punto 0).

*Sintetizzato da tre passate di ricerca a ventaglio: passata 1 (architettura) — 5 angoli, 25
fonti, 118 affermazioni, 25 verificate (16 confermate, 9 confutate); passata 2 (librerie) — 5
angoli, 23 fonti, 111 affermazioni, 25 verificate (24 confermate, 1 confutata); passata 3 (demone
di raccolta) — 5 angoli, fonti su documentazione primaria, 25 verificate (18 confermate, 7
confutate).*
