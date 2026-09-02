-- ===========================================================================
-- Promptanalyse, Datenbankschema (SQLite)
--
-- Haelt den vollstaendigen Bestand beider Teilstudien. Ersetzt die frueheren
-- Ordner 0_questions, 1_prompts, 2_redundancy und 3_compressed.
--
-- Ebenen:
--   frage         Aufgabe aus PopQA, Grundeinheit des Designs
--   demonstration acht fixierte Demonstrationsslots je Aufgabe
--   paraphrase    Material der Redundanzstufe semantic aus PopQA-TP
--   prompt        Bedingung vor Kompression, also Aufgabe x Variante
--   segment       Auszeichnung von Kern und Redundanz nach Konstruktion
--   wort          Wortebene des Prompts, Grundlage der Erhaltungsraten
--   kompressat    Bedingung nach Kompression, also prompt x Reduktionsstufe
--   wort_erhalt   je Wort und Kompressat, ob es erhalten blieb
--   inferenz      Modelllauf der Teilstudie B
--   lauf          Provenienz jedes Verarbeitungsschritts
--
-- Namenskonvention: Bezeichner deutsch, Variantennamen englisch wie im
-- bisherigen Dateibestand (basis, semantic, instruction, demonstration,
-- filler).
-- ===========================================================================

PRAGMA foreign_keys = ON;


-- ---------------------------------------------------------------------------
-- Provenienz
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lauf (
    lauf_id         INTEGER PRIMARY KEY,
    schritt         TEXT    NOT NULL,   -- init|prompts|redundanz|kompression|
                                        -- selektivitaet|inferenz|export
    gestartet       TEXT    NOT NULL,
    beendet         TEXT,
    skript          TEXT,
    skript_sha256   TEXT,
    konfiguration   TEXT,               -- JSON, vollstaendiger Konfigblock
    git_commit      TEXT,
    python_version  TEXT,
    plattform       TEXT,
    n_zeilen        INTEGER,
    bemerkung       TEXT
);


-- ---------------------------------------------------------------------------
-- Aufgaben
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS frage (
    frage_id         TEXT    PRIMARY KEY,   -- PopQA-ID
    in_teilstudie_a  INTEGER NOT NULL DEFAULT 1,
    in_teilstudie_b  INTEGER NOT NULL DEFAULT 0,
    relation         TEXT    NOT NULL,      -- prop, Relationskategorie
    relation_id      TEXT,
    subjekt          TEXT,
    objekt           TEXT,
    s_pop            INTEGER,               -- Wikipedia-Aufrufe des Subjekts
    pop_terzil       INTEGER,               -- 1..3, innerhalb der Kategorie
    fragetext        TEXT    NOT NULL,
    antworten        TEXT    NOT NULL,      -- JSON-Liste der Aliasformen
    n_antwortformen  INTEGER,
    laenge_goldantwort_ws INTEGER,          -- Woerter der kuerzesten Aliasform
    n_paraphrasen    INTEGER,
    n_ws_token_frage INTEGER,
    lauf_id          INTEGER REFERENCES lauf(lauf_id)
);

CREATE INDEX IF NOT EXISTS ix_frage_relation ON frage(relation);
CREATE INDEX IF NOT EXISTS ix_frage_terzil   ON frage(pop_terzil);
CREATE INDEX IF NOT EXISTS ix_frage_b        ON frage(in_teilstudie_b);


CREATE TABLE IF NOT EXISTS demonstration (
    frage_id      TEXT    NOT NULL REFERENCES frage(frage_id) ON DELETE CASCADE,
    slot          INTEGER NOT NULL,         -- 1..8
    rolle         TEXT    NOT NULL,         -- base (1..4) | demonstration (5..8)
    demo_id       TEXT    NOT NULL,         -- PopQA-ID der Demonstration
    demo_relation TEXT    NOT NULL,
    demo_frage    TEXT    NOT NULL,
    demo_antwort  TEXT    NOT NULL,
    PRIMARY KEY (frage_id, slot)
);

CREATE INDEX IF NOT EXISTS ix_demo_id ON demonstration(demo_id);


CREATE TABLE IF NOT EXISTS paraphrase (
    quelle_id               TEXT    NOT NULL,  -- PopQA-ID der Ursprungsfrage
    template_id             INTEGER NOT NULL,  -- 0 = Originalformulierung
    text                    TEXT    NOT NULL,
    kodierung_repariert     INTEGER NOT NULL DEFAULT 0,
    identisch_zum_original  INTEGER NOT NULL DEFAULT 0,
    verwendbar              INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (quelle_id, template_id)
);


-- ---------------------------------------------------------------------------
-- Prompts vor Kompression
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt (
    prompt_id            INTEGER PRIMARY KEY,
    frage_id             TEXT    NOT NULL REFERENCES frage(frage_id) ON DELETE CASCADE,
    variante             TEXT    NOT NULL,  -- basis|semantic|instruction|
                                            -- demonstration|filler
    studie               TEXT    NOT NULL DEFAULT 'A',
    text                 TEXT    NOT NULL,
    sha256               TEXT    NOT NULL,

    -- Laengen
    n_zeichen            INTEGER,
    n_ws_token           INTEGER,   -- Whitespace-Woerter, Ratenparameter
    n_modell_token       INTEGER,   -- Tokenizer des Zielmodells, Kosten
    n_ws_token_kern      INTEGER,
    n_ws_token_red       INTEGER,
    n_modell_token_kern  INTEGER,
    n_modell_token_red   INTEGER,
    n_demos              INTEGER,

    -- Kalibrierung gegen den Basisprompt derselben Aufgabe (eq:kalibrierung)
    n_ws_token_basis     INTEGER,
    n_modell_token_basis INTEGER,
    kalibrierung_ws      REAL,      -- L_red / L_basis in Woertern
    kalibrierung_token   REAL,      -- L_red / L_basis in Modelltoken, massgeblich
    kalibrierung_im_band INTEGER,   -- 1, wenn |x - Ziel| <= Band
    kalibrierung_abweichung REAL,   -- |x - Ziel|, fuer die Berichterstattung
    -- Welche Teilmenge der Einfuegeeinheiten die Kalibrierung gewaehlt hat.
    -- Erst damit ist die Konstruktion einer Bedingung nachvollziehbar.
    bau_parameter        TEXT,      -- JSON, siehe promptbau.kandidaten
    n_einheiten          INTEGER,   -- Zahl der eingefuegten Einheiten

    -- Kontrollgroessen der Konstruktion
    lex_overlap_red_kern REAL,      -- V13, Anteil Wortformen der Redundanz,
                                    -- die auch im Kern vorkommen
    antwortleckage       INTEGER,   -- V14, Goldantwort ausserhalb des Frageblocks
    leckage_abschnitt    TEXT,

    erzeugt              TEXT,
    lauf_id              INTEGER REFERENCES lauf(lauf_id),
    UNIQUE (frage_id, variante, studie)
);

CREATE INDEX IF NOT EXISTS ix_prompt_variante ON prompt(variante);
CREATE INDEX IF NOT EXISTS ix_prompt_frage    ON prompt(frage_id);


CREATE TABLE IF NOT EXISTS segment (
    segment_id     INTEGER PRIMARY KEY,
    prompt_id      INTEGER NOT NULL REFERENCES prompt(prompt_id) ON DELETE CASCADE,
    ord            INTEGER NOT NULL,
    label          TEXT    NOT NULL,   -- instruktion, demonstration_3, filler ...
    herkunft       TEXT    NOT NULL,   -- kern | redundanz
    abschnitt      TEXT    NOT NULL,   -- instruktion | demonstration |
                                       -- testfrage | filler
    slot           INTEGER,
    start_zeichen  INTEGER NOT NULL,
    ende_zeichen   INTEGER NOT NULL,
    start_wort     INTEGER,
    ende_wort      INTEGER,
    n_ws_token     INTEGER,
    UNIQUE (prompt_id, ord)
);

CREATE INDEX IF NOT EXISTS ix_segment_prompt ON segment(prompt_id);


-- Wortebene. Eine Zeile je Token des Gesamtprompts. Traegt die Auszeichnung,
-- gegen die nach der Kompression abgeglichen wird.
CREATE TABLE IF NOT EXISTS wort (
    wort_id     INTEGER PRIMARY KEY,
    prompt_id   INTEGER NOT NULL REFERENCES prompt(prompt_id) ON DELETE CASCADE,
    pos         INTEGER NOT NULL,   -- Index in der Tokenfolge
    rel_pos     REAL,               -- pos / (n - 1), fuer Positionseffekte
    form        TEXT    NOT NULL,
    schluessel  TEXT    NOT NULL,   -- casefold, Abgleichsschluessel
    ist_wort    INTEGER NOT NULL,   -- 0 fuer reine Satzzeichen
    ist_ziffer  INTEGER NOT NULL,
    herkunft    TEXT    NOT NULL,   -- kern | redundanz
    abschnitt   TEXT    NOT NULL,
    segment_id  INTEGER REFERENCES segment(segment_id),
    UNIQUE (prompt_id, pos)
);

CREATE INDEX IF NOT EXISTS ix_wort_prompt ON wort(prompt_id);


-- ---------------------------------------------------------------------------
-- Kompressate
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kompressat (
    kompressat_id   INTEGER PRIMARY KEY,
    prompt_id       INTEGER NOT NULL REFERENCES prompt(prompt_id) ON DELETE CASCADE,
    rho_ziel        REAL    NOT NULL,   -- angezielte Reduktionsrate
    rate_parameter  REAL    NOT NULL,   -- t = 1 - rho, Parameter des Kompressors
    text            TEXT    NOT NULL,
    sha256          TEXT,

    -- Ratenkonformitaet (eq:istrate)
    n_ws_token      INTEGER,
    n_modell_token  INTEGER,
    rho_ist_wort    REAL,
    rho_ist_token   REAL,
    rho_delta_wort  REAL,
    rho_delta_token REAL,

    -- Selbstauskunft des Kompressors
    komp_origin_tokens     INTEGER,
    komp_compressed_tokens INTEGER,
    komp_rate_ist          REAL,
    dauer_kompression_s    REAL,
    kompressor_modell      TEXT,
    geraet                 TEXT,

    -- Selektivitaet (eq:erhaltungsraten, eq:selektivitaet)
    n_kern          INTEGER,
    n_red           INTEGER,
    n_gesamt        INTEGER,
    redundanzanteil REAL,
    r_kern          REAL,
    r_red           REAL,
    s_index         REAL,          -- R_kern / R_red nach eq:selektivitaet
    ln_s            REAL,
    r_red_null      INTEGER,       -- 1, wenn R_red = 0 und s_index undefiniert

    -- Pruefgroesse der Arbeit (eq:selektivitaetfrei).
    --
    -- Gerechnet wird ueber die freien Woerter, also ueber die, bei denen der
    -- Kompressor ueberhaupt eine Wahl hatte. Geschuetzte Marken und Ziffern
    -- fallen aus Zaehler und Nenner beider Raten heraus. Das ist nicht
    -- kosmetisch: die Marken Q und A liegen zu sechzehn Prozent im Kern, aber
    -- zu null Prozent in der Redundanz der Stufen filler und instruction und
    -- zu neunzehn Prozent in der von demonstration. Ein Verfahren, das die
    -- Herkunft vollstaendig ignoriert, erzeugt allein daraus bei rho = 0,75
    -- einen Index von rund 1,8 fuer filler und 0,9 fuer demonstration, also
    -- genau die in H2 postulierte Rangfolge. Ueber die freien Woerter ist der
    -- Index eines indifferenten Verfahrens exakt 1.
    --
    -- Zusaetzlich ist die Stetigkeitskorrektur nach Haldane und Anscombe
    -- eingebaut (0,5 auf jeden Zaehler, 1 auf jeden Nenner), damit auch
    -- Bedingungen mit vollstaendig entfernter Redundanz definiert bleiben.
    -- Sie auszuschliessen wuerde die Faelle maximaler Selektivitaet entfernen
    -- und den Index systematisch nach unten verzerren.
    r_kern_frei     REAL,
    r_red_frei      REAL,
    n_kern_frei     INTEGER,
    n_red_frei      INTEGER,
    s_frei          REAL,
    ln_s_frei       REAL,

    -- Robustheit der Auszeichnung
    r_kern_spiegel  REAL,
    r_red_spiegel   REAL,
    s_spiegel       REAL,
    n_mehrdeutig    INTEGER,
    anteil_mehrdeutig REAL,
    s_intervall_offen INTEGER,   -- 1, wenn S und S_spiegel abweichen
    basis_ist_teilfolge   INTEGER,
    n_basis_unzugeordnet  INTEGER,
    n_komp_unzugeordnet   INTEGER,

    -- Strukturintegritaet nach der Kompression
    n_zeilenumbrueche  INTEGER,
    anteil_marken_erhalten REAL,   -- Q: und A:
    anteil_ziffern_erhalten REAL,

    erzeugt   TEXT,
    lauf_id   INTEGER REFERENCES lauf(lauf_id),
    UNIQUE (prompt_id, rho_ziel)
);

CREATE INDEX IF NOT EXISTS ix_komp_prompt ON kompressat(prompt_id);
CREATE INDEX IF NOT EXISTS ix_komp_rho    ON kompressat(rho_ziel);


-- Erhaltungsraten je Promptabschnitt und Herkunft. Grundlage der in
-- subsubsec:selektivitaet geforderten Aufschluesselung.
CREATE TABLE IF NOT EXISTS abschnitt_erhalt (
    kompressat_id  INTEGER NOT NULL REFERENCES kompressat(kompressat_id) ON DELETE CASCADE,
    abschnitt      TEXT    NOT NULL,
    herkunft       TEXT    NOT NULL,
    n_woerter      INTEGER NOT NULL,
    n_erhalten     INTEGER NOT NULL,
    erhaltungsrate REAL,
    PRIMARY KEY (kompressat_id, abschnitt, herkunft)
);


-- Feinste Ebene. Optional befuellbar, ermoeglicht Modelle auf Wortebene.
CREATE TABLE IF NOT EXISTS wort_erhalt (
    kompressat_id    INTEGER NOT NULL REFERENCES kompressat(kompressat_id) ON DELETE CASCADE,
    wort_id          INTEGER NOT NULL REFERENCES wort(wort_id) ON DELETE CASCADE,
    erhalten         INTEGER NOT NULL,
    erhalten_spiegel INTEGER,
    PRIMARY KEY (kompressat_id, wort_id)
);

CREATE INDEX IF NOT EXISTS ix_worterhalt_wort ON wort_erhalt(wort_id);


-- ---------------------------------------------------------------------------
-- Teilstudie B
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inferenz (
    inferenz_id   INTEGER PRIMARY KEY,
    kompressat_id INTEGER NOT NULL REFERENCES kompressat(kompressat_id) ON DELETE CASCADE,
    wiederholung  INTEGER NOT NULL DEFAULT 1,   -- >1 nur fuer die Determinismuspruefung

    ausgabe               TEXT,
    ausgabe_normalisiert  TEXT,
    korrekt               INTEGER,
    getroffener_alias     TEXT,
    fehlerklasse          TEXT,   -- falsche_entitaet | formatverstoss |
                                  -- verweigerung | leer_abgeschnitten
    fehlerklasse_manuell  TEXT,
    abgeschnitten         INTEGER,

    -- Effizienz (subsubsec:effizienzkennzahlen)
    n_eingabe_token   INTEGER,
    n_ausgabe_token   INTEGER,
    dauer_prefill_s   REAL,
    dauer_dekodierung_s REAL,
    dauer_gesamt_s    REAL,
    energie_j         REAL,

    -- Konfiguration
    modell        TEXT,
    quantisierung TEXT,
    temperatur    REAL,
    top_p         REAL,
    seed          INTEGER,
    max_new_tokens INTEGER,
    laufzeitumgebung TEXT,
    hardware      TEXT,

    erzeugt  TEXT,
    lauf_id  INTEGER REFERENCES lauf(lauf_id),
    UNIQUE (kompressat_id, wiederholung)
);

CREATE INDEX IF NOT EXISTS ix_inferenz_komp ON inferenz(kompressat_id);


-- ===========================================================================
-- Auswertungssichten
-- ===========================================================================

-- Flache Tabelle aller Bedingungen der Teilstudie A. Eine Zeile je
-- Aufgabe x Variante x Reduktionsstufe. Direkte Eingabe fuer R oder pandas.
DROP VIEW IF EXISTS v_bedingung;
CREATE VIEW v_bedingung AS
SELECT
    k.kompressat_id,
    p.prompt_id,
    f.frage_id,
    f.relation,
    f.pop_terzil,
    f.s_pop,
    f.in_teilstudie_b,
    p.variante,
    p.studie,
    k.rho_ziel,
    k.rho_ist_wort,
    k.rho_ist_token,
    k.rho_delta_token,
    p.n_ws_token           AS n_ws_token_prompt,
    p.n_modell_token       AS n_modell_token_prompt,
    p.n_modell_token_basis,
    p.kalibrierung_token,
    p.kalibrierung_im_band,
    p.kalibrierung_abweichung,
    p.n_einheiten,
    p.bau_parameter,
    p.lex_overlap_red_kern,
    p.antwortleckage,
    k.n_kern, k.n_red, k.redundanzanteil,
    k.r_kern, k.r_red, k.s_index, k.ln_s, k.r_red_null,
    k.r_kern_frei, k.r_red_frei, k.n_kern_frei, k.n_red_frei,
    k.s_frei, k.ln_s_frei,
    k.s_intervall_offen, k.anteil_mehrdeutig,
    k.basis_ist_teilfolge, k.n_komp_unzugeordnet,
    k.anteil_marken_erhalten, k.anteil_ziffern_erhalten,
    k.dauer_kompression_s
FROM kompressat k
JOIN prompt p ON p.prompt_id = k.prompt_id
JOIN frage  f ON f.frage_id  = p.frage_id;


-- Teilstudie B: Bedingung plus Modelllauf.
DROP VIEW IF EXISTS v_studie_b;
CREATE VIEW v_studie_b AS
SELECT
    b.*,
    i.inferenz_id,
    i.wiederholung,
    i.ausgabe,
    i.korrekt,
    i.fehlerklasse,
    i.abgeschnitten,
    i.n_eingabe_token,
    i.n_ausgabe_token,
    i.dauer_gesamt_s,
    i.dauer_gesamt_s + COALESCE(b.dauer_kompression_s, 0) AS dauer_pipeline_s
FROM v_bedingung b
JOIN inferenz i ON i.kompressat_id = b.kompressat_id;


-- Ratenkonformitaet je Variante und Stufe (V3).
DROP VIEW IF EXISTS v_ratenkonformitaet;
CREATE VIEW v_ratenkonformitaet AS
SELECT variante, rho_ziel,
       COUNT(*)               AS n,
       AVG(rho_ist_wort)      AS rho_ist_wort_mittel,
       AVG(rho_ist_token)     AS rho_ist_token_mittel,
       AVG(rho_delta_token)   AS rho_delta_token_mittel,
       MIN(rho_ist_token)     AS rho_ist_token_min,
       MAX(rho_ist_token)     AS rho_ist_token_max
FROM v_bedingung
WHERE rho_ziel > 0
GROUP BY variante, rho_ziel;


-- Selektivitaet je Variante und Stufe. S wird geometrisch gemittelt, da ein
-- Verhaeltnis nicht arithmetisch gemittelt werden darf.
DROP VIEW IF EXISTS v_selektivitaet;
CREATE VIEW v_selektivitaet AS
-- Die Variante basis enthaelt keine eingefuegte Redundanz. R_red und damit
-- S sind fuer sie nicht definiert; sie ist Referenz fuer Laenge und
-- Aufgabenleistung, nicht fuer die Selektivitaet.
SELECT variante, rho_ziel,
       COUNT(*)                            AS n,
       -- Pruefgroesse
       AVG(ln_s_frei)                      AS ln_s_frei_mittel,
       EXP(AVG(ln_s_frei))                 AS s_frei_geometrisch,
       AVG(r_kern_frei)                    AS r_kern_frei_mittel,
       AVG(r_red_frei)                     AS r_red_frei_mittel,
       -- deskriptiv, entspricht eq:selektivitaet
       AVG(r_kern)                         AS r_kern_mittel,
       AVG(r_red)                          AS r_red_mittel,
       EXP(AVG(ln_s))                      AS s_geometrisch,
       SUM(COALESCE(r_red_null, 0))        AS n_r_red_null,
       -- Kontrollen der Auszeichnung und des Strukturschutzes
       SUM(s_intervall_offen)              AS n_intervall_offen,
       SUM(CASE WHEN basis_ist_teilfolge = 0 THEN 1 ELSE 0 END) AS n_keine_teilfolge,
       AVG(anteil_marken_erhalten)         AS marken_erhalten_mittel,
       AVG(anteil_ziffern_erhalten)        AS ziffern_erhalten_mittel
FROM v_bedingung
WHERE rho_ziel > 0 AND n_red > 0
GROUP BY variante, rho_ziel;


-- Genauigkeit je Variante und Stufe, Grundlage der Kompressionstoleranz.
DROP VIEW IF EXISTS v_genauigkeit;
CREATE VIEW v_genauigkeit AS
SELECT variante, rho_ziel,
       COUNT(*)          AS n,
       AVG(korrekt)      AS genauigkeit,
       AVG(dauer_gesamt_s)   AS dauer_inferenz_mittel,
       AVG(dauer_pipeline_s) AS dauer_pipeline_mittel,
       AVG(n_eingabe_token)  AS n_eingabe_token_mittel
FROM v_studie_b
WHERE wiederholung = 1
GROUP BY variante, rho_ziel;


-- Kalibrierungsnachweis je Variante (V2, eq:kalibrierung).
DROP VIEW IF EXISTS v_kalibrierung;
CREATE VIEW v_kalibrierung AS
SELECT variante,
       COUNT(*)                        AS n,
       AVG(n_einheiten)                AS n_einheiten_mittel,
       AVG(kalibrierung_token)         AS quote_token_mittel,
       MIN(kalibrierung_token)         AS quote_token_min,
       MAX(kalibrierung_token)         AS quote_token_max,
       AVG(kalibrierung_abweichung)    AS abweichung_mittel,
       MAX(kalibrierung_abweichung)    AS abweichung_max,
       AVG(kalibrierung_im_band)       AS anteil_im_band,
       AVG(kalibrierung_ws)            AS quote_ws_mittel,
       AVG(lex_overlap_red_kern)       AS lex_overlap_mittel,
       AVG(antwortleckage)             AS anteil_antwortleckage
FROM prompt
WHERE variante <> 'basis'
GROUP BY variante;
