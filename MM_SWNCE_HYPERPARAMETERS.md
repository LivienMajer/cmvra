# MM_SWNCE Hyperparameter Übersicht

Diese Datei dokumentiert alle verfügbaren Hyperparameter für MM_SWNCE (Multi-Modal Soft-Weighted NCE) und wie sie in der Config gesetzt werden können.

## Verwendung

Alle Parameter können direkt in der Config-JSON gesetzt werden. Falls ein Parameter nicht angegeben wird, wird der Default-Wert verwendet.

```json
{
  "loss": "xid",
  "temperature": 0.1,
  "mm_swnce_mode": "wccss",
  "soft_mix": 0.5,
  "selfsim_mix": 0.5,
  ...
}
```

---

## Hyperparameter-Kategorien

### 1. KERN-PARAMETER

| Parameter | Typ | Default | Bereich | Beschreibung |
|-----------|-----|---------|---------|--------------|
| `temperature` | float | 0.1 | 0.04-0.2 | Haupttemperatur für InfoNCE Softmax. Niedrigere Werte = schärfere Verteilungen |

**Config-Beispiel:**
```json
{
  "temperature": 0.07,
  "temp_anchor_start": 0.04,
  "temp_anchor_end": 0.07
}
```

---

### 2. FEATURE-AKTIVIERUNG

Die Features werden primär über `mm_swnce_mode` gesteuert, können aber auch manuell überschrieben werden.

#### Modi (mm_swnce_mode)

| Mode | use_weighting | use_soft_targets | use_self_similarity | Beschreibung |
|------|--------------|------------------|---------------------|--------------|
| `w` | ✓ | ✗ | ✗ | Nur Weighting (gegen faulty positives) |
| `ss` | ✗ | ✗ | ✓ | Nur Self-Similarity (intra-modale Konsistenz) |
| `cc` | ✗ | ✓ | ✗ | Nur Cycle-Consistency (gegen faulty negatives) |
| `wcc` | ✓ | ✓ | ✗ | Weighting + Cycle-Consistency (beide Error-Typen) |
| `wss` | ✓ | ✗ | ✓ | Weighting + Self-Similarity (Robustheit + Struktur) |
| `sscc` | ✗ | ✓ | ✓ | Self-Similarity + Cycle-Consistency |
| `wccss` | ✓ | ✓ | ✓ | **Alle Features** (empfohlen für beste Performance) |

**Manuelle Überschreibung:**
```json
{
  "mm_swnce_mode": "wcc",
  "use_weighting": true,
  "use_soft_targets": false,
  "use_self_similarity": true
}
```

---

### 3. SOFT TARGETS (Cycle-Consistency)

Behandelt faulty negatives durch weiche Zielvorgaben.

| Parameter | Typ | Default | Bereich | Beschreibung |
|-----------|-----|---------|---------|--------------|
| `soft_target_mode` | str | "cycle" | cycle/swapped | Modus für Soft-Target-Berechnung |
| `soft_mix` | float | 0.5 | 0.0-1.0 | **Wichtig!** Anteil Soft-Targets vs Identity. 0=pure hard, 1=pure soft (identity = standard infoNCE)
| `tau_s` | float | 0.02 | 0.01-0.05 | Temperatur für Source-Modalität (niedrig = scharf) |
| `tau_t` | float | 0.07 | 0.05-0.1 | Temperatur für Target-Modalität |

**Empfehlungen:**
- **Conservative:** `soft_mix=0.3` (70% Identity, 30% Soft)
- **Balanced:** `soft_mix=0.5` (50/50)
- **Aggressive:** `soft_mix=0.7` (30% Identity, 70% Soft)

**Config-Beispiel:**
```json
{
  "soft_target_mode": "cycle",
  "soft_mix": 0.5,
  "tau_s": 0.02,
  "tau_t": 0.07
}
```

---

### 4. SELF-SIMILARITY

Nutzt intra-modale Struktur zur Verbesserung der Targets.

| Parameter | Typ | Default | Bereich | Beschreibung |
|-----------|-----|---------|---------|--------------|
| `selfsim_mix` | float | 0.5 | 0.0-1.0 | Anteil Self-Similarity vs Cycle-Consistency. 0=nur Cycle, 1=nur SelfSim |
| `tau_self_i2j` | float | 0.07 | 0.05-0.1 | Temperatur für i→j Self-Similarity (basierend auf Target-Modalität) |
| `tau_self_j2i` | float | 0.07 | 0.05-0.1 | Temperatur für j→i Self-Similarity (basierend auf Source-Modalität) |

**Empfehlungen:**
- Gleichgewicht: `selfsim_mix=0.5` für ausgewogene Nutzung beider Informationen
- Nur bei aktivem `use_self_similarity` relevant

**Config-Beispiel:**
```json
{
  "selfsim_mix": 0.5,
  "tau_self_i2j": 0.07,
  "tau_self_j2i": 0.07
}
```

---

### 5. WEIGHTING

Robuste Gewichtung von Samples basierend auf positiver Similarity (gegen faulty positives).

| Parameter | Typ | Default | Bereich | Beschreibung |
|-----------|-----|---------|---------|--------------|
| `weighting_delta` | float | 0.0 | 0.0-0.5 | Shift für Robust-CDF. Positiv = mehr Gewicht auf niedrige Similarities |
| `weighting_kappa` | float | 0.5 | 0.3-0.7 | Spread-Faktor für CDF. Niedriger = schärfere Gewichtung |
| `weighting_w_min` | float | 0.25 | 0.1-0.5 | Minimales Gewicht pro Sample (verhindert komplettes Ignorieren) |

**Mechanismus:**
- Berechnet Gewicht `w_i` pro Sample basierend auf positiver Similarity
- Niedrige Similarity (vermutlich faulty positive) → niedrigeres Gewicht
- Nutzt CDF mit EMA-geglätteten Statistiken

**Config-Beispiel:**
```json
{
  "weighting_delta": 0.0,
  "weighting_kappa": 0.5,
  "weighting_w_min": 0.25
}
```

---

### 6. TRAINING-DYNAMIK

Steuert die zeitliche Entwicklung der Loss-Komponenten während des Trainings.

#### Warmup

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `warmup_epochs` | int | 0 | Anzahl Epochen mit Standard-InfoNCE vor Feature-Aktivierung |

**Zweck:** Stabilisiert Training durch anfängliches einfaches Ziel (pure InfoNCE), bevor komplexe Features aktiviert werden.

#### Soft-Mix Scheduling (Curriculum Learning)

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `soft_mix_config.schedule` | bool | false | Scheduling für soft_mix aktivieren |
| `soft_mix_config.initial` | float | 0.1 | Start-Wert für soft_mix (niedrig = mehr Identity) |
| `soft_mix_config.final` | float | 0.9 | End-Wert für soft_mix (hoch = mehr Soft Targets) |
| `soft_mix_config.schedule_type` | str | "cosine" | linear/cosine/exponential |

**Curriculum Learning-Philosophie:**
- **Start (initial=0.1):** 90% Identity (einfaches Ziel: exakte Matches lernen)
- **Ende (final=0.9):** 90% Soft Targets (schwieriges Ziel: Struktur lernen)
- Gradueller Übergang für stabiles Training

**Config-Beispiel:**
```json
{
  "warmup_epochs": 5,
  "soft_mix_config": {
    "schedule": true,
    "initial": 0.1,
    "final": 0.9,
    "schedule_type": "cosine"
  }
}
```

---

### 7. TECHNISCHE PARAMETER

| Parameter | Typ | Default | Bereich | Beschreibung |
|-----------|-----|---------|---------|--------------|
| `eps` | float | 1e-6 | 1e-8 - 1e-5 | Numerische Stabilisierung (Division-by-Zero Prevention) |

---


## Logging

Beim Training werden automatisch alle aktiven Hyperparameter geloggt:

```
=== MM_SWNCE Hyperparameter Configuration ===
Mode: wccss
Temperature (main): 0.1
Use Weighting: True
Use Soft Targets: True
Use Self-Similarity: True
Soft Mix: 0.5 (Identity: 0.50)
Selfsim Mix: 0.5
Tau S/T: 0.02/0.07
Tau Self i2j/j2i: 0.07/0.07
Weighting: delta=0.0, kappa=0.5, w_min=0.25
Warmup Epochs: 5
Soft Mix Scheduling: 0.1 -> 0.9 (cosine)
=============================================
```

---

## Implementierung

Die Hyperparameter werden zentral durch die Funktion `get_mm_swnce_hyperparameters(config)` in `loss.py` verwaltet. Diese Funktion:

1. Definiert Default-Werte für alle Parameter
2. Liest Werte aus der Config
3. Überschreibt Defaults mit Config-Werten
4. Gibt ein vollständiges Parameter-Dictionary zurück

**Code-Beispiel:**
```python
from zeta.loss import get_mm_swnce_hyperparameters

# Lädt alle Parameter aus Config mit Fallback auf Defaults
params = get_mm_swnce_hyperparameters(config)

# Initialisiere Loss mit allen Parametern
loss = MM_SWNCE(
    temperature=params['temperature'],
    use_weighting=params['use_weighting'],
    # ... alle anderen Parameter
)
```


**Erstellt:** 2026-01-27  
