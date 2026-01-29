import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import torch.distributed as dist
from collections import deque


def get_mm_swnce_config(mode: str) -> dict:
    """
    Definiert MM_SWNCE (Multi-Modal Soft-Weighted NCE) Varianten basierend auf Modi.
    
    Args:
        mode (str): Einer von 'w', 'ss', 'cc', 'wcc', 'wss', 'sscc', 'wccss'
            - 'w':     Weighting only (faulty positives)
            - 'ss':    Self-Similarity only (intra-modal consistency)
            - 'cc':    Cycle-Consistency only (faulty negatives)
            - 'wcc':   Weighting + Cycle-Consistency (both error types)
            - 'wss':   Weighting + Self-Similarity (robustness + structure)
            - 'sscc':  Self-Similarity + Cycle-Consistency (structure + robustness)
            - 'wccss': Weighting + Cycle-Consistency + Self-Similarity (alle Features)
    
    Returns:
        dict: Konfiguration mit (use_weighting, use_soft_targets, use_self_similarity)
    
    Raises:
        ValueError: Wenn mode nicht erkannt wird
    """
    CONFIGS = {
        'w':     {'use_weighting': True,  'use_soft_targets': False, 'use_self_similarity': False},
        'ss':    {'use_weighting': False, 'use_soft_targets': False, 'use_self_similarity': True},
        'cc':    {'use_weighting': False, 'use_soft_targets': True,  'use_self_similarity': False},
        'wcc':   {'use_weighting': True,  'use_soft_targets': True,  'use_self_similarity': False},
        'wss':   {'use_weighting': True,  'use_soft_targets': False, 'use_self_similarity': True},
        'sscc':  {'use_weighting': False, 'use_soft_targets': True,  'use_self_similarity': True},
        'wccss': {'use_weighting': True,  'use_soft_targets': True,  'use_self_similarity': True},
    }
    
    if mode not in CONFIGS:
        raise ValueError(f"Unknown MM_SWNCE mode '{mode}'. Choose from: {list(CONFIGS.keys())}")
    
    return CONFIGS[mode]


def get_mm_swnce_hyperparameters(config: dict) -> dict:
    """
    Zentrale Funktion zum Auslesen aller MM_SWNCE Hyperparameter aus der Config.
    
    Diese Funktion sammelt alle Hyperparameter mit Default-Werten und überschreibt sie
    mit Werten aus der Config, falls vorhanden. Dies ermöglicht vollständige Kontrolle
    über alle Parameter via Config-File.
    
    Args:
        config (dict): Konfigurationsdictionary
    
    Returns:
        dict: Dictionary mit allen Hyperparametern für MM_SWNCE
        
    Hyperparameter-Übersicht:
    -------------------------
    
    KERN-PARAMETER:
    - temperature (float): Haupttemperatur für InfoNCE Softmax [0.04-0.1]
    
    FEATURE-AKTIVIERUNG (aus mm_swnce_mode):
    - use_weighting (bool): Robuste Gewichtung gegen faulty positives
    - use_soft_targets (bool): Cycle-Consistency gegen faulty negatives
    - use_self_similarity (bool): Intra-modale Konsistenz
    
    SOFT TARGETS (Cycle-Consistency):
    - soft_target_mode (str): 'cycle' oder 'swapped'
    - soft_mix (float): Anteil Soft-Targets vs Identity [0-1]
    - tau_s (float): Temperatur für Source-Modalität [0.01-0.05]
    - tau_t (float): Temperatur für Target-Modalität [0.05-0.1]
    
    SELF-SIMILARITY:
    - selfsim_mix (float): Anteil Self-Sim vs Cycle [0-1]
    - tau_self_i2j (float): Temperatur für i->j Self-Similarity [0.05-0.1]
    - tau_self_j2i (float): Temperatur für j->i Self-Similarity [0.05-0.1]
    
    WEIGHTING:
    - weighting_params (dict):
        - delta (float): Shift für Robust-CDF [0.0-0.5]
        - kappa (float): Spread-Faktor [0.3-0.7]
        - w_min (float): Minimales Gewicht [0.1-0.5]
    
    TRAINING-DYNAMIK:
    - warmup_epochs (int): Epochen mit Standard-InfoNCE vor Feature-Aktivierung
    - soft_mix_config (dict): Curriculum Learning für soft_mix
        - schedule (bool): Scheduling aktivieren
        - initial (float): Start-Wert für soft_mix [0.0-0.3]
        - final (float): End-Wert für soft_mix [0.7-1.0]
        - schedule_type (str): 'linear', 'cosine', 'exponential'
    
    TECHNISCHE PARAMETER:
    - eps (float): Numerische Stabilisierung [1e-6 - 1e-8]
    """
    
    # Default-Werte für alle Hyperparameter
    defaults = {
        # === KERN-PARAMETER ===
        'temperature': 0.1,
        
        # === FEATURE-AKTIVIERUNG ===
        'mm_swnce_mode': 'wcc',  # Default: Weighting + Cycle-Consistency
        'use_weighting': None,   # Wird aus mm_swnce_mode gesetzt, kann aber überschrieben werden
        'use_soft_targets': None,
        'use_self_similarity': None,
        
        # === SOFT TARGETS ===
        'soft_target_mode': 'cycle',  # 'cycle' oder 'swapped'
        'soft_mix': 0.5,
        'tau_s': 0.02,  # Source temperature
        'tau_t': 0.07,  # Target temperature
        
        # === SELF-SIMILARITY ===
        'selfsim_mix': 0.5,
        'tau_self_i2j': 0.07,
        'tau_self_j2i': 0.07,
        
        # === WEIGHTING ===
        'weighting_delta': 0.0,
        'weighting_kappa': 0.5,
        'weighting_w_min': 0.25,
        
        # === TRAINING-DYNAMIK ===
        'warmup_epochs': 0,
        'soft_mix_schedule': False,
        'soft_mix_initial': 0.1,
        'soft_mix_final': 0.9,
        'soft_mix_schedule_type': 'cosine',
        
        # === TECHNISCH ===
        'eps': 1e-6,
    }
    
    # Extrahiere Werte aus Config mit Fallback auf Defaults
    params = {}
    
    # Kern-Parameter
    params['temperature'] = config.get('temperature', defaults['temperature'])
    
    # Feature-Aktivierung über mm_swnce_mode
    mm_swnce_mode = config.get('mm_swnce_mode', defaults['mm_swnce_mode'])
    mode_config = get_mm_swnce_config(mm_swnce_mode)
    
    # Erlaube manuelle Überschreibung der Feature-Flags
    params['use_weighting'] = config.get('use_weighting', mode_config['use_weighting'])
    params['use_soft_targets'] = config.get('use_soft_targets', mode_config['use_soft_targets'])
    params['use_self_similarity'] = config.get('use_self_similarity', mode_config['use_self_similarity'])
    
    # Soft Targets
    params['soft_target_mode'] = config.get('soft_target_mode', defaults['soft_target_mode'])
    params['soft_mix'] = config.get('soft_mix', defaults['soft_mix'])
    params['tau_s'] = config.get('tau_s', defaults['tau_s'])
    params['tau_t'] = config.get('tau_t', defaults['tau_t'])
    
    # Self-Similarity
    params['selfsim_mix'] = config.get('selfsim_mix', defaults['selfsim_mix'])
    params['tau_self_i2j'] = config.get('tau_self_i2j', defaults['tau_self_i2j'])
    params['tau_self_j2i'] = config.get('tau_self_j2i', defaults['tau_self_j2i'])
    
    # Weighting Parameters
    params['weighting_params'] = {
        'delta': config.get('weighting_delta', defaults['weighting_delta']),
        'kappa': config.get('weighting_kappa', defaults['weighting_kappa']),
        'w_min': config.get('weighting_w_min', defaults['weighting_w_min']),
    }
    
    # Training-Dynamik
    params['warmup_epochs'] = config.get('warmup_epochs', defaults['warmup_epochs'])
    
    # Soft-Mix Scheduler (aus soft_mix_config)
    soft_mix_config = config.get('soft_mix_config', {})
    params['soft_mix_schedule'] = soft_mix_config.get('schedule', defaults['soft_mix_schedule'])
    params['soft_mix_initial'] = soft_mix_config.get('initial', defaults['soft_mix_initial'])
    params['soft_mix_final'] = soft_mix_config.get('final', defaults['soft_mix_final'])
    params['soft_mix_schedule_type'] = soft_mix_config.get('schedule_type', defaults['soft_mix_schedule_type'])
    
    # Technisch
    params['eps'] = config.get('eps', defaults['eps'])
    
    return params


class MM_SWNCE(nn.Module):
    """
    Multi-Modal Soft-Weighted NCE (MM_SWNCE) Loss.
    
    Drop-in replacement for InfoNCELoss1 with advanced robustness features:
    - Standard InfoNCE (symmetrisch)
    - Optional: Robuste Gewichtung nach positiver Similarity (Weighting for faulty positives)
    - Optional: Soft Targets (Cycle-Consistent für faulty negatives)
    - Optional: Self-Similarity (intra-modale Konsistenz)
    
    Modi können über get_mm_swnce_config(mode) gesetzt werden:
    - 'w':    Weighting only
    - 'ss':   Self-Similarity only
    - 'cc':   Cycle-Consistency only
    - 'wcc':  Weighting + Cycle-Consistency
    - 'sscc': Self-Similarity + Cycle-Consistency
    
    Aufruf identisch zu InfoNCELoss1:
        total_loss, loss_dict = loss_fn(*feature_sets)

    feature_sets: Liste/Tuple von 2+ Tensors (B, D), eine pro Modality.
    """

    def __init__(
        self,
        temperature: float = 0.1,           # τ für die Softmax über Paar-Ähnlichkeiten
        use_weighting: bool = True,          # gewichtetes xID aktivieren (faulty positives)
        weighting_params: dict = None,       # {'delta':0.0,'kappa':0.5,'w_min':0.25}
        use_soft_targets: bool = False,      # Soft Targets aktivieren (faulty negatives)
        soft_target_mode: str = "cycle",     # 'swapped' oder 'cycle'
        soft_mix: float = 0.5,               # Anteil soft targets (0=pure hard, 1=pure soft)
        tau_s: float = 0.02,                 # Temperatur für weiche Teile (source)
        tau_t: float = 0.07,                 # Temperatur für weiche Teile (target)
        eps: float = 1e-6,                   # numerische Stabilisierung
        selfsim_mix: float = 0.5,
        use_self_similarity: bool = False,   # L_SS berechnen und einmischen
        alpha_ss: float = 0.5,               # Mischung (0 -> nur xID, 1 -> nur Self-Sim)
        beta_ss: float = 0.2,                # P_self = beta*softmax(AA/tau_self) + (1-beta)*I
        tau_self: float = 0.07,              # Temperatur für intra-modale Softmax
        tau_self_i2j: float = 0.07,          # Temperatur für i->j Self-Similarity (basierend auf z2)
        tau_self_j2i: float = 0.07,          # Temperatur für j->i Self-Similarity (basierend auf z1)
    ):
        super().__init__()
        self.temperature = temperature
        self.use_weighting = use_weighting
        self.use_soft_targets = use_soft_targets
        self.soft_target_mode = soft_target_mode
        self.soft_mix = soft_mix
        self.tau_s = tau_s
        self.tau_t = tau_t
        self.eps = eps

        # Soft-Self-Similarity
        self.use_self_similarity = use_self_similarity
        self.alpha_ss = alpha_ss
        self.beta_ss = beta_ss
        self.tau_self = tau_self
        self.soft_mix = soft_mix

        self.selfsim_mix = selfsim_mix
        self.tau_self_i2j = tau_self_i2j
        self.tau_self_j2i = tau_self_j2i

        # Standard-Parameter für die Gewichtung, falls nichts übergeben wurde
        default_wp = dict(delta=0.0, kappa=0.5, w_min=0.25)
        self.wp = {**default_wp, **(weighting_params or {})}


        # DDP-Status
        self._ddp = dist.is_available() and dist.is_initialized()

        # EMA-Puffer für globale Statistik (mu, sigma)
        self.register_buffer("ema_mu", torch.tensor(0.0))
        self.register_buffer("ema_sig", torch.tensor(1.0))
        self.ema_momentum = 0.99  # ggf. konfigurierbar machen

        # Cross-batch Memory als Fallback auf Single-GPU (letzte K Batches)
        self.cbmem = deque(maxlen=64)

    # --------------------------
    # Hilfsfunktionen
    # --------------------------
    def _intra_targets(self, z, tau):
        """
        Erzeuge pure intra-modale Soft-Targets (nur Softmax, keine Identity):
        P_pure = softmax(sim_zz / tau)
        """
        B = z.size(0)
        sim_zz = (z @ z.t()) / (tau + 1e-12)   # z erwartet L2-normalisiert
        P_pure = F.softmax(sim_zz, dim=1)
        return P_pure
    
    def _self_similarity_targets(self, z1, z2):
        """
        Berechne Self-Similarity Targets (pure, ohne Identity).
        
        Berücksichtigt beide Richtungen:
        - P_i2j: Zielraum für i->j basierend auf z2's Self-Similarity (tau_self_i2j)
        - P_j2i: Zielraum für j->i basierend auf z1's Self-Similarity (tau_self_j2i)
        
        Returns:
            Tuple[Tensor, Tensor]: (P_i2j, P_j2i) mit Shape [B, B]
        """
        P_i2j = self._intra_targets(z2, tau=self.tau_self_i2j)  # Self-similarity von z2 für i->j
        P_j2i = self._intra_targets(z1, tau=self.tau_self_j2i)  # Self-similarity von z1 für j->i
        return P_i2j, P_j2i


    @staticmethod
    def _normalize(*xs):
        """L2-Normalisierung entlang der letzten Dimension (cosine sim)."""
        out = []
        for x in xs:
            # Sicherstellen, dass Tensor vorhanden ist (kann None sein)
            out.append(None if x is None else F.normalize(x, dim=-1))
        return out if len(out) > 1 else out[0]

    @staticmethod
    def _pairwise_sim(z1, z2):
        """Paarweise Cosine-Similarities (unskaliert), Form [B, B]."""
        return z1 @ z2.t()

    def _pairwise_logits(self, z1, z2):
        """Logits = cos-sim / τ, Form [B, B]."""
        return self._pairwise_sim(z1, z2) / self.temperature

    def _soft_targets_swapped(self, sim_va):
        """
        Swapped Targets:
        - Für P(a|v): T_v(j|i) ~ softmax(sim_av[i,j] / τ_s)  (sim_av = sim_va^T)
        - Für P(v|a): T_a(j|i) ~ softmax(sim_va[i,j] / τ_s)
        """
        sim_av = sim_va.t()
        T_v = torch.softmax(sim_av / self.tau_s, dim=1)  # [B,B], a_i -> v_j
        T_a = torch.softmax(sim_va / self.tau_s, dim=1)  # [B,B], v_i -> a_j
        return T_v, T_a

    def _soft_targets_cycle(self, sim_va):
        """
        Cycle-Consistent Targets (vereinfachte CCP-Variante):
        S_v(j|i) ∝ exp( (v_i^T a_i)/τ_t + (a_i^T v_j)/τ_s + (v_j^T a_j)/τ_t )
        S_a(j|i) ∝ exp( (a_i^T v_i)/τ_t + (v_i^T a_j)/τ_s + (a_j^T v_j)/τ_t )
         """
        sim_av = sim_va.t()                 # [B,B]
        diag = torch.diag(sim_va)           # [B]  (v_i^T a_i)
        # Terme für T_v (P(a|v))
        term1_v = diag[:, None] / self.tau_t      # (v_i^T a_i)/τ_t  -> broadcast über j
        term2_v = sim_av / self.tau_s             # (a_i^T v_j)/τ_s
        term3_v = diag[None, :] / self.tau_t      # (v_j^T a_j)/τ_t
        S_v = torch.softmax(term1_v + term2_v + term3_v, dim=1)

        # Terme für T_a (P(v|a))
        term1_a = diag[:, None] / self.tau_t
        term2_a = sim_va / self.tau_s
        term3_a = diag[None, :] / self.tau_t
        S_a = torch.softmax(term1_a + term2_a + term3_a, dim=1)
        return S_v, S_a

    def _pair_loss(self, z1, z2, name_left="mod_i", name_right="mod_j"):
        z1, z2 = self._normalize(z1, z2)

        logits_i2j = self._pairwise_logits(z1, z2)  # [B,B]
        logits_j2i = logits_i2j.t()
        B = logits_i2j.size(0)
        I = torch.eye(B, device=logits_i2j.device, dtype=logits_i2j.dtype)

        # --- Schritt 1: Berechne pure soft targets (OHNE Identity) ---
        T_soft_i2j = None  # i->j (mod_i -> mod_j)
        T_soft_j2i = None  # j->i (mod_j -> mod_i)
        
        # Cycle Consistency (falls aktiv)
        if self.use_soft_targets:
            sim_va = self._pairwise_sim(z1, z2)
            if self.soft_target_mode == "swapped":
                T_cycle_j2i, T_cycle_i2j = self._soft_targets_swapped(sim_va)
            elif self.soft_target_mode == "cycle":
                T_cycle_j2i, T_cycle_i2j = self._soft_targets_cycle(sim_va)
            else:
                raise ValueError(f"Unknown soft_target_mode: {self.soft_target_mode}")
            T_soft_i2j = T_cycle_i2j
            T_soft_j2i = T_cycle_j2i
        
        # Self-Similarity (falls aktiv)
        if self.use_self_similarity:
            # Berechne vollständige Self-Similarity Targets (beide Richtungen)
            P_ss_i2j, P_ss_j2i = self._self_similarity_targets(z1, z2)
            
            # Schritt 2: Mische Cycle + Self-Similarity (falls beide aktiv)
            if self.use_soft_targets:
                # Beide aktiv: mische mit selfsim_mix
                T_soft_i2j = (1.0 - self.selfsim_mix) * T_soft_i2j + self.selfsim_mix * P_ss_i2j
                T_soft_j2i = (1.0 - self.selfsim_mix) * T_soft_j2i + self.selfsim_mix * P_ss_j2i
            else:
                # Nur Self-Similarity aktiv
                T_soft_i2j = P_ss_i2j
                T_soft_j2i = P_ss_j2i
        
        # --- Schritt 3: Mische soft targets mit Identity (soft_mix) ---
        if T_soft_i2j is not None:
            # Mindestens ein soft target aktiv
            T_final_i2j = (1.0 - self.soft_mix) * I + self.soft_mix * T_soft_i2j
            T_final_j2i = (1.0 - self.soft_mix) * I + self.soft_mix * T_soft_j2i
            
            # Loss berechnen mit soft targets
            logP_i = F.log_softmax(logits_i2j, dim=1)
            logP_j = F.log_softmax(logits_j2i, dim=1)
            loss_i_vec = -(T_final_i2j * logP_i).sum(dim=1)  # [B]
            loss_j_vec = -(T_final_j2i * logP_j).sum(dim=1)  # [B]
        else:
            # Keine soft targets: Standard InfoNCE
            labels = torch.arange(B, device=logits_i2j.device)
            loss_i_vec = F.cross_entropy(logits_i2j, labels, reduction="none")
            loss_j_vec = F.cross_entropy(logits_j2i, labels, reduction="none")

        # --- Schritt 4: Weighting anwenden ---
        if self.use_weighting:
            pos_sims = torch.diag(z1 @ z2.t())
            w = self._robust_weights_from_pos_sims(pos_sims)  # [B]
        else:
            w = torch.ones(B, device=z1.device, dtype=loss_i_vec.dtype)

        loss_i_final = (w * loss_i_vec).sum() / (w.sum() + self.eps)
        loss_j_final = (w * loss_j_vec).sum() / (w.sum() + self.eps)
        loss_pair = 0.5 * (loss_i_final + loss_j_final)

        # --- Logging ---
        details = {
            f"{name_left}_to_{name_right}": loss_i_vec.mean().detach().cpu().item(),
            f"{name_right}_to_{name_left}": loss_j_vec.mean().detach().cpu().item(),
            f"w_mean_{name_left}_{name_right}": w.mean().detach().cpu().item(),
            "soft_mix": float(self.soft_mix),
            "selfsim_mix": float(self.selfsim_mix) if self.use_self_similarity else 0.0,
            "tau_self_i2j": float(self.tau_self_i2j) if self.use_self_similarity else 0.0,
            "tau_self_j2i": float(self.tau_self_j2i) if self.use_self_similarity else 0.0,
        }
        return loss_pair, details

    def _gather_pos_sims(self, pos_sims: torch.Tensor) -> torch.Tensor:
        """
        Sammle die Diagonal-Similarities über alle GPUs (DDP).
        Single-GPU: nutze Cross-batch Memory als Fallback.
        """
        ps = pos_sims.detach()
        if self._ddp:
            world_size = dist.get_world_size()
            gather_list = [torch.zeros_like(ps) for _ in range(world_size)]
            dist.all_gather(gather_list, ps)
            ps_glob = torch.cat(gather_list, dim=0)
            return ps_glob
        else:
            self.cbmem.append(ps.cpu())
            ps_glob = torch.cat(list(self.cbmem), dim=0).to(ps.device)
            return ps_glob


    def _robust_weights_from_pos_sims(self, pos_sims):
        """
        Bestimme w_i per CDF mit stabiler mu/sigma:
          - DDP: all_gather über GPUs
          - Single-GPU: Cross-batch Memory
          - EMA-Glättung auf mu/sigma
        """
        delta, kappa, w_min = self.wp["delta"], self.wp["kappa"], self.wp["w_min"]

        # 1) globale/gestützte Sammlung
        ps_glob = self._gather_pos_sims(pos_sims)  # [global_B]

        # 2) Moment-Schätzer (detach)
        mu_b = ps_glob.mean().detach().cpu()
        sig_b = ps_glob.std(unbiased=False).clamp_min(self.eps).detach().cpu()

        # 3) EMA-Update
        with torch.no_grad():
            self.ema_mu.mul_(self.ema_momentum).add_((1 - self.ema_momentum) * mu_b)
            self.ema_sig.mul_(self.ema_momentum).add_((1 - self.ema_momentum) * sig_b)

        mu = self.ema_mu
        sigma = self.ema_sig.clamp_min(self.eps)

        # 4) z-Score mit Shift/Spread
        z = (pos_sims - (mu + delta * sigma)) / (sigma * (kappa ** 0.5) + self.eps)

        # 5) CDF + Soft-Truncation
        cdf = 0.5 * (1.0 + torch.erf(z / 1.41421356237))
        w = cdf * (1.0 - w_min) + w_min
        return w

    # --------------------------
    # Forward über ALLE Modalitäten-Paare
    # --------------------------
    def forward(self, *feature_sets):
        """
        feature_sets: Sequenz von Tensoren [B, D], eine pro Modality.
        Wir berechnen den Loss über alle Paare i<j (symmetrisch).
        """
        num_modalities = len(feature_sets)
        assert num_modalities >= 2, "RobustXIDLoss benötigt >= 2 Modalitäten."

        total_loss = 0.0
        loss_dict = {}
        count = 0

        # Über alle Modality-Paare iterieren (i<j)
        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Paar-Loss berechnen
                pair_loss, details = self._pair_loss(
                    feature_sets[i], feature_sets[j],
                    name_left=f"mod_{i}",
                    name_right=f"mod_{j}", 
                )
                total_loss = total_loss + pair_loss
                count += 1
                loss_dict.update(details)

        # Durchschnitt über alle Paare
        total_loss = total_loss / count
        return total_loss, loss_dict


class FastApproxMM_SWNCE(nn.Module):
    """
    Drop-in replacement for MM_SWNCE
    ------------------------------------
    * Same call signature: loss, dict = loss_fn(*feature_sets)
    * O(M · B²) softmaxes (InfoNCE only)
    * All robustness terms are O(B·D)
    * Typically 2–3× InfoNCE wall-clock
    """

    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.1,   # ≈ soft targets
        confidence_scale: float = 5.0,  # ≈ robust weighting sharpness
        var_weight: float = 0.05,       # ≈ self-similarity strength
        eps: float = 1e-6,
    ):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.confidence_scale = confidence_scale
        self.var_weight = var_weight
        self.eps = eps

    # ------------------------------------------------------------
    # cheap self-similarity (VICReg-style)
    # ------------------------------------------------------------

    def _variance_loss(self, z):
        # z: [B,D], normalized
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return torch.mean(F.relu(1.0 - std))

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------

    def forward(self, *feature_sets):
        """
        feature_sets: list of [B,D] tensors (modalities)
        returns: total_loss, loss_dict
        """
        num_modalities = len(feature_sets)
        assert num_modalities >= 2

        Z = [F.normalize(z, dim=-1) for z in feature_sets]
        B = Z[0].size(0)
        device = Z[0].device

        labels = torch.arange(B, device=device)

        total_loss = 0.0
        pair_count = 0
        loss_dict = {}

        # iterate over modality pairs (cheap loop, heavy ops are batched)
        for i in range(num_modalities):
            zi = Z[i]

            for j in range(i + 1, num_modalities):
                zj = Z[j]

                # --------------------------------------------------
                # InfoNCE (ONE softmax)
                # --------------------------------------------------
                logits = zi @ zj.t() / self.temperature

                loss_i = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=self.label_smoothing,
                    reduction="none",
                )
                loss_j = F.cross_entropy(
                    logits.t(),
                    labels,
                    label_smoothing=self.label_smoothing,
                    reduction="none",
                )

                # --------------------------------------------------
                # fast confidence weighting (batch-local)
                # --------------------------------------------------
                pos = (zi * zj).sum(dim=1)  # cosine sim

                mu = pos.mean()
                sig = pos.std(unbiased=False).clamp_min(self.eps)

                w = torch.sigmoid(self.confidence_scale * (pos - mu) / sig)

                pair_loss = 0.5 * (
                    (w * loss_i).mean() +
                    (w * loss_j).mean()
                )

                # --------------------------------------------------
                # cheap self-similarity regularization
                # --------------------------------------------------
                if self.var_weight > 0:
                    pair_loss = pair_loss + self.var_weight * (
                        self._variance_loss(zi) +
                        self._variance_loss(zj)
                    )

                total_loss = total_loss + pair_loss
                pair_count += 1

                loss_dict[f"mod_{i}_mod_{j}"] = pair_loss.detach().item()
                loss_dict[f"w_mean_{i}_{j}"] = w.mean().detach().item()

        total_loss = total_loss / pair_count
        return total_loss, loss_dict



class NCEContrastiveLoss(nn.Module):
    """
    Contrastive Loss for multi-modal learning.
    
    This loss encourages embeddings of corresponding samples from different modalities
    to be similar, while pushing non-corresponding samples apart.

    Args:
        temp (float): Temperature parameter to scale the similarity scores.
    """

    def __init__(self, temp):
        super(NCEContrastiveLoss, self).__init__()
        self.temp = temp

    def forward(self, vis_feat, text_feat):

        vis_feat, text_feat = normalize(vis_feat, text_feat)
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) / self.temp  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        loss =    (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label) ) / 2
        return loss

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

#cross entropy loss with real soft targets 
def soft_cross_entropy(logits, soft_targets, reduction='mean'):
    log_probs = F.log_softmax(logits, dim=1)  # apply log softmax
    loss = -torch.sum(soft_targets * log_probs, dim=1)  # batch-wise loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # no reduction

#modified NCE loss with self-similarity scaled by a beta parameter
class NCEwithSelfSimilarity(nn.Module): 
    
    """
    NCE Contrastive Loss with self-similarity Soft targets for multi-modal learning.
    
    This loss computes the NCE loss between pairs of features from different modalities,
    while also considering self-similarity within each modality.
    """

    def __init__(self, temperature=0.1, beta=0.2):
        super(NCEwithSelfSimilarity, self).__init__()
        self.temperature = temperature
        self.beta = beta


    def forward(self, feat_a, feat_b): 

        feat_a, feat_b = normalize(feat_a, feat_b)
        # Compute similarity scores
        sim_aa = torch.matmul(feat_a, feat_a.permute(1, 0)) / self.temperature
        sim_bb = torch.matmul(feat_b, feat_b.permute(1, 0)) / self.temperature

        P_intra_ab = self.beta * F.softmax(sim_aa, dim=1) + (1 - self.beta) * torch.eye(sim_aa.size(0)).to(sim_aa.device)
        P_intra_ba = self.beta * F.softmax(sim_bb, dim=1) + (1 - self.beta) * torch.eye(sim_bb.size(0)).to(sim_bb.device) 

        # Compute cross-modal similarity scores
        sim_ab = torch.matmul(feat_a, feat_b.permute(1, 0)) / self.temperature
        sim_ba = torch.matmul(feat_b, feat_a.permute(1, 0)) / self.temperature

        # Compute Loss 
        loss_ab = soft_cross_entropy(sim_ab, P_intra_ab)
        loss_ba = soft_cross_entropy(sim_ba, P_intra_ba)

        # combined loss
        total_loss = loss_ab + loss_ba

        return total_loss 

class infoNCEwithIntraModalCrossConsistency(nn.Module):
    def __init__(self, temperature=0.1):
        super(infoNCEwithIntraModalCrossConsistency, self).__init__()
        self.temperature = temperature

    def forward(self, feat_a, feat_b):
        feat_a, feat_b = normalize(feat_a, feat_b)

        # Compute similarity scores
        sim_ab = torch.matmul(feat_a, feat_b.permute(1, 0)) / self.temperature # AB^T 
        sim_ba = torch.matmul(feat_b, feat_a.permute(1, 0)) / self.temperature # BA^T

        sim_aa = torch.matmul(feat_a, feat_a.permute(1, 0)) / self.temperature # AA^T
        sim_bb = torch.matmul(feat_b, feat_b.permute(1, 0)) / self.temperature # BB^T

        batch_size = sim_ab.size(0) 

        I = torch.eye(batch_size, device=sim_ab.device)  # Identity matrix for intra-modal consistency

        J = torch.ones(batch_size, batch_size, device=sim_ab.device) # Matrix of ones for cross-modal consistency


        # Teacher-Targets: Keine Gradienten für Q_a2b
        Q_a2b = F.softmax(sim_ab + torch.matmul(sim_ba, J) + torch.matmul(J, sim_ab), dim=1)

        # Student-Targets: Gradienten erlaubt für Q_b2a
        Q_b2a = F.softmax(sim_ba + torch.matmul(sim_ab, J) + torch.matmul(J, sim_ba), dim=1)


        # Cross-modal losses
        loss_ab = soft_cross_entropy(sim_ab, Q_a2b)
        loss_ba = soft_cross_entropy(sim_ba, Q_b2a)

        # Total loss
        total_loss = loss_ab + loss_ba 

        return total_loss 


class NCE_soft_targets(nn.Module):
    """
    NCE Contrastive Loss with soft targets for multi-modal learning.
    
    This loss combines NCEwithSelfSimilarity and infoNCEwithIntraModalCrossConsistency
    using a weighted combination controlled by the alpha parameter.
    
    Args:
        temperature (float): Temperature parameter to scale the similarity scores.
        alpha (float): Weighting parameter for self-similarity component (0-1).
                      Should be scheduled to decrease during training (start high, end low).
                      High alpha = more self-similarity, Low alpha = more cross-consistency.
    """
    def __init__(self, temperature=0.1, alpha=0.9):
        super(NCE_soft_targets, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        
        # Initialize the two loss components
        self.nce_self_similarity = NCEwithSelfSimilarity(temperature=temperature)
        self.nce_cross_consistency = infoNCEwithIntraModalCrossConsistency(temperature=temperature)

    def forward(self, a_feat, b_feat):
        # Compute both loss components
        loss_self_similarity = self.nce_self_similarity(a_feat, b_feat)
        loss_cross_consistency = self.nce_cross_consistency(a_feat, b_feat)
        
        # Weighted combination: alpha controls self-similarity weight, (1-alpha) controls cross-consistency weight
        total_loss = self.alpha * loss_self_similarity + (1 - self.alpha) * loss_cross_consistency
        
        return total_loss
    
    def update_alpha(self, new_alpha):
        """
        Update the alpha parameter during training.
        
        Args:
            new_alpha (float): New value for alpha parameter.
        """
        self.alpha = new_alpha
    
    def get_alpha(self):
        """
        Get current alpha value.
        
        Returns:
            float: Current alpha value.
        """
        return self.alpha   

class MSELoss(nn.Module):
    """
    Mean Squared Error (MSE) Contrastive Loss for multi-modal learning.
    
    This loss computes the MSE between corresponding embeddings from two modalities,
    encouraging direct alignment between paired samples.
    """

    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # Default reduction is 'mean'

    def forward(self, vis_feat, text_feat):
        # Ensure input features have the same dimensions
        if vis_feat.size(0) != text_feat.size(0):
            raise ValueError("The number of features in each set must match")

        # Compute MSE loss directly between corresponding elements
        loss = self.mse_loss(vis_feat, text_feat)
        return loss


class SoftLabelCrossEntropyLoss(nn.Module):
    """
    Soft Label Cross Entropy Loss with label smoothing.
    
    This loss applies label smoothing to standard cross-entropy, which can help
    prevent overfitting and improve generalization.

    Args:
        num_classes (int): Number of classes in the classification task.
        smoothing (float): Label smoothing factor (0-1).
    """
    def __init__(self, num_classes=34, smoothing=0.1):
        super(SoftLabelCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.smoothing)

    def forward(self, logits, hard_labels):
        # hard_labels are expected to be class indices; for true soft targets, you would need a different approach
        return self.loss_fn(logits, hard_labels)
    

class DiagonalKLDivLoss(nn.Module):
    """
    Diagonal Kullback-Leibler Divergence Loss for multi-modal learning.
    
    This loss computes the KL divergence between the predicted distribution (logits)
    and the target distribution, useful for aligning probability distributions
    across modalities.

    Args:
        temperature (float): Temperature parameter to scale the logits.
    """
    def __init__(self, temperature=1.0):
        super(DiagonalKLDivLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, targets):
        if logits.size(0) != targets.size(0):
            raise ValueError("The number of features in each set must match")

        # Apply softmax to targets to convert them into probability distributions
        targets = F.softmax(targets / self.temperature, dim=-1)

        # Apply log_softmax to logits
        logits = F.log_softmax(logits / self.temperature, dim=-1)

        # Compute KL divergence
        loss = self.kl_div(logits, targets)
        return loss


class InfoNCELoss2(nn.Module):
    """
    InfoNCE Loss for multiple modalities.
    
    This loss extends the NCE Contrastive Loss to handle multiple modalities,
    computing pairwise losses between all modality combinations.
    But other than the InfoNCELoss1, this loss uses softened targets for the NCE loss.

    Args:
        temperature (float): Temperature parameter for the NCE loss.
    """

    def __init__(self, temperature=0.1):
        super(InfoNCELoss2, self).__init__()
        self.temperature = temperature
        # Instantiate NCEwithSelfSimilarity with the given temperature
        self.nce_loss = NCEwithSelfSimilarity(temperature)

    def forward(self, *feature_sets):
        num_modalities = len(feature_sets)
        total_loss = 0.0
        count = 0
        loss_dict = {}

        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Calculate loss for each pair using NCEContrastiveLoss
                # Here, we consider only one direction (i -> j)
                loss_ij = self.nce_loss(feature_sets[i], feature_sets[j])
                total_loss += loss_ij
                count += 1
                # Detach the loss, move it to CPU, and convert to Python scalar
                loss_value = loss_ij.detach().cpu().item()
                loss_dict[f'modality_{i}_to_modality_{j}'] = loss_value #dictorionary to store losses for each modality pair

        # Average loss over all modality pairs
        total_loss /= count
        return total_loss, loss_dict


class InfoNCESoftTargets(nn.Module):
    """
    InfoNCE Loss for multiple modalities using NCE_soft_targets.
    
    This loss extends the NCE_soft_targets to handle multiple modalities,
    computing pairwise losses between all modality combinations using the 
    adaptive alpha-weighted combination of self-similarity and cross-consistency.

    Args:
        temperature (float): Temperature parameter for the NCE loss.
        alpha (float): Initial alpha value for NCE_soft_targets.
    """

    def __init__(self, temperature=0.1, alpha=0.9):
        super(InfoNCESoftTargets, self).__init__()
        self.temperature = temperature
        # Instantiate NCE_soft_targets with the given temperature and alpha
        self.nce_loss = NCE_soft_targets(temperature=temperature, alpha=alpha)

    def forward(self, *feature_sets):
        num_modalities = len(feature_sets)
        total_loss = 0.0
        count = 0
        loss_dict = {}

        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Calculate loss for each pair using NCE_soft_targets
                loss_ij = self.nce_loss(feature_sets[i], feature_sets[j])
                total_loss += loss_ij
                count += 1
                # Detach the loss, move it to CPU, and convert to Python scalar
                loss_value = loss_ij.detach().cpu().item()
                loss_dict[f'modality_{i}_to_modality_{j}'] = loss_value

        # Average loss over all modality pairs
        total_loss /= count
        return total_loss, loss_dict
    
    def update_alpha(self, new_alpha):
        """
        Update the alpha parameter in the underlying NCE_soft_targets loss.
        
        Args:
            new_alpha (float): New value for alpha parameter.
        """
        self.nce_loss.update_alpha(new_alpha)
    
    def get_alpha(self):
        """
        Get current alpha value from the underlying NCE_soft_targets loss.
        
        Returns:
            float: Current alpha value.
        """
        return self.nce_loss.get_alpha()

    
class InfoNCELoss1(nn.Module):
    """
    InfoNCE Loss for multiple modalities.
    
    This loss extends the NCE Contrastive Loss to handle multiple modalities,
    computing pairwise losses between all modality combinations.

    Args:
        temperature (float): Temperature parameter for the NCE loss.
    """

    def __init__(self, temperature=0.1):
        super(InfoNCELoss1, self).__init__()
        self.temperature = temperature
        # Instantiate NCEContrastiveLoss with the given temperature
        self.nce_loss = NCEContrastiveLoss(temperature)

    def forward(self, *feature_sets):
        num_modalities = len(feature_sets)
        total_loss = 0.0
        count = 0
        loss_dict = {}

        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Calculate loss for each pair using NCEContrastiveLoss
                # Here, we consider only one direction (i -> j)
                loss_ij = self.nce_loss(feature_sets[i], feature_sets[j])
                total_loss += loss_ij
                count += 1
                # Detach the loss, move it to CPU, and convert to Python scalar
                loss_value = loss_ij.detach().cpu().item()
                loss_dict[f'modality_{i}_to_modality_{j}'] = loss_value #dictorionary to store losses for each modality pair

        # Average loss over all modality pairs
        total_loss /= count
        return total_loss, loss_dict
    

class SigmoidContrastiveMultiModalLoss(nn.Module):
    """
    Sigmoid Contrastive Loss for multiple modalities with learnable temperature and bias.
    
    This loss uses a sigmoid function to measure similarity between modalities,
    with learnable temperature and bias parameters for flexibility.

    Args:
        temperature_initial (float): Initial value for the temperature parameter.
        bias_initial (float): Initial value for the bias parameter.
    """
    def __init__(self, temperature_initial=10, bias_initial=-10):
        super(SigmoidContrastiveMultiModalLoss, self).__init__()
        # Initialize temperature and bias as learnable parameters
        self.temperature = nn.Parameter(torch.tensor([temperature_initial]).float())
        self.bias = nn.Parameter(torch.tensor([bias_initial]).float())

    def forward(self, *feature_sets):
        """
        *feature_sets are the normalized feature vectors from the different modalities.
        Each element in feature_sets should have the shape [batch_size, feature_size].
        """
        num_modalities = len(feature_sets)
        batch_size = feature_sets[0].size(0)

        total_loss = 0.0
        count = 0

        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Compute similarity scores between features of modality i and j
                logits = torch.matmul(feature_sets[i], feature_sets[j].T)
                logits = logits * self.temperature.exp().to(logits.device) + self.bias.to(logits.device)

                # Create labels: 1 for matching pairs (diagonal), -1 for non-matching pairs
                labels = 2 * torch.eye(batch_size).to(logits.device) - 1

                # Compute the sigmoid loss for this pair of modalities
                loss = -torch.mean(F.logsigmoid(labels * logits))
                total_loss += loss
                count += 1

        # Average loss over all modality pairs
        total_loss /= count
        return total_loss
    

    
# Loss function
def mse_loss(reconstructed, original):
    criterion = nn.MSELoss()
    loss = criterion(reconstructed, original)
    return loss

def create_scheduler(optimizer, config):
    """
    Create a learning rate scheduler based on the configuration.
    
    Supports 'cosine', 'exponential', 'step', and 'plateau' schedulers.

    Args:
        optimizer: The optimizer to schedule.
        config (dict): Configuration containing scheduler type and parameters.

    Returns:
        torch.optim.lr_scheduler: The configured learning rate scheduler.
    """
    scheduler_config = config.get('scheduler_config', {})
    scheduler_type = scheduler_config.get('type', 'step')
    scheduler_params = scheduler_config.get('params', {})

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler


def create_alpha_scheduler(initial_alpha=0.9, final_alpha=0.1, total_epochs=100, schedule_type='cosine'):
    """
    Create an alpha scheduler for NCE_soft_targets loss.
    
    Args:
        initial_alpha (float): Starting value of alpha (default 0.9 for high self-similarity weight).
        final_alpha (float): Final value of alpha (default 0.1 for low self-similarity weight).
        total_epochs (int): Total number of training epochs.
        schedule_type (str): Type of scheduling ('linear', 'cosine', 'exponential').
    
    Returns:
        function: A function that takes current epoch and returns alpha value.
    """
    def linear_schedule(epoch):
        progress = min(epoch / total_epochs, 1.0)
        return initial_alpha + (final_alpha - initial_alpha) * progress
    
    def cosine_schedule(epoch):
        progress = min(epoch / total_epochs, 1.0)
        # Cosine annealing: starts at initial_alpha, smoothly decreases to final_alpha
        cosine_progress = (1 + torch.cos(torch.tensor(progress * torch.pi))) / 2
        return final_alpha + (initial_alpha - final_alpha) * cosine_progress.item()
    
    def exponential_schedule(epoch):
        progress = min(epoch / total_epochs, 1.0)
        exp_progress = (torch.exp(torch.tensor(progress)) - 1) / (torch.exp(torch.tensor(1.0)) - 1)
        return initial_alpha + (final_alpha - initial_alpha) * exp_progress.item()
    
    if schedule_type == 'linear':
        return linear_schedule
    elif schedule_type == 'cosine':
        return cosine_schedule
    elif schedule_type == 'exponential':
        return exponential_schedule
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")


class AlphaScheduler:
    """
    Alpha parameter scheduler for NCE_soft_targets loss.
    
    This scheduler gradually decreases the alpha parameter during training,
    transitioning from emphasizing self-similarity to cross-consistency.
    """
    def __init__(self, loss_function, initial_alpha=0.9, final_alpha=0.1, 
                 total_epochs=100, schedule_type='cosine'):
        """
        Initialize the alpha scheduler.
        
        Args:
            loss_function: Instance of NCE_soft_targets loss.
            initial_alpha (float): Starting value of alpha (high = more self-similarity).
            final_alpha (float): Final value of alpha (low = more cross-consistency).
            total_epochs (int): Total number of training epochs.
            schedule_type (str): Type of scheduling ('linear', 'cosine', 'exponential').
        """
        self.loss_function = loss_function
        self.schedule_fn = create_alpha_scheduler(initial_alpha, final_alpha, 
                                                total_epochs, schedule_type)
        
    def step(self, epoch):
        """
        Update alpha parameter based on current epoch.
        
        Args:
            epoch (int): Current training epoch.
        """
        new_alpha = self.schedule_fn(epoch)
        self.loss_function.update_alpha(new_alpha)
        return new_alpha


class DINOStyleLoss(nn.Module):
    """
    DINO-style cross-modal distillation loss (sample-wise).
    
    This loss treats RGB as the frozen teacher and other modalities as students.
    Students learn to match the teacher's output distribution using KL divergence.
    Compatible with existing InfoNCE infrastructure.
    
    Args:
        embed_dim (int): Dimensionality of embedding features.
        teacher_temp (float): Temperature for teacher softmax (lower = sharper).
        student_temp (float): Temperature for student softmax (higher = smoother).
        center_momentum (float): Momentum for teacher output centering.
        teacher_modality_idx (int): Index of teacher modality in feature_sets (default: 0 for RGB).
    """

    def __init__(self, embed_dim=512, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9, teacher_modality_idx=0):
        super(DINOStyleLoss, self).__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.teacher_modality_idx = teacher_modality_idx
        self.training_steps = 0  # Track training progress for adaptive momentum

        # Running center buffer (initialized as 0)
        self.register_buffer("center", torch.zeros(1, embed_dim))

    def forward(self, *feature_sets):
        """
        Compute DINO-style teacher-student loss.
        
        Args:
            *feature_sets: Variable number of feature tensors from different modalities.
                          First tensor (index 0) is assumed to be RGB (teacher).
        
        Returns:
            tuple: (total_loss, loss_dict) compatible with existing infrastructure.
        """
        num_modalities = len(feature_sets)
        
        if num_modalities < 2:
            raise ValueError("Need at least 2 modalities for teacher-student learning")
        
        if self.teacher_modality_idx >= num_modalities:
            raise ValueError(f"Teacher modality index {self.teacher_modality_idx} out of range")
        
        # Get teacher embeddings (typically RGB - frozen encoder)
        teacher_feats = feature_sets[self.teacher_modality_idx]
        teacher_feats = F.normalize(teacher_feats, dim=-1)
        
        # Ensure center is on the same device as teacher_feats
        center = self.center.to(teacher_feats.device)
        
        # Compute teacher softmax output (with centering and temperature)
        # Centering prevents collapse as in DINO paper
        teacher_logits = (teacher_feats - center) / self.teacher_temp
        teacher_probs = F.softmax(teacher_logits, dim=-1).detach()  # Stop gradient to teacher
        
        total_loss = 0.0
        num_students = 0
        loss_dict = {}
        
        # Define modality names for better logging
        modality_names = ['rgb', 'ir', 'depth', 'skeleton']
        teacher_name = modality_names[self.teacher_modality_idx] if self.teacher_modality_idx < len(modality_names) else f'modality_{self.teacher_modality_idx}'
        
        # Iterate through all student modalities
        for i, student_feats in enumerate(feature_sets):
            if i == self.teacher_modality_idx:
                continue  # Skip teacher modality (frozen RGB)
                
            student_name = modality_names[i] if i < len(modality_names) else f'modality_{i}'
            
            # Normalize student embeddings
            student_feats = F.normalize(student_feats, dim=-1)
            
            # Compute student log probabilities (higher temperature for smoother gradients)
            student_logits = student_feats / self.student_temp
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            
            # Cross-entropy loss: student learns to match teacher's soft targets
            # This is the core DINO loss - no negative sampling needed
            student_loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
            
            total_loss += student_loss
            num_students += 1
            
            # Store individual loss for logging (compatible with InfoNCE format)
            loss_dict[f'dino_{teacher_name}_to_{student_name}'] = student_loss.detach().cpu().item()
        
        if num_students == 0:
            raise ValueError("No student modalities found")
        
        # Update center (EMA) - only during training to prevent mode collapse
        # For frozen teacher, use student features for center computation to maintain dynamics
        if self.training:
            self.training_steps += 1
            
            # Adaptive momentum: start with faster updates (0.7), end with slower (0.95)
            # This maintains training signal strength as students approach teacher
            adaptive_momentum = 0.7 + 0.25 * min(1.0, self.training_steps / 1000.0)
            
            # Collect all student features for center computation
            all_student_features = []
            for i, student_feats in enumerate(feature_sets):
                if i != self.teacher_modality_idx:
                    normalized_student = F.normalize(student_feats, dim=-1)
                    all_student_features.append(normalized_student)
            
            if all_student_features:
                # Compute center from concatenated student features
                combined_students = torch.cat(all_student_features, dim=0)
                batch_center = torch.mean(combined_students, dim=0, keepdim=True)
                
                # Check if center is getting too close to teacher (preserve training signal)
                teacher_center_distance = torch.norm(teacher_feats.mean(dim=0, keepdim=True) - batch_center)
                min_distance = 0.05  # Minimum distance to maintain training signal
                
                if teacher_center_distance > min_distance:
                    # Ensure center update happens on the same device
                    self.center.data = self.center.data.to(teacher_feats.device) * adaptive_momentum + batch_center * (1 - adaptive_momentum)
                else:
                    # Skip center update if too close to teacher to maintain training dynamics
                    pass
            else:
                # Fallback: use teacher features if no students (shouldn't happen)
                batch_center = torch.mean(teacher_feats, dim=0, keepdim=True)
                self.center.data = self.center.data.to(teacher_feats.device) * adaptive_momentum + batch_center * (1 - adaptive_momentum)
            
        # Average loss over all students
        total_loss = total_loss / num_students
        loss_dict['dino_total_loss'] = total_loss.detach().cpu().item()
        
        # Additional monitoring for training dynamics
        if self.training:
            teacher_center_distance = torch.norm(teacher_feats.mean(dim=0, keepdim=True) - self.center).item()
            loss_dict['teacher_center_distance'] = teacher_center_distance
            loss_dict['adaptive_momentum'] = adaptive_momentum
            loss_dict['training_steps'] = self.training_steps
        
        return total_loss, loss_dict

    def update_temperatures(self, new_teacher_temp=None, new_student_temp=None):
        """
        Update temperature parameters during training.
        
        Args:
            new_teacher_temp (float, optional): New teacher temperature.
            new_student_temp (float, optional): New student temperature.
        """
        if new_teacher_temp is not None:
            self.teacher_temp = new_teacher_temp
        if new_student_temp is not None:
            self.student_temp = new_student_temp
    
    def reset_center(self):
        """Reset the center buffer to zeros."""
        self.center.zero_()
