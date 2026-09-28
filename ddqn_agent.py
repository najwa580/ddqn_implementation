import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# =========================================================
# RÉSEAU Q — architecture simple (identique au DQN pur)
#   Une seule sortie directe vers les Q-valeurs
#   Q(s, a) pour chaque action possible
# =========================================================
class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),   # sortie directe → une Q-valeur par action
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# REPLAY BUFFER UNIFORME
#   Toutes les transitions ont la même probabilité
#   d'être rejouées (pas de priorité)
# =========================================================
class ReplayBuffer:

    def __init__(self, capacity=60000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================
# DOUBLE DQN AGENT — version stabilisée
#   - Double DQN     → q_net CHOISIT l'action suivante,
#                       target_net l'ÉVALUE seulement
#   - SOFT UPDATE    → target_net suit q_net progressivement
#                       (τ à CHAQUE step, plus de saut brutal
#                        toutes les 100 étapes)
#   - epsilon_min bas → la politique finale est quasi 100%
#                        greedy, pour un plateau de reward stable
#   - lr réduit       → moins de bruit dans la loss
#   - Pas de Dueling     → réseau linéaire simple
#   - Pas de Prioritized → replay uniforme
# =========================================================
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size

        # Hyperparamètres
        self.gamma       = 0.99
        self.batch_size  = 128
        self.lr          = 1e-4          # ↓ réduit (était 3e-4) — moins de bruit dans la loss

        # Soft update (Polyak averaging) — remplace le hard update périodique
        self.tau = 0.005                 # taux de suivi du target_net à chaque step
        self.step_count = 0

        # Exploration epsilon-greedy
        self.epsilon     = 1.0
        self.epsilon_min = 0.01          # ↓ réduit (était 0.05) — politique quasi greedy en fin d'entraînement
        self.epsilon_dec = 0.995         # multiplicatif par épisode

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DoubleDQN] Device : {self.device}")

        # Réseaux
        self.q_net      = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimiseur
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        # Loss Huber (robuste aux grandes récompenses)
        self.loss_fn = nn.SmoothL1Loss()

        # Mémoire uniforme
        self.memory = ReplayBuffer(capacity=60000)

    # --------------------------------------------------------
    # ACTION — epsilon-greedy
    # --------------------------------------------------------
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(state_t)
        return int(torch.argmax(q).item())

    # --------------------------------------------------------
    # STOCKER EXPÉRIENCE
    # --------------------------------------------------------
    def remember(self, s, a, r, s2, done):
        self.memory.add(s, a, r, s2, done)

    # --------------------------------------------------------
    # ENTRAÎNEMENT — DOUBLE DQN + SOFT UPDATE
    #   1) q_net choisit l'action suivante      (argmax)
    #   2) target_net évalue cette action précise (gather)
    #   3) target_net est mis à jour EN DOUCEUR à chaque step
    #      (au lieu d'un copiage brutal toutes les 100 étapes)
    # --------------------------------------------------------
    def replay(self):
        if len(self.memory) < self.batch_size * 2:
            return 0.0

        s, a, r, s2, d = self.memory.sample(self.batch_size)

        # Tenseurs
        S  = torch.FloatTensor(s).to(self.device)
        A  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        R  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        S2 = torch.FloatTensor(s2).to(self.device)
        D  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # Q(s, a) — valeurs courantes
        Q_current = self.q_net(S).gather(1, A)

        # ---- DOUBLE DQN ----
        with torch.no_grad():
            # 1) SÉLECTION : quelle action le réseau "actuel" q_net
            #    choisirait-il dans l'état suivant S2 ?
            next_actions = self.q_net(S2).argmax(dim=1, keepdim=True)

            # 2) ÉVALUATION : on demande au target_net la Q-valeur
            #    de CETTE action précise (pas son propre max)
            Q_next = self.target_net(S2).gather(1, next_actions)

            Q_target = R + (1 - D) * self.gamma * Q_next
        # ---------------------

        # Loss et backprop
        loss = self.loss_fn(Q_current, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — évite l'explosion du gradient
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ---- SOFT UPDATE (Polyak averaging) ----
        # target = τ·q_net + (1-τ)·target, appliqué à CHAQUE step
        # → le target_net "suit" q_net en douceur, plus de sauts brutaux
        self._soft_update()
        self.step_count += 1

        return float(loss.item())

    # --------------------------------------------------------
    # SOFT UPDATE — mise à jour progressive du target network
    # --------------------------------------------------------
    def _soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    # --------------------------------------------------------
    # DECAY EPSILON — à appeler en fin d'épisode
    # --------------------------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_dec)

    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------
    def save(self, path):
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "step_count": self.step_count,
        }, path)
        print(f"[DoubleDQN] Modèle sauvegardé → {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon    = checkpoint.get("epsilon",    self.epsilon_min)
        self.step_count = checkpoint.get("step_count", 0)
        self.target_net.eval()
        print(f"[DoubleDQN] Modèle chargé ← {path} | ε={self.epsilon:.3f}")
