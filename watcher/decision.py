"""Решение по нескольким кадрам подряд.

Один кадр с бликом или рукой в камере не должен будить Алексея. Тревога - только если
из последних `window` кадров не меньше `need` показали брак с уверенностью не ниже
`threshold` (по любому виду брака). После тревоги `cooldown_s` секунд тишины: одна
тревога на одну беду, а не сообщение каждые 15 секунд.

Кадры старше max_age_s выпадают из окна: после долгого перерыва (камера не отвечала,
печать на паузе) старые голоса не складываются с новыми.
"""
from collections import Counter, deque
from dataclasses import dataclass

from watcher.classes import FAILURES


@dataclass(frozen=True)
class Alarm:
    cls: str            # вид брака, за который больше всего голосов
    score: float        # средняя уверенность по голосам за этот вид
    votes: int
    window: int
    t: float


def frame_vote(probs, threshold):
    """Вид брака, за который голосует кадр, или None. probs: {класс: вероятность}."""
    best, best_p = None, -1.0
    for cls in FAILURES:
        p = probs.get(cls, 0.0)
        if p >= threshold and p > best_p:
            best, best_p = cls, p
    return best


class Decider:
    def __init__(self, threshold=0.7, window=4, need=3, cooldown_s=600, max_age_s=None):
        self.threshold = float(threshold)
        self.window = max(1, int(window))
        self.need = max(1, min(int(need), self.window))
        self.cooldown_s = float(cooldown_s)
        self.max_age_s = max_age_s
        self.frames = deque(maxlen=self.window)   # (t, vote, p)
        self.last_alarm_t = None

    def reset(self):
        """Новая печать или печать кончилась: голоса и пауза между тревогами - с нуля."""
        self.frames.clear()
        self.last_alarm_t = None

    def update(self, probs, t):
        vote = frame_vote(probs, self.threshold)
        self.frames.append((t, vote, probs.get(vote, 0.0) if vote else 0.0))
        if self.max_age_s is not None:
            while self.frames and t - self.frames[0][0] > self.max_age_s:
                self.frames.popleft()
        votes = [(v, p) for _, v, p in self.frames if v]
        if len(votes) < self.need:
            return None
        if self.last_alarm_t is not None and t - self.last_alarm_t < self.cooldown_s:
            return None
        count = Counter(v for v, _ in votes)
        top = max(count.values())
        tied = [c for c in FAILURES if count.get(c) == top]

        def mean_p(c):
            ps = [p for v, p in votes if v == c]
            return sum(ps) / len(ps)
        cls = max(tied, key=mean_p)
        self.last_alarm_t = t
        return Alarm(cls, mean_p(cls), len(votes), len(self.frames), t)
