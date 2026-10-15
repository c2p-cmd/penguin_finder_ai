import type { Game } from './game';

/**
 * A number whose changed characters animate: horizontal tickers slide the new digit in from the left
 * while the old one leaves to the right (score); vertical ones roll upward (clock).
 */
const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)');

class Ticker {
  private slots: HTMLElement[] = [];
  private text = '';
  constructor(private host: HTMLElement, private mode: 'horizontal' | 'vertical') {
    host.classList.add('ticker');
  }

  set(text: string) {
    if (text === this.text) return;
    const prev = this.text;
    this.text = text;
    // align right so units stay put when the length changes
    while (this.slots.length < text.length) {
      const s = document.createElement('span');
      s.className = 'slot';
      this.host.insertBefore(s, this.host.firstChild);
      this.slots.unshift(s);
    }
    while (this.slots.length > text.length) this.slots.shift()!.remove();
    const off = text.length - prev.length;
    this.slots.forEach((slot, i) => {
      const ch = text[i];
      const was = prev[i - off];
      if (slot.dataset.ch === ch) return;
      // drop any glyph still animating out from a previous change, then swap the current one
      slot.querySelectorAll('.glyph:not(.cur)').forEach((g) => g.remove());
      const old = slot.querySelector('.glyph.cur') as HTMLElement | null;
      const fresh = document.createElement('span');
      fresh.className = 'glyph cur';
      fresh.textContent = ch;
      slot.dataset.ch = ch;
      slot.appendChild(fresh);
      // tenths of a second change ten times a second: keep them still; honour reduced motion
      const still = REDUCED_MOTION.matches || (this.mode === 'vertical' && i === text.length - 1);
      const animatable = !still && was !== undefined && /\d/.test(ch) && /\d/.test(was);
      if (old && animatable) {
        const [from, leave] = this.mode === 'horizontal'
          ? ['translateX(-100%)', 'translateX(100%)']
          : ['translateY(100%)', 'translateY(-100%)'];
        old.classList.remove('cur');
        old.style.position = 'absolute';
        old.style.left = '0';
        old.style.right = '0';
        fresh.animate([{ transform: from, opacity: 0 }, { transform: 'none', opacity: 1 }], { duration: 170, easing: 'ease-out' });
        const out = old.animate([{ transform: 'none', opacity: 1 }, { transform: leave, opacity: 0 }], { duration: 170, easing: 'ease-out' });
        out.onfinish = () => old.remove();
      } else {
        old?.remove();
      }
    });
  }
}

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export class Hud {
  private time = new Ticker(el('time'), 'vertical');
  private score = new Ticker(el('score'), 'horizontal');
  private shownModels = '';
  private lastState = '';
  private lastLevel = -1;
  private lastProbs: unknown = null;

  constructor(private game: Game) {
    el<HTMLSelectElement>('player').addEventListener('change', (e) => {
      const v = (e.target as HTMLSelectElement).value;
      (e.target as HTMLSelectElement).blur();
      void this.game.setPlayer(v);
    });
    // on-screen d-pad (touch)
    document.querySelectorAll<HTMLButtonElement>('#dpad button').forEach((b) => {
      const code = b.dataset.key!;
      const set = (down: boolean) => (e: Event) => { e.preventDefault(); this.game.setKey(code, down); };
      b.addEventListener('pointerdown', set(true));
      for (const t of ['pointerup', 'pointercancel', 'pointerleave']) b.addEventListener(t, set(false));
    });
    el('btn-primary').addEventListener('click', () => {
      if (this.game.state === 'won') void this.game.nextLevel();
      else this.game.reset();
    });
  }

  private clock(t: number) {
    const m = Math.floor(t / 60);
    const s = t - m * 60;
    return `${m}:${s < 10 ? '0' : ''}${s.toFixed(1)}`;
  }

  update(fps: number, backend: string) {
    const g = this.game;
    this.time.set(this.clock(g.shownTime));
    this.score.set(String(g.score));
    el('best').textContent = String(g.bestScore);
    el('lvl-label').textContent = `Level ${g.levelIndex + 1} of ${g.levelCount}`;
    el('lvl-name').textContent = g.levelName;
    el('lvl-tag').textContent = g.levelTagline;
    el('loading').hidden = !g.loading;

    // player picker
    const key = g.models.map((m) => m.id).join('|') + '#' + g.player;
    if (key !== this.shownModels) {
      this.shownModels = key;
      const sel = el<HTMLSelectElement>('player');
      sel.innerHTML = '';
      const add = (parent: HTMLElement, value: string, label: string) => {
        const o = document.createElement('option');
        o.value = value;
        o.textContent = label;
        parent.appendChild(o);
      };
      add(sel, '', 'You');
      for (const [group, title] of [['browser', 'In your browser'], ['ollama', 'Local Ollama']] as const) {
        const items = g.models.filter((m) => m.group === group);
        if (!items.length) continue;
        const og = document.createElement('optgroup');
        og.label = title;
        for (const m of items) add(og, m.id, m.label);
        sel.appendChild(og);
      }
      sel.value = g.player;
    }

    // AI status + direction bars
    const st = el('ai-status');
    st.hidden = !g.aiStatus;
    st.textContent = g.aiThinking && !g.aiLoading ? 'thinking…' : g.aiStatus;
    st.classList.toggle('error', g.aiError !== null);
    st.classList.toggle('busy', g.aiThinking || g.aiLoading);
    const pctMatch = /(\d+)%/.exec(g.aiStatus);
    st.classList.toggle('progress', g.aiLoading && !!pctMatch);
    st.style.setProperty('--p', `${pctMatch ? pctMatch[1] : 0}%`);
    const bars = el('ai-bars');
    bars.hidden = !(g.player && g.aiProbs.length);
    if (g.aiProbs !== this.lastProbs) {
      this.lastProbs = g.aiProbs;
      bars.innerHTML = g.aiProbs
        .map((d) => `<div class="bar"><div class="track"><div class="fill" style="height:${Math.max(2, 34 * d.p)}px"></div></div><span>${d.label}</span></div>`)
        .join('');
    }

    // end card
    const over = g.state !== 'playing';
    const card = el('overlay');
    card.hidden = !over;
    if (over && this.lastState !== g.state) {
      el('card-title').textContent = g.state === 'won' ? 'You found the penguin! 🐧' : 'Caught by a ghost 👻';
      el('btn-primary').textContent = g.state === 'won' ? (g.isLastLevel ? 'Start over, harder' : 'Next level') : 'Try again';
      // announce the result to screen readers and put focus on the one action
      el('sr-status').textContent = `${el('card-title').textContent}. Score ${g.score}.`;
      el<HTMLButtonElement>('btn-primary').focus({ preventScroll: true });
    }
    if (this.lastLevel !== g.levelIndex) {
      this.lastLevel = g.levelIndex;
      if (!over) el('sr-status').textContent = `Level ${g.levelIndex + 1}: ${g.levelName}. ${g.levelTagline}`;
    }
    if (over) {
      el('card-score').textContent = `Score ${g.score}`;
      el('card-bonus').textContent = g.state === 'won' ? `includes +${g.bonus} for the catch` : '';
      el('card-time').textContent = this.clock(g.shownTime);
    }
    this.lastState = g.state;
    el('hint').hidden = over || !!g.player;
    el('fps').textContent = `${backend} · ${fps.toFixed(0)} fps`;
  }
}
