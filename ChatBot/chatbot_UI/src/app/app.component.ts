import { Component } from '@angular/core';
import { ChatService, ChatResponse } from './services/chat.service';

type Msg = { role: 'user' | 'assistant', content: string, raw?: ChatResponse|null };

function makeSessionId() {
  return 's-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  sessionId = makeSessionId();
  input = '';
  loading = false;

  messages: Msg[] = [
    { role: 'assistant', content: 'Bienvenue 👋' },
    { role: 'assistant', content: 'Comment puis-je vous aider ?' }
  ];

  constructor(private chat: ChatService) {}

  send() {
    const text = this.input.trim();
    if (!text || this.loading) return;

    this.messages.push({ role: 'user', content: text });
    this.input = '';
    this.loading = true;

    this.chat.send(this.sessionId, text).subscribe({
      next: (res) => {
        this.messages.push({ role: 'assistant', content: this.formatReply(res), raw: res });
        this.loading = false;
      },
      error: (err) => {
        console.error('HTTP /api/chat error', err);
        this.messages.push({ role: 'assistant', content: "⚠️ API indisponible (port 8000). Vérifie le backend." });
        this.loading = false;
      }
    });
  }

  private formatReply(res: ChatResponse): string {
    // Simple fallback text if no specialized renderer below
    if (res.type === 'text' && typeof res.content?.message === 'string') {
      return res.content.message;
    }
    if (res.type === 'error') {
      return typeof res.content?.message === 'string'
        ? res.content.message
        : 'Une erreur est survenue.';
    }
    // For candidate_screening and market_research, the HTML template renders cards/sections using last assistant msg raw data.
    // Here we return a short label; the pretty render happens in the template via res.raw.
    if (res.type === 'candidate_screening') return '✅ Résultats candidats reçus.';
    if (res.type === 'market_research')   return '✅ Rapport marché reçu.';
    return typeof res.content === 'string' ? res.content : JSON.stringify(res.content);
  }

  lastAssistantRaw(): ChatResponse | null {
    // Get the most recent assistant message that has a raw payload
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const m = this.messages[i];
      if (m.role === 'assistant' && m.raw) return m.raw;
    }
    return null;
  }

  // Helpers for template type-guards
  isCand(res: ChatResponse|null): boolean { return !!res && res.type === 'candidate_screening'; }
  isMarket(res: ChatResponse|null): boolean { return !!res && res.type === 'market_research'; }
}
