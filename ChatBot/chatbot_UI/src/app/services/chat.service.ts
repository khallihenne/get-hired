import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export type ChatType = 'text' | 'candidate_screening' | 'market_research' | 'error';
export interface ChatResponse {
  role: 'assistant';
  type: ChatType;
  content: any;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  constructor(private http: HttpClient) {}
  send(sessionId: string, message: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>('/api/chat', { sessionId, message });
  }
}
