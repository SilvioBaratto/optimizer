import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class MockFetchService {
  fetch<T>(data: T, delay = 300): Promise<T> {
    const jitter = 200 + Math.random() * 300; // 200-500ms
    const ms = Math.max(delay, jitter);
    return new Promise((resolve) => setTimeout(() => resolve(data), ms));
  }

  fetchWithError<T>(data: T, errorRate = 0.1, delay = 300): Promise<T> {
    const jitter = 200 + Math.random() * 300;
    const ms = Math.max(delay, jitter);
    return new Promise((resolve, reject) =>
      setTimeout(() => {
        if (Math.random() < errorRate) {
          reject(new Error('Failed to fetch data. Please try again.'));
        } else {
          resolve(data);
        }
      }, ms),
    );
  }
}
