import type { AppEnvironment } from './environment.model';

export const environment: AppEnvironment = {
    // Use relative URL - nginx will proxy /api requests to the backend
    apiUrl: '/api/v1/',
};
