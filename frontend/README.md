# Portfolio Optimizer Frontend

Angular 20 web application for the quantitative portfolio optimization system with Supabase authentication.

## Current Features

### Technology Stack
- **Angular 20** with standalone components
- **Tailwind CSS 4.1** for styling
- **Supabase client** (@supabase/supabase-js 2.58.0) for authentication and API calls
- **RxJS 7.8** for reactive programming
- **TypeScript 5.9** with strict mode

### Architecture
- Standalone components (no NgModules)
- Signal-based state management
- OnPush change detection strategy
- Lazy-loaded feature routes
- Reactive forms

---

This project was generated using [Angular CLI](https://github.com/angular/angular-cli) version 20.3.3.

## Development server

To start a local development server, run:

```bash
ng serve
```

Once the server is running, open your browser and navigate to `http://localhost:4200/`. The application will automatically reload whenever you modify any of the source files.

## Code scaffolding

Angular CLI includes powerful code scaffolding tools. To generate a new component, run:

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```

## Building

To build the project run:

```bash
ng build
```

This will compile your project and store the build artifacts in the `dist/` directory. By default, the production build optimizes your application for performance and speed.

## Running unit tests

To execute unit tests with the [Karma](https://karma-runner.github.io) test runner, use the following command:

```bash
ng test
```

## Running end-to-end tests

For end-to-end (e2e) testing, run:

```bash
ng e2e
```

Angular CLI does not come with an end-to-end testing framework by default. You can choose one that suits your needs.

## Additional Resources

For more information on using the Angular CLI, including detailed command references, visit the [Angular CLI Overview and Command Reference](https://angular.dev/tools/cli) page.

---

## Project-Specific Information

### Supabase Integration

The application uses Supabase for authentication and backend communication.

**Environment Configuration:**
Configure Supabase credentials in your environment or directly in the app:

```typescript
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  'YOUR_SUPABASE_URL',
  'YOUR_SUPABASE_ANON_KEY'
);
```

### Angular Best Practices (Project Standards)

**See `.claude/CLAUDE.md` for comprehensive Angular patterns followed in this project:**

- Always use standalone components (default in Angular 20)
- Use `input()` and `output()` functions instead of decorators
- Use signals with `computed()` for derived state
- Set `changeDetection: ChangeDetectionStrategy.OnPush`
- Use native control flow (`@if`, `@for`, `@switch`)
- Use `inject()` function instead of constructor injection
- Use `class` and `style` bindings instead of `ngClass`/`ngStyle`

### Backend Integration

The Angular frontend connects to the FastAPI backend located in `../optimizer/`.

**Backend endpoints:** `http://localhost:8000/api/v1/`

**Key API integrations:**
- User authentication via Supabase
- Stock signals and portfolio data
- Macro regime classifications
- Real-time market indicators

### Project Structure

```
src/
├── app/                    # Application root
│   ├── components/        # Reusable components
│   ├── pages/            # Route components
│   ├── services/         # Business logic & API calls
│   ├── models/           # TypeScript interfaces
│   └── guards/           # Route guards
├── assets/               # Static assets
└── styles/              # Global styles (Tailwind)
```

### Development Workflow

1. **Start backend:** `cd ../optimizer && uvicorn app.main:app --reload`
2. **Start frontend:** `ng serve`
3. **Access app:** `http://localhost:4200`

### Documentation

- See `.claude/CLAUDE.md` for Angular best practices and coding standards
- Backend API documentation: `../optimizer/README.md`
- Main project documentation: `../CLAUDE.md`
