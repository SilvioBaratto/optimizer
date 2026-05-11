"""
Services Package
================

This package contains business logic services that orchestrate operations
between repositories, external APIs, and other services.

File Structure
--------------
services/
├── __init__.py           # This file - exports service classes
├── base.py               # Base service class (optional)
└── <domain>_service.py   # Domain-specific services

Service Layer Responsibilities
------------------------------
- Business logic and validation
- Orchestrating repository operations
- External API integrations
- Event publishing
- Transaction management
- Caching strategies

Service Pattern
---------------
```python
# services/portfolio/portfolio_service.py
from typing import Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.portfolio.portfolio import Portfolio
from app.schemas.portfolio.portfolio import PortfolioCreate, PortfolioUpdate, PortfolioResponse
from app.repositories.portfolio.portfolio_repository import PortfolioRepository
from app.core.exceptions import NotFoundError, ConflictError


class PortfolioService:
    \"\"\"
    Service for portfolio-related business logic.

    Handles:
    - Portfolio CRUD with business rules
    - Name uniqueness validation
    - Benchmark assignment
    - Rebalancing triggers
    \"\"\"

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = PortfolioRepository(session)

    async def get(self, portfolio_id: str) -> Optional[Portfolio]:
        \"\"\"Get portfolio by ID.\"\"\"
        return await self.repository.get(portfolio_id)

    async def get_by_name(self, name: str) -> Optional[Portfolio]:
        \"\"\"Get portfolio by name.\"\"\"
        return await self.repository.get_by_name(name)

    async def get_multi(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> Sequence[Portfolio]:
        \"\"\"Get paginated list of portfolios.\"\"\"
        return await self.repository.get_multi(skip=skip, limit=limit)

    async def create(self, portfolio_in: PortfolioCreate) -> Portfolio:
        \"\"\"
        Create a new portfolio.

        Raises:
            ConflictError: If portfolio name already exists
        \"\"\"
        # Check name uniqueness
        existing = await self.repository.get_by_name(portfolio_in.name)
        if existing:
            raise ConflictError(f"Portfolio '{portfolio_in.name}' already exists")

        # Create portfolio
        portfolio_data = portfolio_in.model_dump()
        portfolio = Portfolio(**portfolio_data)
        self.session.add(portfolio)
        await self.session.commit()
        await self.session.refresh(portfolio)

        return portfolio

    async def update(
        self,
        portfolio_id: str,
        portfolio_in: PortfolioUpdate
    ) -> Portfolio:
        \"\"\"
        Update an existing portfolio.

        Raises:
            NotFoundError: If portfolio not found
            ConflictError: If new name already exists
        \"\"\"
        portfolio = await self.repository.get(portfolio_id)
        if not portfolio:
            raise NotFoundError(f"Portfolio {portfolio_id} not found")

        update_data = portfolio_in.model_dump(exclude_unset=True)

        # Check name uniqueness if changing
        if "name" in update_data and update_data["name"] != portfolio.name:
            existing = await self.repository.get_by_name(update_data["name"])
            if existing:
                raise ConflictError(f"Portfolio '{update_data['name']}' already exists")

        # Update fields
        for field, value in update_data.items():
            setattr(portfolio, field, value)

        await self.session.commit()
        await self.session.refresh(portfolio)

        return portfolio

    async def delete(self, portfolio_id: str) -> bool:
        \"\"\"
        Delete a portfolio.

        Raises:
            NotFoundError: If portfolio not found
        \"\"\"
        portfolio = await self.repository.get(portfolio_id)
        if not portfolio:
            raise NotFoundError(f"Portfolio {portfolio_id} not found")

        await self.session.delete(portfolio)
        await self.session.commit()
        return True

    async def assign_benchmark(
        self,
        portfolio_id: str,
        benchmark_ticker: str
    ) -> Portfolio:
        \"\"\"
        Assign a benchmark to a portfolio.

        Returns:
            Portfolio with updated benchmark

        Raises:
            NotFoundError: If portfolio not found
        \"\"\"
        portfolio = await self.repository.get(portfolio_id)
        if not portfolio:
            raise NotFoundError(f"Portfolio {portfolio_id} not found")

        portfolio.benchmark_ticker = benchmark_ticker
        await self.session.commit()
        await self.session.refresh(portfolio)
        return portfolio
```

Using Services in Routes
------------------------
```python
# api/v1/portfolio/portfolio.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.portfolio.portfolio_service import PortfolioService
from app.schemas.portfolio.portfolio import PortfolioCreate, PortfolioResponse
from app.core.exceptions import NotFoundError, ConflictError

router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


@router.post("/", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    portfolio_in: PortfolioCreate,
    db: AsyncSession = Depends(get_db),
):
    service = PortfolioService(db)
    try:
        return await service.create(portfolio_in)
    except ConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
```

Service with External APIs
--------------------------
```python
# services/payment_service.py
import httpx
from app.config import settings


class PaymentService:
    \"\"\"Service for payment processing.\"\"\"

    def __init__(self, session: AsyncSession):
        self.session = session
        self.api_key = settings.stripe_api_key
        self.base_url = "https://api.stripe.com/v1"

    async def create_payment_intent(
        self,
        amount: int,
        currency: str = "usd"
    ) -> dict:
        \"\"\"Create a Stripe payment intent.\"\"\"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/payment_intents",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data={"amount": amount, "currency": currency}
            )
            response.raise_for_status()
            return response.json()
```

Best Practices
--------------
1. **Business Logic Only**: Keep HTTP/API concerns in routes
2. **Transaction Boundaries**: Services control commit/rollback
3. **Repository Composition**: Services can use multiple repositories
4. **Exception Handling**: Raise domain exceptions, not HTTP exceptions
5. **Dependency Injection**: Inject session/repos via constructor
6. **Async Operations**: Use async/await consistently
7. **Single Responsibility**: One service per domain/aggregate
8. **Testing**: Services should be easily testable with mocked repos
9. **Caching**: Implement caching strategies in services
10. **Logging**: Log important business operations
"""

# Import and export commonly used services here:
from app.services.jobs.background_job import (
    BackgroundJobService,
    JobAlreadyRunningError,
)
from app.services.jobs.scheduler import create_scheduler
from app.services.universe.trading212 import (
    BuildProgress,
    BuildResult,
    Trading212Client,
    UniverseBuilder,
    UniverseBuilderConfig,
    YFinanceTickerMapper,
)

__all__ = [
    "BackgroundJobService",
    "BuildProgress",
    "BuildResult",
    "JobAlreadyRunningError",
    "Trading212Client",
    "UniverseBuilder",
    "UniverseBuilderConfig",
    "YFinanceTickerMapper",
    "create_scheduler",
]
