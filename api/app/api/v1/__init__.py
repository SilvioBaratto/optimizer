"""
API Version 1 Package
=====================

This package contains all v1 API endpoints organized by domain.

File Structure
--------------
v1/
├── __init__.py      # This file - package documentation
├── router.py        # Main router - imports and includes all domain routers
└── <domain>.py      # Domain-specific endpoints (one file per resource)

Creating a New Route Module
---------------------------
Each domain module should follow this pattern:

```python
# api/v1/portfolio/portfolio.py
\"\"\"Portfolio management endpoints\"\"\"

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.portfolio.portfolio import PortfolioCreate, PortfolioResponse, PortfolioUpdate
from app.services.portfolio.portfolio_service import PortfolioService
from app.dependencies import get_current_user

router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


@router.get("/", response_model=list[PortfolioResponse])
async def list_portfolios(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    \"\"\"
    List all portfolios with pagination.

    - **skip**: Number of records to skip (default: 0)
    - **limit**: Maximum number of records to return (default: 100)
    \"\"\"
    service = PortfolioService(db)
    return await service.get_multi(skip=skip, limit=limit)


@router.post("/", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    portfolio_in: PortfolioCreate,
    db: AsyncSession = Depends(get_db),
):
    \"\"\"Create a new portfolio.\"\"\"
    service = PortfolioService(db)
    return await service.create(portfolio_in)


@router.get("/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
):
    \"\"\"Get a specific portfolio by ID.\"\"\"
    service = PortfolioService(db)
    portfolio = await service.get(portfolio_id)
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    return portfolio


@router.put("/{portfolio_id}", response_model=PortfolioResponse)
async def update_portfolio(
    portfolio_id: str,
    portfolio_in: PortfolioUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    \"\"\"Update an existing portfolio.\"\"\"
    service = PortfolioService(db)
    portfolio = await service.update(portfolio_id, portfolio_in)
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    return portfolio


@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_portfolio(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    \"\"\"Delete a portfolio.\"\"\"
    service = PortfolioService(db)
    success = await service.delete(portfolio_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
```

Router Registration (router.py)
-------------------------------
```python
from fastapi import APIRouter
from app.api.v1.portfolio.portfolio import router as portfolio_router
from app.api.v1.market_data.yfinance_data import router as market_data_router

api_router = APIRouter()
api_router.include_router(portfolio_router)
api_router.include_router(market_data_router)
```

Best Practices
--------------
1. **Use Dependency Injection**: Inject database sessions and services via Depends()
2. **Type Hints**: Always use Pydantic schemas for request/response types
3. **HTTP Status Codes**: Use appropriate status codes (201 for create, 204 for delete)
4. **Documentation**: Write docstrings - they appear in OpenAPI/Swagger docs
5. **Error Handling**: Raise HTTPException with meaningful error messages
6. **Tags**: Use tags=["Resource"] for OpenAPI organization
7. **Async**: Use async def for all endpoints when using async database
"""
