"""Test endpoints for API demonstration"""

from typing import Dict
import uuid

from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    utc_now,
    MessageResponse,
    EchoRequest,
    EchoResponse,
    ItemCreate,
    ItemUpdate,
    ItemResponse,
    ItemListResponse,
    SuccessResponse,
)

router = APIRouter(prefix="/test", tags=["Test"])

# ===========================
# In-Memory Storage for Demo
# ===========================

# Simple in-memory storage for items (demo purposes)
_items_store: Dict[str, dict] = {}


# ===========================
# Health/Echo Endpoints
# ===========================

@router.get("/ping", response_model=MessageResponse)
async def ping():
    """
    Simple ping endpoint to verify API is responding.

    Returns:
        {"message": "pong"}
    """
    return {"message": "pong"}


@router.get("/echo/{message}", response_model=EchoResponse)
async def echo_get(message: str):
    """
    Echo back the message provided in the path.

    Args:
        message: The message to echo back

    Returns:
        Echo response with the message and timestamp
    """
    return EchoResponse(
        echo=message,
        received_at=utc_now(),
        metadata=None
    )


@router.post("/echo", response_model=EchoResponse)
async def echo_post(request: EchoRequest):
    """
    Echo back the message provided in the request body.

    Args:
        request: EchoRequest containing the message

    Returns:
        Echo response with the message and timestamp
    """
    return EchoResponse(
        echo=request.message,
        received_at=utc_now(),
        metadata=request.metadata
    )


# ===========================
# CRUD Endpoints for Items
# ===========================

@router.get("/items", response_model=ItemListResponse)
async def list_items():
    """
    List all items in the in-memory store.

    Returns:
        List of all items with total count
    """
    items = [
        ItemResponse(
            id=item_id,
            name=item["name"],
            description=item.get("description"),
            price=item.get("price"),
            is_active=item.get("is_active", True),
            created_at=item["created_at"],
            updated_at=item["updated_at"],
        )
        for item_id, item in _items_store.items()
    ]
    return ItemListResponse(items=items, total=len(items))


@router.post("/items", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemCreate):
    """
    Create a new item in the in-memory store.

    Args:
        item: ItemCreate schema with item data

    Returns:
        Created item with generated ID and timestamps
    """
    item_id = str(uuid.uuid4())
    now = utc_now()

    item_data = {
        "name": item.name,
        "description": item.description,
        "price": item.price,
        "is_active": item.is_active,
        "created_at": now,
        "updated_at": now,
    }

    _items_store[item_id] = item_data

    return ItemResponse(
        id=item_id,
        name=item_data["name"],
        description=item_data["description"],
        price=item_data["price"],
        is_active=item_data["is_active"],
        created_at=item_data["created_at"],
        updated_at=item_data["updated_at"],
    )


@router.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: str):
    """
    Get a specific item by ID.

    Args:
        item_id: The ID of the item to retrieve

    Returns:
        The item if found

    Raises:
        HTTPException: 404 if item not found
    """
    if item_id not in _items_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with ID '{item_id}' not found"
        )

    item = _items_store[item_id]
    return ItemResponse(
        id=item_id,
        name=item["name"],
        description=item.get("description"),
        price=item.get("price"),
        is_active=item.get("is_active", True),
        created_at=item["created_at"],
        updated_at=item["updated_at"],
    )


@router.put("/items/{item_id}", response_model=ItemResponse)
async def update_item(item_id: str, item_update: ItemUpdate):
    """
    Update an existing item.

    Args:
        item_id: The ID of the item to update
        item_update: ItemUpdate schema with fields to update

    Returns:
        Updated item

    Raises:
        HTTPException: 404 if item not found
    """
    if item_id not in _items_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with ID '{item_id}' not found"
        )

    item = _items_store[item_id]

    # Update only provided fields
    update_data = item_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        item[field] = value

    item["updated_at"] = utc_now()

    return ItemResponse(
        id=item_id,
        name=item["name"],
        description=item.get("description"),
        price=item.get("price"),
        is_active=item.get("is_active", True),
        created_at=item["created_at"],
        updated_at=item["updated_at"],
    )


@router.delete("/items/{item_id}", response_model=SuccessResponse)
async def delete_item(item_id: str):
    """
    Delete an item by ID.

    Args:
        item_id: The ID of the item to delete

    Returns:
        Success response

    Raises:
        HTTPException: 404 if item not found
    """
    if item_id not in _items_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with ID '{item_id}' not found"
        )

    del _items_store[item_id]

    return SuccessResponse(
        success=True,
        message=f"Item '{item_id}' deleted successfully"
    )
