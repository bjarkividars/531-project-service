from fastapi import APIRouter

router = APIRouter(
    prefix="/items",
    tags=["items"]
)

@router.get("/")
async def get_items():
    return {"items": ["item1", "item2"]}

@router.get("/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id} 