import pytest
from app.database import SessionLocal, Base, engine


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_wallet():
    """Provide a test wallet address."""
    return "0xTestWallet123456789"