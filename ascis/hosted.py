"""Operations website and storefront data endpoints for hosted demos."""

from ascis.web import app, book, book_cart, catalog, quote_cart

app.add_api_route("/api/catalog", catalog, methods=["GET"], include_in_schema=False)
app.add_api_route("/api/bookings", book, methods=["POST"], include_in_schema=False)
app.add_api_route("/api/booking-carts", book_cart, methods=["POST"], include_in_schema=False)
app.add_api_route("/api/booking-carts/quote", quote_cart, methods=["POST"], include_in_schema=False)


@app.get("/healthz", include_in_schema=False)
def health():
    return {"status": "ready"}
